import os
import json
import logging
from typing import Optional

import psycopg2
from psycopg2.extras import RealDictCursor
from app.schemas.reasoning import ReasoningResponse

logger = logging.getLogger(__name__)

DB_URL = os.getenv("DATABASE_URL", "postgresql://reasonops:reasonops_pass@db:5432/reasonops")


class TelemetryLogger:
    def __init__(self):
        self._schema_ensured = False

    def _get_connection(self):
        conn = psycopg2.connect(DB_URL)
        if not self._schema_ensured:
            self._ensure_schema(conn)
        return conn

    def _ensure_schema(self, conn):
        """Auto-apply v0.5 schema migration for existing databases."""
        try:
            with conn.cursor() as cur:
                # Check if new columns exist, if not add them
                cur.execute("""
                    SELECT column_name FROM information_schema.columns
                    WHERE table_name = 'requests' AND column_name = 'verification_status'
                """)
                if not cur.fetchone():
                    logger.info("Applying v0.5 schema migration...")
                    cur.execute("""
                        ALTER TABLE requests
                            ADD COLUMN IF NOT EXISTS verification_status VARCHAR(10) DEFAULT NULL,
                            ADD COLUMN IF NOT EXISTS verification_confidence FLOAT DEFAULT NULL,
                            ADD COLUMN IF NOT EXISTS difficulty_level VARCHAR(10) DEFAULT NULL,
                            ADD COLUMN IF NOT EXISTS retry_used BOOLEAN DEFAULT FALSE;
                    """)
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_requests_verification_status ON requests(verification_status)")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_requests_difficulty_level ON requests(difficulty_level)")
                    conn.commit()
                    logger.info("v0.5 schema migration applied successfully")
            self._schema_ensured = True
        except Exception as e:
            logger.error(f"Schema migration check failed: {e}")
            self._schema_ensured = True  # Don't retry endlessly

    def log_trace(self, query: str, response: ReasoningResponse):
        """Logs request and reasoning steps to PostgreSQL with enriched verification data."""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO requests (
                            query, strategy_selected, hallucination_risk, confidence_score,
                            trust_score, tokens_used, latency_ms, final_answer,
                            verification_status, verification_confidence, difficulty_level, retry_used
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id
                    """, (
                        query,
                        response.strategy_selected.value,
                        response.hallucination_risk,
                        response.confidence_score,
                        response.trust_score.aggregate_score,
                        response.tokens_used,
                        response.latency_ms,
                        response.final_answer,
                        response.verification_status.value if response.verification_status else None,
                        response.verification_confidence,
                        response.difficulty_level.value if response.difficulty_level else None,
                        response.retry_used,
                    ))
                    request_id = cur.fetchone()[0]

                    for step in response.reasoning_steps:
                        cur.execute("""
                            INSERT INTO reasoning_steps (
                                request_id, step_index, content, entropy, flagged, assumptions
                            ) VALUES (%s, %s, %s, %s, %s, %s)
                        """, (
                            request_id,
                            step.step_index,
                            step.content,
                            step.entropy,
                            step.flagged,
                            json.dumps(step.assumptions),
                        ))
                conn.commit()
            logger.info(f"Telemetry logged successfully for request {request_id}")
        except Exception as e:
            logger.error(f"Failed to log telemetry: {e}")

    def get_dashboard_metrics(self) -> dict:
        """Fetches aggregated metrics for the Analytics Dashboard."""
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    # Strategy Distribution
                    cur.execute("""
                        SELECT strategy_selected as name, COUNT(*) as value 
                        FROM requests GROUP BY strategy_selected
                    """)
                    strategies = cur.fetchall()

                    # System Averages
                    cur.execute("""
                        SELECT 
                            ROUND(AVG(latency_ms), 0) as avg_latency,
                            ROUND(AVG(tokens_used), 0) as avg_tokens,
                            ROUND(AVG(trust_score)::numeric, 3) as avg_trust,
                            ROUND(AVG(hallucination_risk)::numeric, 3) as avg_risk,
                            COUNT(*) as total_requests
                        FROM requests
                    """)
                    averages = cur.fetchone()

                    # Timeline
                    cur.execute("""
                        SELECT DATE(created_at) as date, 
                               COUNT(*) as total,
                               SUM(CASE WHEN trust_score < 0.7 THEN 1 ELSE 0 END) as low_trust_count
                        FROM requests
                        GROUP BY DATE(created_at)
                        ORDER BY date DESC LIMIT 7
                    """)
                    timeline = cur.fetchall()

                    # Recent Traces
                    cur.execute("""
                        SELECT id, query, strategy_selected, trust_score, latency_ms,
                               verification_status, verification_confidence, difficulty_level,
                               created_at
                        FROM requests
                        ORDER BY created_at DESC LIMIT 10
                    """)
                    recent_traces = cur.fetchall()

                    # Verification distribution
                    cur.execute("""
                        SELECT verification_status as name, COUNT(*) as value
                        FROM requests
                        WHERE verification_status IS NOT NULL
                        GROUP BY verification_status
                    """)
                    verification_dist = cur.fetchall()

                    # Difficulty distribution
                    cur.execute("""
                        SELECT difficulty_level as name, COUNT(*) as value
                        FROM requests
                        WHERE difficulty_level IS NOT NULL
                        GROUP BY difficulty_level
                    """)
                    difficulty_dist = cur.fetchall()

            return {
                "strategies": strategies,
                "averages": averages,
                "timeline": timeline,
                "recent_traces": recent_traces,
                "verification_distribution": verification_dist,
                "difficulty_distribution": difficulty_dist,
            }
        except Exception as e:
            logger.error(f"Failed to fetch dashboard metrics: {e}")
            return {"error": str(e)}

    def get_trace_detail(self, trace_id: str) -> dict:
        """Fetches a single trace with all its reasoning steps."""
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT id, query, strategy_selected, hallucination_risk,
                               confidence_score, trust_score, tokens_used, latency_ms,
                               final_answer, verification_status, verification_confidence,
                               difficulty_level, retry_used, created_at
                        FROM requests WHERE id = %s
                    """, (trace_id,))
                    request_row = cur.fetchone()
                    if not request_row:
                        return {"error": "Trace not found"}

                    cur.execute("""
                        SELECT step_index, content, assumptions, flagged
                        FROM reasoning_steps
                        WHERE request_id = %s
                        ORDER BY step_index ASC
                    """, (trace_id,))
                    steps = cur.fetchall()

            return {
                "request": dict(request_row),
                "reasoning_steps": [dict(s) for s in steps],
            }
        except Exception as e:
            logger.error(f"Failed to fetch trace detail: {e}")
            return {"error": str(e)}

    def list_traces(
        self, search: Optional[str] = None, limit: int = 20, offset: int = 0
    ) -> dict:
        """List traces with optional search/filter support."""
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    if search:
                        cur.execute("""
                            SELECT id, query, strategy_selected, trust_score, latency_ms,
                                   verification_status, verification_confidence, difficulty_level,
                                   created_at
                            FROM requests
                            WHERE query ILIKE %s
                            ORDER BY created_at DESC
                            LIMIT %s OFFSET %s
                        """, (f"%{search}%", limit, offset))
                    else:
                        cur.execute("""
                            SELECT id, query, strategy_selected, trust_score, latency_ms,
                                   verification_status, verification_confidence, difficulty_level,
                                   created_at
                            FROM requests
                            ORDER BY created_at DESC
                            LIMIT %s OFFSET %s
                        """, (limit, offset))

                    traces = cur.fetchall()

                    # Get total count
                    if search:
                        cur.execute("SELECT COUNT(*) FROM requests WHERE query ILIKE %s", (f"%{search}%",))
                    else:
                        cur.execute("SELECT COUNT(*) FROM requests")
                    total = cur.fetchone()["count"]

            return {
                "traces": [dict(t) for t in traces],
                "total": total,
                "limit": limit,
                "offset": offset,
            }
        except Exception as e:
            logger.error(f"Failed to list traces: {e}")
            return {"error": str(e)}

    def delete_trace(self, trace_id: str) -> dict:
        """Delete a single trace and its reasoning steps."""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    # Steps are CASCADE deleted via FK
                    cur.execute("DELETE FROM requests WHERE id = %s RETURNING id", (trace_id,))
                    deleted = cur.fetchone()
                    if not deleted:
                        return {"error": "Trace not found"}
                conn.commit()
            return {"status": "deleted", "trace_id": trace_id}
        except Exception as e:
            logger.error(f"Failed to delete trace: {e}")
            return {"error": str(e)}

    def delete_all_traces(self) -> dict:
        """Delete all traces and reasoning steps."""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM reasoning_steps")
                    cur.execute("DELETE FROM requests")
                conn.commit()
            return {"status": "cleared", "message": "All traces deleted"}
        except Exception as e:
            logger.error(f"Failed to clear traces: {e}")
            return {"error": str(e)}