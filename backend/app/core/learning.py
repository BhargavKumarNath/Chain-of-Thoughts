"""
learning.py — Contextual Bandit Policy Learner

Analyzes historical telemetry, computes reward function:
    R = Trust - λ_cost * NormCost - λ_latency * NormLatency
Updates routing decision surface and provides data for:
- Reward curves over time
- A/B test results (Always-CoT vs Adaptive vs Hybrid)
- Token cost savings vs always-CoT baseline
- Latency distribution by strategy
- Current policy weights / decision surface
"""

import os
import json
import logging
import math
from datetime import datetime
from typing import Dict, Any, List, Optional

import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)

DB_URL = os.getenv("DATABASE_URL", "postgresql://reasonops:reasonops_pass@db:5432/reasonops")
POLICY_PATH = "/app/models/policy_weights.json"

# Reward function hyperparameters
LAMBDA_COST = 0.2
LAMBDA_LATENCY = 0.3

# Token cost normalization: tokens / MAX_TOKENS → [0, 1]
MAX_TOKENS = 2000
# Latency normalization: latency_ms / MAX_LATENCY → [0, 1]
MAX_LATENCY = 30000

# A/B simulation: always-CoT uses LONG_COT for everything
ALWAYS_COT_STRATEGY = "LONG_COT"


class PolicyLearner:
    """
    Contextual bandit optimizer.
    Calculates rewards from historical telemetry and updates the routing decision surface.
    """

    def __init__(self):
        os.makedirs(os.path.dirname(POLICY_PATH), exist_ok=True)

    def _get_connection(self):
        return psycopg2.connect(DB_URL)

    # ------------------------------------------------------------------
    # Core reward computation
    # ------------------------------------------------------------------
    @staticmethod
    def compute_reward(trust: float, tokens: int, latency_ms: int, completion_score: float = 1.0) -> float:
        """R = (Trust * CompletionScore) - λ_cost * NormCost - λ_latency * NormLatency"""
        norm_cost = min(tokens / MAX_TOKENS, 1.0)
        norm_latency = min(latency_ms / MAX_LATENCY, 1.0)
        return round((trust * completion_score) - LAMBDA_COST * norm_cost - LAMBDA_LATENCY * norm_latency, 6)

    # ------------------------------------------------------------------
    # Optimize policy from DB
    # ------------------------------------------------------------------
    def optimize_policy(self) -> Dict[str, Any]:
        """
        Runs the bandit reward optimization:
        1. Fetch per-strategy, per-context rewards from DB
        2. Select best strategy per context bin
        3. Write new policy weights to disk
        Returns the new policy with metadata.
        """
        query = """
        WITH binned AS (
            SELECT
                strategy_selected,
                CASE 
                    WHEN hallucination_risk < 0.3 THEN 'LOW_RISK'
                    WHEN hallucination_risk < 0.7 THEN 'MED_RISK'
                    ELSE 'HIGH_RISK'
                END as risk_bin,
                trust_score,
                COALESCE(completion_score, 1.0) as completion_score,
                tokens_used,
                latency_ms,
                created_at
            FROM requests
            WHERE trust_score IS NOT NULL
        ),
        rewards AS (
            SELECT 
                strategy_selected,
                risk_bin,
                COUNT(*) as sample_size,
                AVG(trust_score) as avg_trust,
                AVG(tokens_used) as avg_tokens,
                AVG(latency_ms) as avg_latency,
                AVG(
                    (trust_score * completion_score)
                    - %s * LEAST(tokens_used::float / %s, 1.0)
                    - %s * LEAST(latency_ms::float / %s, 1.0)
                ) as avg_reward,
                STDDEV(trust_score) as trust_stddev
            FROM binned
            GROUP BY strategy_selected, risk_bin
            HAVING COUNT(*) >= 1
        ),
        ranked AS (
            SELECT 
                risk_bin,
                strategy_selected,
                avg_reward,
                avg_trust,
                avg_tokens,
                avg_latency,
                sample_size,
                trust_stddev,
                ROW_NUMBER() OVER(PARTITION BY risk_bin ORDER BY avg_reward DESC) as rank
            FROM rewards
        )
        SELECT 
            risk_bin, 
            strategy_selected as optimal_strategy, 
            avg_reward, avg_trust, avg_tokens, avg_latency,
            sample_size, trust_stddev
        FROM ranked 
        WHERE rank = 1;
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, (LAMBDA_COST, MAX_TOKENS, LAMBDA_LATENCY, MAX_LATENCY))
                    results = cur.fetchall()

            new_policy = {}
            for row in results:
                new_policy[row["risk_bin"]] = {
                    "strategy": row["optimal_strategy"],
                    "expected_reward": float(row["avg_reward"] or 0),
                    "avg_trust": float(row["avg_trust"] or 0),
                    "avg_tokens": float(row["avg_tokens"] or 0),
                    "avg_latency": float(row["avg_latency"] or 0),
                    "samples": int(row["sample_size"]),
                    "trust_stddev": float(row["trust_stddev"] or 0),
                }

            # Merge with defaults for any missing bins
            current = self.get_current_policy()
            current["weights"].update(new_policy)
            current["last_optimized"] = datetime.utcnow().isoformat()
            current["lambda_cost"] = LAMBDA_COST
            current["lambda_latency"] = LAMBDA_LATENCY

            with open(POLICY_PATH, "w") as f:
                json.dump(current, f, indent=4)

            return current
        except Exception as e:
            logger.error(f"Policy optimization failed: {e}")
            return self.get_current_policy()

    # ------------------------------------------------------------------
    # Current policy weights
    # ------------------------------------------------------------------
    def get_current_policy(self) -> Dict[str, Any]:
        """Retrieves the active policy decision surface."""
        if os.path.exists(POLICY_PATH):
            try:
                with open(POLICY_PATH, "r") as f:
                    data = json.load(f)
                    # Ensure it has the expected format
                    if "weights" in data:
                        return data
            except (json.JSONDecodeError, KeyError):
                pass

        return {
            "weights": {
                "LOW_RISK": {"strategy": "DIRECT", "expected_reward": 0.0, "avg_trust": 0.0, "avg_tokens": 0, "avg_latency": 0, "samples": 0, "trust_stddev": 0},
                "MED_RISK": {"strategy": "SHORT_COT", "expected_reward": 0.0, "avg_trust": 0.0, "avg_tokens": 0, "avg_latency": 0, "samples": 0, "trust_stddev": 0},
                "HIGH_RISK": {"strategy": "LONG_COT", "expected_reward": 0.0, "avg_trust": 0.0, "avg_tokens": 0, "avg_latency": 0, "samples": 0, "trust_stddev": 0},
            },
            "last_optimized": None,
            "lambda_cost": LAMBDA_COST,
            "lambda_latency": LAMBDA_LATENCY,
        }

    # ------------------------------------------------------------------
    # Reward curves (time-series)
    # ------------------------------------------------------------------
    def get_reward_curves(self) -> Dict[str, Any]:
        """
        Returns per-strategy reward time series for charting.
        Bins by date, computes average reward per strategy per day.
        """
        query = """
        SELECT
            DATE(created_at) as date,
            strategy_selected as strategy,
            COUNT(*) as n,
            AVG(trust_score) as avg_trust,
            AVG(tokens_used) as avg_tokens,
            AVG(latency_ms) as avg_latency,
            AVG(
                (trust_score * COALESCE(completion_score, 1.0))
                - %s * LEAST(tokens_used::float / %s, 1.0)
                - %s * LEAST(latency_ms::float / %s, 1.0)
            ) as avg_reward
        FROM requests
        WHERE trust_score IS NOT NULL
        GROUP BY DATE(created_at), strategy_selected
        ORDER BY date ASC, strategy_selected;
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, (LAMBDA_COST, MAX_TOKENS, LAMBDA_LATENCY, MAX_LATENCY))
                    rows = cur.fetchall()

            # Convert to chart-friendly format
            curves: Dict[str, List[Dict]] = {}
            for row in rows:
                strategy = row["strategy"]
                if strategy not in curves:
                    curves[strategy] = []
                curves[strategy].append({
                    "date": str(row["date"]),
                    "reward": float(row["avg_reward"] or 0),
                    "trust": float(row["avg_trust"] or 0),
                    "tokens": float(row["avg_tokens"] or 0),
                    "latency": float(row["avg_latency"] or 0),
                    "n": int(row["n"]),
                })

            return {"curves": curves}
        except Exception as e:
            logger.error(f"Failed to get reward curves: {e}")
            return {"curves": {}, "error": str(e)}

    # ------------------------------------------------------------------
    # A/B test simulation
    # ------------------------------------------------------------------
    def simulate_ab_test(self) -> Dict[str, Any]:
        """
        Simulates three routing strategies using historical data:
        - Always-CoT: pretend every request used LONG_COT
        - Adaptive: actual system behavior (current data)
        - Hybrid: uses learned policy weights
        
        Returns comparative metrics for each arm.
        """
        query = """
        SELECT
            strategy_selected,
            trust_score,
            COALESCE(completion_score, 1.0) as completion_score,
            tokens_used,
            latency_ms,
            hallucination_risk
        FROM requests
        WHERE trust_score IS NOT NULL;
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query)
                    rows = cur.fetchall()

            if not rows:
                return {"error": "No data available for A/B simulation"}

            # -- Arm 1: Always-CoT baseline --
            # Get LONG_COT averages (if any exist), otherwise estimate
            cot_rows = [r for r in rows if r["strategy_selected"] == ALWAYS_COT_STRATEGY]
            if cot_rows:
                cot_avg_trust = sum(float(r["trust_score"]) for r in cot_rows) / len(cot_rows)
                cot_avg_completion = sum(float(r["completion_score"]) for r in cot_rows) / len(cot_rows)
                cot_avg_tokens = sum(int(r["tokens_used"]) for r in cot_rows) / len(cot_rows)
                cot_avg_latency = sum(int(r["latency_ms"]) for r in cot_rows) / len(cot_rows)
            else:
                # Estimate: LONG_COT is typically 1.5x tokens and latency
                cot_avg_trust = sum(float(r["trust_score"]) for r in rows) / len(rows)
                cot_avg_completion = sum(float(r["completion_score"]) for r in rows) / len(rows)
                cot_avg_tokens = sum(int(r["tokens_used"]) for r in rows) / len(rows) * 1.5
                cot_avg_latency = sum(int(r["latency_ms"]) for r in rows) / len(rows) * 1.5

            cot_total_tokens = cot_avg_tokens * len(rows)
            cot_reward = self.compute_reward(cot_avg_trust, int(cot_avg_tokens), int(cot_avg_latency), cot_avg_completion)

            # -- Arm 2: Adaptive (actual system behavior) --
            adaptive_avg_trust = sum(float(r["trust_score"]) for r in rows) / len(rows)
            adaptive_avg_completion = sum(float(r["completion_score"]) for r in rows) / len(rows)
            adaptive_avg_tokens = sum(int(r["tokens_used"]) for r in rows) / len(rows)
            adaptive_avg_latency = sum(int(r["latency_ms"]) for r in rows) / len(rows)
            adaptive_total_tokens = sum(int(r["tokens_used"]) for r in rows)
            adaptive_reward = self.compute_reward(adaptive_avg_trust, int(adaptive_avg_tokens), int(adaptive_avg_latency), adaptive_avg_completion)

            # -- Arm 3: Hybrid (learned policy applied retroactively) --
            policy = self.get_current_policy()
            weights = policy.get("weights", {})
            hybrid_trust_sum = 0.0
            hybrid_token_sum = 0
            hybrid_latency_sum = 0
            for r in rows:
                risk = float(r["hallucination_risk"])
                if risk < 0.3:
                    bin_key = "LOW_RISK"
                elif risk < 0.7:
                    bin_key = "MED_RISK"
                else:
                    bin_key = "HIGH_RISK"

                w = weights.get(bin_key, {})
                # Use the policy bin's averages if available, else use the row's actual data
                if w.get("samples", 0) > 0:
                    hybrid_trust_sum += float(w.get("avg_trust", r["trust_score"]))
                    hybrid_token_sum += int(w.get("avg_tokens", r["tokens_used"]))
                    hybrid_latency_sum += int(w.get("avg_latency", r["latency_ms"]))
                else:
                    hybrid_trust_sum += float(r["trust_score"])
                    hybrid_token_sum += int(r["tokens_used"])
                    hybrid_latency_sum += int(r["latency_ms"])

            n = len(rows)
            hybrid_avg_trust = hybrid_trust_sum / n if n else 0
            hybrid_avg_tokens = hybrid_token_sum / n if n else 0
            hybrid_avg_latency = hybrid_latency_sum / n if n else 0
            
            # For hybrid we approximate completion score to be consistent with trust mapping roughly,
            # or simply use adaptive's avg completion since they are retro-applied on the same dataset.
            hybrid_reward = self.compute_reward(hybrid_avg_trust, int(hybrid_avg_tokens), int(hybrid_avg_latency), adaptive_avg_completion)

            # Token savings
            token_savings_vs_cot = max(0, cot_total_tokens - adaptive_total_tokens)
            savings_pct = (token_savings_vs_cot / cot_total_tokens * 100) if cot_total_tokens > 0 else 0

            return {
                "sample_size": n,
                "arms": {
                    "always_cot": {
                        "label": "Always CoT",
                        "strategy": ALWAYS_COT_STRATEGY,
                        "avg_trust": round(cot_avg_trust, 4),
                        "avg_tokens": round(cot_avg_tokens, 1),
                        "avg_latency": round(cot_avg_latency, 1),
                        "avg_reward": round(cot_reward, 4),
                        "total_tokens": round(cot_total_tokens, 0),
                    },
                    "adaptive": {
                        "label": "Adaptive (Current)",
                        "strategy": "MIXED",
                        "avg_trust": round(adaptive_avg_trust, 4),
                        "avg_tokens": round(adaptive_avg_tokens, 1),
                        "avg_latency": round(adaptive_avg_latency, 1),
                        "avg_reward": round(adaptive_reward, 4),
                        "total_tokens": round(adaptive_total_tokens, 0),
                    },
                    "hybrid": {
                        "label": "Hybrid (Learned)",
                        "strategy": "POLICY",
                        "avg_trust": round(hybrid_avg_trust, 4),
                        "avg_tokens": round(hybrid_avg_tokens, 1),
                        "avg_latency": round(hybrid_avg_latency, 1),
                        "avg_reward": round(hybrid_reward, 4),
                        "total_tokens": round(hybrid_token_sum, 0),
                    },
                },
                "token_savings": {
                    "saved_vs_always_cot": round(token_savings_vs_cot, 0),
                    "savings_pct": round(savings_pct, 1),
                },
            }
        except Exception as e:
            logger.error(f"A/B simulation failed: {e}")
            return {"error": str(e)}

    # ------------------------------------------------------------------
    # Latency distribution by strategy
    # ------------------------------------------------------------------
    def get_latency_distribution(self) -> Dict[str, Any]:
        """Returns latency statistics grouped by strategy for box-plot / histogram."""
        query = """
        SELECT
            strategy_selected as strategy,
            COUNT(*) as n,
            AVG(latency_ms) as avg,
            MIN(latency_ms) as min,
            MAX(latency_ms) as max,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY latency_ms) as p50,
            PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY latency_ms) as p90,
            PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY latency_ms) as p99,
            STDDEV(latency_ms) as stddev
        FROM requests
        GROUP BY strategy_selected
        ORDER BY avg ASC;
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query)
                    rows = cur.fetchall()

            result = []
            for row in rows:
                result.append({
                    "strategy": row["strategy"],
                    "n": int(row["n"]),
                    "avg": round(float(row["avg"] or 0), 1),
                    "min": round(float(row["min"] or 0), 1),
                    "max": round(float(row["max"] or 0), 1),
                    "p50": round(float(row["p50"] or 0), 1),
                    "p90": round(float(row["p90"] or 0), 1),
                    "p99": round(float(row["p99"] or 0), 1),
                    "stddev": round(float(row["stddev"] or 0), 1),
                })
            return {"distributions": result}
        except Exception as e:
            logger.error(f"Failed to get latency distribution: {e}")
            return {"distributions": [], "error": str(e)}

    # ------------------------------------------------------------------
    # Per-strategy performance summary
    # ------------------------------------------------------------------
    def get_strategy_performance(self) -> Dict[str, Any]:
        """Detailed per-strategy performance for decision surface visualization."""
        query = """
        SELECT
            strategy_selected as strategy,
            COUNT(*) as n,
            AVG(trust_score) as avg_trust,
            AVG(tokens_used) as avg_tokens,
            AVG(latency_ms) as avg_latency,
            AVG(hallucination_risk) as avg_risk,
            AVG(
                (trust_score * COALESCE(completion_score, 1.0))
                - %s * LEAST(tokens_used::float / %s, 1.0)
                - %s * LEAST(latency_ms::float / %s, 1.0)
            ) as avg_reward
        FROM requests
        WHERE trust_score IS NOT NULL
        GROUP BY strategy_selected
        ORDER BY avg_reward DESC;
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, (LAMBDA_COST, MAX_TOKENS, LAMBDA_LATENCY, MAX_LATENCY))
                    rows = cur.fetchall()

            results = []
            for row in rows:
                results.append({
                    "strategy": row["strategy"],
                    "n": int(row["n"]),
                    "avg_trust": round(float(row["avg_trust"] or 0), 4),
                    "avg_tokens": round(float(row["avg_tokens"] or 0), 1),
                    "avg_latency": round(float(row["avg_latency"] or 0), 1),
                    "avg_risk": round(float(row["avg_risk"] or 0), 4),
                    "avg_reward": round(float(row["avg_reward"] or 0), 4),
                })
            return {"strategies": results}
        except Exception as e:
            logger.error(f"Failed to get strategy performance: {e}")
            return {"strategies": [], "error": str(e)}
