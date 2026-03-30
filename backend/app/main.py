import time
import logging
from contextlib import asynccontextmanager
from typing import Optional, List
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query

from app.schemas.reasoning import (
    ReasoningRequest,
    ReasoningResponse,
    TrustScore,
    ReasoningStep,
    StrategyEnum,
    ReasoningGraph,
    VerificationStatusEnum,
    VerificationDetail,
    DifficultyLevel,
)

from app.core.complexity import ComplexityEstimator
from app.core.policy import PolicyRouter
from app.core.generator import LLMGenerator
from app.core.verifier import VerificationEngine
from app.core.telemetry import TelemetryLogger

logger = logging.getLogger(__name__)

# Global instances loaded during startup
ml_services = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing ML Infrastructure...")

    ml_services["complexity_estimator"] = ComplexityEstimator()
    ml_services["policy_router"] = PolicyRouter()
    ml_services["generator"] = LLMGenerator()
    ml_services["verifier"] = VerificationEngine()
    ml_services["telemetry"] = TelemetryLogger()

    print("ML Infrastructure loaded:", ml_services.keys())
    yield

    ml_services.clear()


app = FastAPI(
    title="ReasonOps API",
    description="Meta-reasoning orchestration system infrastructure",
    version="0.5.0",
    lifespan=lifespan,
)

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "reasonops-backend", "version": "0.5.0"}


@app.post("/api/v1/reason", response_model=ReasoningResponse)
async def reason_query(request: ReasoningRequest, background_tasks: BackgroundTasks):
    start_time = time.time()
    debug_log: List[str] = [] if request.debug else None

    def _debug(msg: str):
        if debug_log is not None:
            elapsed = int((time.time() - start_time) * 1000)
            debug_log.append(f"[{elapsed}ms] {msg}")

    # 1. Estimate complexity & classify difficulty
    estimator: ComplexityEstimator = ml_services["complexity_estimator"]
    difficulty, pre_risk, difficulty_level = estimator.estimate(request.query)
    query_type = estimator.classify_query_type(request.query)
    _debug(f"Complexity: difficulty={difficulty:.3f}, risk={pre_risk:.3f}, level={difficulty_level.value}, query_type={query_type.value}")

    # 2. Route strategy
    router: PolicyRouter = ml_services["policy_router"]
    selected_strategy = router.route(difficulty, pre_risk, difficulty_level, request.force_strategy)
    _debug(f"Strategy selected: {selected_strategy.value}")

    # 3. Generate structured LLM output
    generator: LLMGenerator = ml_services["generator"]
    try:
        llm_output = await generator.generate(request.query, selected_strategy)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM Generation failed: {str(e)}")

    _debug(f"Generation complete: {len(llm_output.get('reasoning_steps', []))} steps, {llm_output.get('tokens_used', 0)} tokens")

    # 4. Verify output
    verifier: VerificationEngine = ml_services["verifier"]
    self_consistency_score = float(llm_output.get("_self_consistency_score", 0.0))

    trust_score, flagged_indices, verification_status, verification_confidence, verification_details = (
        verifier.verify_and_score(llm_output, pre_risk, self_consistency_score, query_type=query_type.value)
    )
    _debug(f"Verification: status={verification_status.value}, confidence={verification_confidence:.3f}")
    _debug(f"Trust: aggregate={trust_score.aggregate_score:.3f}")

    # 5. Conditional self-consistency (only when verification is not PASSED)
    retry_used = False
    retry_strategy = None

    if (
        difficulty_level == DifficultyLevel.HARD
        and verification_status != VerificationStatusEnum.PASSED
        and verification_confidence < 0.8
        and selected_strategy != StrategyEnum.MULTI_SAMPLE
    ):
        _debug("Triggering conditional self-consistency (2 additional samples)...")
        try:
            sc_output = await generator.generate_for_self_consistency(request.query, selected_strategy, n=2)
            self_consistency_score = float(sc_output.get("_self_consistency_score", 0.0))
            _debug(f"Self-consistency score: {self_consistency_score:.3f}")

            # Use the consensus output if it has higher self-consistency
            if self_consistency_score > 0.5:
                llm_output = sc_output
                # Re-verify with the consensus output
                trust_score, flagged_indices, verification_status, verification_confidence, verification_details = (
                    verifier.verify_and_score(llm_output, pre_risk, self_consistency_score, query_type=query_type.value)
                )
                _debug(f"Post-SC verification: status={verification_status.value}, confidence={verification_confidence:.3f}")
        except Exception as e:
            _debug(f"Self-consistency failed (continuing with original): {e}")

    # 6. Failure recovery — 1 retry with escalated strategy
    if verification_status == VerificationStatusEnum.FAILED and verification_confidence >= 0.7:
        escalated = router.escalate_strategy(selected_strategy)
        if escalated != selected_strategy:
            _debug(f"Failure recovery: retrying with escalated strategy {escalated.value}")
            try:
                retry_output = await generator.generate(request.query, escalated)
                retry_trust, retry_flagged, retry_status, retry_conf, retry_details = (
                    verifier.verify_and_score(retry_output, pre_risk, self_consistency_score, query_type=query_type.value)
                )
                _debug(f"Retry verification: status={retry_status.value}, confidence={retry_conf:.3f}")

                # Accept retry only if it's better than original
                if retry_trust.aggregate_score > trust_score.aggregate_score:
                    llm_output = retry_output
                    trust_score = retry_trust
                    flagged_indices = retry_flagged
                    verification_status = retry_status
                    verification_confidence = retry_conf
                    verification_details = retry_details
                    retry_used = True
                    retry_strategy = escalated
                    _debug("Retry accepted (better trust score)")
                else:
                    _debug("Retry discarded (original was better)")
            except Exception as e:
                _debug(f"Failure recovery failed (continuing with original): {e}")

    # 7. Compute final hallucination risk (post-verification, grounded)
    hallucination_risk = verifier._compute_hallucination_risk(
        verification_status, verification_confidence,
        trust_score.reasoning_consistency_score,
        len(flagged_indices), len(llm_output.get("reasoning_steps", [])),
        query_type=query_type.value,
    )
    _debug(f"Hallucination risk: {hallucination_risk:.3f}")

    # 8. Build response
    steps = []
    for idx, s in enumerate(llm_output.get("reasoning_steps", []), start=1):
        assumptions = s.get("assumptions", [])
        if isinstance(assumptions, str):
            assumptions = [assumptions]
        if not isinstance(assumptions, list):
            assumptions = []

        step_index = int(s.get("step_index", idx))
        steps.append(ReasoningStep(
            step_index=step_index,
            content=str(s.get("content", "")),
            assumptions=[str(a) for a in assumptions if str(a).strip()],
            flagged=step_index in flagged_indices,
        ))

    latency = int((time.time() - start_time) * 1000)

    response_obj = ReasoningResponse(
        query=request.query,
        final_answer=llm_output.get("final_answer", "No answer provided."),
        strategy_selected=selected_strategy,
        difficulty_level=difficulty_level,
        query_type=query_type.value,
        verification_status=verification_status,
        verification_confidence=round(verification_confidence, 4),
        verification_details=verification_details,
        confidence_score=trust_score.aggregate_score,
        trust_score=trust_score,
        hallucination_risk=hallucination_risk,
        tokens_used=llm_output.get("tokens_used", 0),
        latency_ms=latency,
        reasoning_steps=steps,
        flagged_steps=flagged_indices,
        reasoning_graph=ReasoningGraph(nodes=[], edges=[]),
        debug_log=debug_log,
        retry_used=retry_used,
        retry_strategy=retry_strategy,
    )

    # 9. Log telemetry
    telemetry: TelemetryLogger = ml_services["telemetry"]
    background_tasks.add_task(telemetry.log_trace, request.query, response_obj)

    return response_obj


# Analytics & Trace Management Endpoints
@app.get("/api/v1/analytics")
async def get_analytics():
    telemetry: TelemetryLogger = ml_services["telemetry"]
    return telemetry.get_dashboard_metrics()


@app.get("/api/v1/traces")
async def list_traces(
    search: Optional[str] = Query(None, description="Search query text"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    telemetry: TelemetryLogger = ml_services["telemetry"]
    return telemetry.list_traces(search=search, limit=limit, offset=offset)


@app.get("/api/v1/traces/{trace_id}")
async def get_trace_detail(trace_id: str):
    telemetry: TelemetryLogger = ml_services["telemetry"]
    result = telemetry.get_trace_detail(trace_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.delete("/api/v1/traces/{trace_id}")
async def delete_trace(trace_id: str):
    telemetry: TelemetryLogger = ml_services["telemetry"]
    result = telemetry.delete_trace(trace_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.delete("/api/v1/traces")
async def clear_all_traces():
    telemetry: TelemetryLogger = ml_services["telemetry"]
    return telemetry.delete_all_traces()
