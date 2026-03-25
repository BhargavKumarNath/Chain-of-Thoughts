import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException

from app.schemas.reasoning import (
    ReasoningRequest, 
    ReasoningResponse, 
    TrustScore, 
    ReasoningStep, 
    StrategyEnum, 
    ReasoningGraph
)

from app.core.complexity import ComplexityEstimator
from app.core.policy import PolicyRouter
from app.core.generator import LLMGenerator
from app.core.verifier import VerificationEngine

# Global instances loaded during startup
ml_services = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing ML Infrastructure...")

    ml_services["complexity_estimator"] = ComplexityEstimator()
    ml_services["policy_router"] = PolicyRouter()
    ml_services["generator"] = LLMGenerator()
    ml_services["verifier"] = VerificationEngine()

    print("ML Infrastructure loaded:", ml_services.keys())

    yield

    ml_services.clear()

app = FastAPI(
    title="ReasonOps API",
    description="Meta-reasoning orchestration system infrastructure",
    version="0.2.0",
    lifespan=lifespan
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
    return {"status": "ok", "service": "reasonops-backend"}

@app.post("/api/v1/reason", response_model=ReasoningResponse)
@app.post("/api/v1/reason", response_model=ReasoningResponse)
async def reason_query(request: ReasoningRequest):
    start_time = time.time()
    
    # 1. Estimate & Route
    estimator: ComplexityEstimator = ml_services["complexity_estimator"]
    difficulty, risk = estimator.estimate(request.query)
    
    router: PolicyRouter = ml_services["policy_router"]
    selected_strategy = router.route(difficulty, risk, request.force_strategy)
    
    # 2. Generate Structured LLM Output
    generator: LLMGenerator = ml_services["generator"]
    try:
        llm_output = await generator.generate(request.query, selected_strategy)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM Generation failed: {str(e)}")
        
    # 3. Verify Output & Calculate Trust
    verifier: VerificationEngine = ml_services["verifier"]
    trust_score, flagged_indices = verifier.verify_and_score(llm_output, risk)
    
    # Format steps
    steps =[]
    for s in llm_output.get("reasoning_steps",[]):
        steps.append(ReasoningStep(
            step_index=s.get("step_index", 0),
            content=s.get("content", ""),
            assumptions=s.get("assumptions",[]),
            flagged=s.get("step_index") in flagged_indices
        ))

    latency = int((time.time() - start_time) * 1000)
    
    return ReasoningResponse(
        final_answer=llm_output.get("final_answer", "No answer provided."),
        confidence_score=trust_score.aggregate_score, # Binding confidence closely to trust for now
        trust_score=trust_score,
        hallucination_risk=risk,
        tokens_used=llm_output.get("tokens_used", 0),
        latency_ms=latency,
        strategy_selected=selected_strategy,
        reasoning_steps=steps,
        flagged_steps=flagged_indices,
        reasoning_graph=ReasoningGraph(nodes=[], edges=[])
    )

