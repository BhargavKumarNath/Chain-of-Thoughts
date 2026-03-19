import time
from contextlib import asynccontextmanager
from fastapi import FastAPI

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

# Global instances loaded during startup
ml_services = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load ML models on startup
    print("Initializing Complexity Estimator (loading SentenceTransformer and LightGBM)...")
    ml_services["complexity_estimator"] = ComplexityEstimator()
    ml_services["policy_router"] = PolicyRouter()
    print("ML Infrastructure loaded.")
    yield
    # Clean up on shutdown
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
async def reason_query(request: ReasoningRequest):
    """
    Phase 2 Endpoint.
    Dynamically routes the request based on ML complexity estimation.
    LLM output generation is still mocked until Phase 3.
    """
    start_time = time.time()
    
    # 1. Complexity Estimation (LightGBM + Embeddings)
    estimator: ComplexityEstimator = ml_services["complexity_estimator"]
    difficulty, risk = estimator.estimate(request.query)
    
    # 2. Strategy Routing (Policy Engine)
    router: PolicyRouter = ml_services["policy_router"]
    selected_strategy = router.route(difficulty, risk, request.force_strategy)
    
    # Simulate LLM/Verification processing time based on strategy selected
    # Tree of thoughts is "slower" than direct
    simulated_delay = {
        StrategyEnum.DIRECT: 0.5,
        StrategyEnum.SHORT_COT: 1.2,
        StrategyEnum.LONG_COT: 2.5,
        StrategyEnum.MULTI_SAMPLE: 3.0,
        StrategyEnum.TREE_OF_THOUGHTS: 4.5,
        StrategyEnum.EXTERNAL_SOLVER: 2.0
    }.get(selected_strategy, 1.0)
    
    time.sleep(simulated_delay)
    
    trust = TrustScore(
        logical_consistency=0.92,
        evidence_alignment=0.88,
        entropy=0.15,
        contradiction_rate=0.0,
        aggregate_score=max(0.0, 1.0 - risk) # Tie trust roughly to inverse risk for Phase 2
    )
    
    steps =[
        ReasoningStep(
            step_index=1,
            content=f"Evaluated query complexity. Difficulty: {difficulty:.2f}, Risk: {risk:.2f}",
            assumptions=["Query is bounded and requires specific factual retrieval."],
            entropy=0.01,
            flagged=False
        ),
        ReasoningStep(
            step_index=2,
            content=f"Policy engine routed request to {selected_strategy.value}.",
            assumptions=[],
            entropy=0.02,
            flagged=False
        )
    ]
    
    latency = int((time.time() - start_time) * 1000)
    tokens = int(difficulty * 1000) + 50 # Heuristic mock token usage
    
    return ReasoningResponse(
        final_answer=f"This response was routed using the {selected_strategy.value} strategy due to a complexity score of {difficulty:.2f}.",
        confidence_score=0.95,
        trust_score=trust,
        hallucination_risk=risk,
        tokens_used=tokens,
        latency_ms=latency,
        strategy_selected=selected_strategy,
        reasoning_steps=steps,
        flagged_steps=[],
        reasoning_graph=ReasoningGraph(nodes=[], edges=[])
    )

