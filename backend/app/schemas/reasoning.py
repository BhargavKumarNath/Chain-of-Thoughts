from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class StrategyEnum(str, Enum):
    DIRECT = "DIRECT"
    SHORT_COT = "SHORT_COT"
    LONG_COT = "LONG_COT"
    TREE_OF_THOUGHTS = "TREE_OF_THOUGHTS"
    MULTI_SAMPLE = "MULTI_SAMPLE"
    EXTERNAL_SOLVER = "EXTERNAL_SOLVER"

class TrustScore(BaseModel):
    logical_consistency: float = Field(..., ge=0.0, le=1.0)
    evidence_alignment: float = Field(..., ge=0.0, le=1.0)
    entropy: float = Field(..., ge=0.0)
    contradiction_rate: float = Field(..., ge=0.0, le=1.0)
    aggregate_score: float = Field(..., ge=0.0, le=1.0)

class ReasoningStep(BaseModel):
    step_index: int
    content: str
    assumptions: List[str] = Field(default_factory=list)
    entropy: Optional[float] = None
    flagged: bool = False

class ReasoningGraph(BaseModel):
    nodes: List[Dict[str, Any]] = Field(default_factory=list)
    edges: List[Dict[str, Any]] = Field(default_factory=list)

class ReasoningRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    force_strategy: Optional[StrategyEnum] = None

class ReasoningResponse(BaseModel):
    final_answer: str
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    trust_score: TrustScore
    hallucination_risk: float = Field(..., ge=0.0, le=1.0)
    tokens_used: int
    latency_ms: int
    strategy_selected: StrategyEnum
    reasoning_steps: List[ReasoningStep]
    flagged_steps: List[int] = Field(default_factory=list)
    reasoning_graph: ReasoningGraph
