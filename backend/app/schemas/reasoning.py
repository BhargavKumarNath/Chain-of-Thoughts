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


class VerificationStatusEnum(str, Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    PARTIAL = "PARTIAL"
    HEURISTIC = "HEURISTIC"


class DifficultyLevel(str, Enum):
    EASY = "EASY"
    MEDIUM = "MEDIUM"
    HARD = "HARD"
    VERY_HARD = "VERY_HARD"


class QueryTypeEnum(str, Enum):
    DETERMINISTIC = "DETERMINISTIC"
    SEMI_STRUCTURED = "SEMI_STRUCTURED"
    OPEN_ENDED = "OPEN_ENDED"


class VerificationTolerances(BaseModel):
    """Configurable tolerance thresholds for verification."""
    float_abs_tol: float = Field(default=0.01, description="Absolute tolerance for float comparisons")
    float_rel_tol: float = Field(default=1e-6, description="Relative tolerance for float comparisons")
    integer_exact: bool = Field(default=True, description="Require exact match for integer results")
    probability_tol: float = Field(default=0.005, description="Tolerance for probability comparisons (0-1 range)")


class VerificationDetail(BaseModel):
    """Detailed breakdown of a single verification check."""
    method: str = Field(..., description="Verification method used (e.g., 'sympy_arithmetic', 'probability_formula', 'logical_consistency')")
    passed: bool
    expected: Optional[str] = None
    actual: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    message: str = ""


class TrustScore(BaseModel):
    verification_confidence: float = Field(default=0.0, ge=0.0, le=1.0,
        description="Confidence from deterministic verification (highest weight)")
    reasoning_consistency_score: float = Field(default=0.0, ge=0.0, le=1.0,
        description="Internal step-to-step coherence score")
    self_consistency_score: float = Field(default=0.0, ge=0.0, le=1.0,
        description="Agreement across multiple reasoning passes")
    aggregate_score: float = Field(default=0.0, ge=0.0, le=1.0)

    # Legacy fields kept for backward compatibility with existing DB data
    logical_consistency: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_alignment: float = Field(default=0.0, ge=0.0, le=1.0)
    entropy: float = Field(default=0.0, ge=0.0)
    contradiction_rate: float = Field(default=0.0, ge=0.0, le=1.0)


class ReasoningStep(BaseModel):
    step_index: int
    content: str
    assumptions: List[str] = Field(default_factory=list)
    entropy: Optional[float] = None
    flagged: bool = False
    verification_note: Optional[str] = None


class ReasoningGraph(BaseModel):
    nodes: List[Dict[str, Any]] = Field(default_factory=list)
    edges: List[Dict[str, Any]] = Field(default_factory=list)


class ReasoningRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    force_strategy: Optional[StrategyEnum] = None
    debug: bool = Field(default=False, description="Enable debug mode for detailed logs")


class ReasoningResponse(BaseModel):
    # Core output
    query: str = ""
    final_answer: str
    strategy_selected: StrategyEnum
    difficulty_level: DifficultyLevel = DifficultyLevel.MEDIUM
    query_type: str = "SEMI_STRUCTURED"

    # Verification
    verification_status: VerificationStatusEnum = VerificationStatusEnum.PARTIAL
    verification_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    verification_details: List[VerificationDetail] = Field(default_factory=list)

    # Scoring
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    completion_score: float = Field(default=1.0, ge=0.0, le=1.0)
    trust_score: TrustScore
    hallucination_risk: float = Field(default=0.0, ge=0.0, le=1.0)

    # Telemetry
    tokens_used: int = 0
    latency_ms: int = 0

    # Trace
    reasoning_steps: List[ReasoningStep] = Field(default_factory=list)
    flagged_steps: List[int] = Field(default_factory=list)
    reasoning_graph: ReasoningGraph = Field(default_factory=ReasoningGraph)

    # Observability
    debug_log: Optional[List[str]] = None

    # Recovery metadata
    retry_used: bool = False
    retry_strategy: Optional[StrategyEnum] = None
