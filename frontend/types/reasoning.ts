export enum StrategyEnum {
    DIRECT = "DIRECT",
    SHORT_COT = "SHORT_COT",
    LONG_COT = "LONG_COT",
    TREE_OF_THOUGHTS = "TREE_OF_THOUGHTS",
    MULTI_SAMPLE = "MULTI_SAMPLE",
    EXTERNAL_SOLVER = "EXTERNAL_SOLVER"
}

export enum VerificationStatus {
    PASSED = "PASSED",
    FAILED = "FAILED",
    PARTIAL = "PARTIAL",
    HEURISTIC = "HEURISTIC"
}

export enum DifficultyLevel {
    EASY = "EASY",
    MEDIUM = "MEDIUM",
    HARD = "HARD"
}

export enum QueryType {
    DETERMINISTIC = "DETERMINISTIC",
    SEMI_STRUCTURED = "SEMI_STRUCTURED",
    OPEN_ENDED = "OPEN_ENDED"
}

export interface VerificationDetail {
    method: string;
    passed: boolean;
    expected?: string;
    actual?: string;
    confidence: number;
    message: string;
}

export interface TrustScore {
    verification_confidence: number;
    reasoning_consistency_score: number;
    self_consistency_score: number;
    aggregate_score: number;
    // Legacy
    logical_consistency: number;
    evidence_alignment: number;
    entropy: number;
    contradiction_rate: number;
}

export interface ReasoningStep {
    step_index: number;
    content: string;
    assumptions: string[];
    entropy?: number;
    flagged: boolean;
    verification_note?: string;
}

export interface ReasoningGraph {
    nodes: any[];
    edges: any[];
}

export interface ReasoningRequest {
    query: string;
    force_strategy?: StrategyEnum;
    debug?: boolean;
}

export interface ReasoningResponse {
    query: string;
    final_answer: string;
    strategy_selected: StrategyEnum;
    difficulty_level: DifficultyLevel;
    query_type: string;
    verification_status: VerificationStatus;
    verification_confidence: number;
    verification_details: VerificationDetail[];
    confidence_score: number;
    trust_score: TrustScore;
    hallucination_risk: number;
    tokens_used: number;
    latency_ms: number;
    reasoning_steps: ReasoningStep[];
    flagged_steps: number[];
    reasoning_graph: ReasoningGraph;
    debug_log?: string[];
    retry_used: boolean;
    retry_strategy?: StrategyEnum;
}

export interface TraceListItem {
    id: string;
    query: string;
    strategy_selected: string;
    trust_score: number;
    latency_ms: number;
    verification_status?: string;
    verification_confidence?: number;
    difficulty_level?: string;
    created_at: string;
}