export enum StrategyEnum {
    DIRECT = "DIRECT",
    SHORT_COT = "SHORT_COT",
    LONG_COT = "LONG_COT",
    TREE_OF_THOUGHTS = "TREE_OF_THOUGHTS",
    MULTI_SAMPLE = "MULTI_SAMPLE",
    EXTERNAL_SOLVER = "EXTERNAL_SOLVER"
}

export interface TrustScore {
    logical_consistency: number;
    evidence_alignment: number;
    entropy: number;
    contradiction_rate: number;
    aggregate_score: number;
}

export interface ReasoningStep {
    step_index: number;
    content: string;
    assumptions: string[];
    entropy?: number;
    flagged: boolean;
}

export interface ReasoningGraph {
    nodes: any[];
    edges: any[];
}

export interface ReasoningRequest {
    query: string;
    force_strategy?: StrategyEnum;
}

export interface ReasoningResponse {
    final_answer: string;
    confidence_score: number;
    trust_score: TrustScore;
    hallucination_risk: number;
    tokens_used: number;
    latency_ms: number;
    strategy_selected: StrategyEnum;
    reasoning_steps: ReasoningStep[];
    flagged_steps: number[];
    reasoning_graph: ReasoningGraph;
}