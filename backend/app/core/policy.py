import random
from app.schemas.reasoning import StrategyEnum, DifficultyLevel
from app.core.learning import PolicyLearner

POLICY_PATH = "/app/models/policy_weights.json"

class PolicyRouter:
    """
    Simplified strategy routing based on difficulty level classification.
    Maps: EASY→DIRECT, MEDIUM→SHORT_COT, HARD→LONG_COT.
    Supports force_strategy override.
    """

    # Difficulty level → default strategy
    LEVEL_STRATEGY_MAP = {
        DifficultyLevel.EASY: StrategyEnum.DIRECT,
        DifficultyLevel.MEDIUM: StrategyEnum.SHORT_COT,
        DifficultyLevel.HARD: StrategyEnum.LONG_COT,
    }

    def __init__(self):
        self.LAMBDA_COST = 0.2
        self.LAMBDA_LATENCY = 0.3
        self.learner = PolicyLearner()

    def route(
        self,
        query: str,
        difficulty: float,
        hallucination_risk: float,
        difficulty_level: DifficultyLevel,
        force_strategy: StrategyEnum = None,
    ) -> StrategyEnum:
        """
        Select reasoning strategy based on bandit learned policy weights.
        Falls back to difficulty heuristics if confidence/samples are low.
        force_strategy overrides automatic routing.
        """
        if force_strategy:
            return force_strategy
            
        # Problem-type routing heuristics (execution before bandit)
        query_lower = query.lower()
        
        # 1. Self-referential or paradox
        if "itself" in query_lower or "its own" in query_lower or "flags itself" in query_lower or "trustworthy if and only if" in query_lower or "paradox" in query_lower:
            return StrategyEnum.MULTI_SAMPLE
            
        # 2. Constraint satisfaction (schedule, allocate, assign, workers, tasks, constraints, makespan)
        constraint_keywords = ["schedule", "allocate", "assign", "workers", "tasks", "constraints", "makespan"]
        if any(kw in query_lower for kw in constraint_keywords) and ("constraint" in query_lower or "optimize" in query_lower or "minimize" in query_lower):
            # If VERY_HARD, prefer EXTERNAL_SOLVER if deterministic, otherwise TREE_OF_THOUGHTS
            return StrategyEnum.TREE_OF_THOUGHTS
            
        # 3. Medical triage / multi-branch decision
        medical_keywords = ["patient", "diagnosis", "decision tree", "tradeoffs"]
        if sum(1 for kw in medical_keywords if kw in query_lower) >= 2 or ("medical" in query_lower and "triage" in query_lower) or ("patient" in query_lower and "presents with" in query_lower):
            return StrategyEnum.TREE_OF_THOUGHTS
            
        # 4. Multi-regime economic
        if "economic" in query_lower or "central bank" in query_lower or "interest rates" in query_lower or "inflation" in query_lower:
            return StrategyEnum.LONG_COT
            
        # 5. VERY_HARD baseline
        if difficulty_level == DifficultyLevel.VERY_HARD:
            return StrategyEnum.TREE_OF_THOUGHTS

        # Map hallucination risk to policy context bin
        if hallucination_risk < 0.3:
            bin_key = "LOW_RISK"
        elif hallucination_risk < 0.7:
            bin_key = "MED_RISK"
        else:
            bin_key = "HIGH_RISK"

        # Load learned policy weights
        policy_data = self.learner.get_current_policy()
        weights = policy_data.get("weights", {})
        bin_policy = weights.get(bin_key, {})

        samples = bin_policy.get("samples", 0)
        learned_strategy_name = bin_policy.get("strategy")

        # 1. Forced Exploration (15% chance for HIGH_RISK to gather complex mapping data)
        if bin_key == "HIGH_RISK" and random.random() < 0.15:
            # Randomly explore complex strategies
            explore_candidates = [StrategyEnum.LONG_COT, StrategyEnum.TREE_OF_THOUGHTS, StrategyEnum.MULTI_SAMPLE]
            return random.choice(explore_candidates)

        # 2. Minimum Sample Threshold Guard
        if samples < 10 or not learned_strategy_name:
            # Fallback to static mapping (heuristics)
            return self.LEVEL_STRATEGY_MAP.get(difficulty_level, StrategyEnum.SHORT_COT)

        # 3. Hybrid / Bandit Assigned Strategy
        try:
            return StrategyEnum[learned_strategy_name]
        except KeyError:
            return self.LEVEL_STRATEGY_MAP.get(difficulty_level, StrategyEnum.SHORT_COT)

    def escalate_strategy(self, current: StrategyEnum) -> StrategyEnum:
        """
        Escalate strategy for failure recovery retry.
        Used when verification fails and we want a more thorough approach.
        """
        escalation = {
            StrategyEnum.DIRECT: StrategyEnum.SHORT_COT,
            StrategyEnum.SHORT_COT: StrategyEnum.LONG_COT,
            StrategyEnum.LONG_COT: StrategyEnum.LONG_COT,
            StrategyEnum.TREE_OF_THOUGHTS: StrategyEnum.LONG_COT,
            StrategyEnum.MULTI_SAMPLE: StrategyEnum.LONG_COT,
            StrategyEnum.EXTERNAL_SOLVER: StrategyEnum.LONG_COT,
        }
        return escalation.get(current, StrategyEnum.LONG_COT)
