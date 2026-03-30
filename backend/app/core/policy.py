from app.schemas.reasoning import StrategyEnum, DifficultyLevel


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

    def route(
        self,
        difficulty: float,
        hallucination_risk: float,
        difficulty_level: DifficultyLevel,
        force_strategy: StrategyEnum = None,
    ) -> StrategyEnum:
        """
        Select reasoning strategy based on difficulty level.
        force_strategy overrides automatic routing.
        """
        if force_strategy:
            return force_strategy

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
