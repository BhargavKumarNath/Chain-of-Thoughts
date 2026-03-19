from app.schemas.reasoning import StrategyEnum

from app.schemas.reasoning import StrategyEnum

class PolicyRouter:
    """
    Contextual bandit prior / Rule engine.
    Optimizes: Maximize Accuracy - lambda1(Tokens) - lambda2(Latency)
    """

    def __init__(self):
        self.LAMBDA_COST = 0.2
        self.LAMBDA_LATENCY = 0.3

    def route(self, difficulty: float, hallucination_risk: float, force_strategy: StrategyEnum = None) -> StrategyEnum:

        if force_strategy:
            return force_strategy

        if difficulty < 0.25 and hallucination_risk < 0.2:
            return StrategyEnum.DIRECT

        if difficulty < 0.6:
            return StrategyEnum.SHORT_COT

        if difficulty >= 0.6 and hallucination_risk < 0.7:
            return StrategyEnum.LONG_COT

        if hallucination_risk >= 0.7 and difficulty < 0.8:
            return StrategyEnum.MULTI_SAMPLE

        return StrategyEnum.TREE_OF_THOUGHTS

