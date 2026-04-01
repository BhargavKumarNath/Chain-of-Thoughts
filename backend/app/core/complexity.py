import os
import logging
import warnings

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")

import re
import numpy as np
import lightgbm as lgb
from typing import Tuple
from sentence_transformers import SentenceTransformer
from app.schemas.reasoning import DifficultyLevel, QueryTypeEnum

logger = logging.getLogger(__name__)


# Query complexity signal keywords
_MULTI_STEP_KEYWORDS = {
    "then", "therefore", "hence", "thus", "subsequently", "consequently",
    "given that", "if", "assuming", "provided that", "such that",
    "find the probability", "find the value", "calculate", "compute",
    "determine", "evaluate", "derive", "prove", "show that",
    "at least", "exactly", "identify", "compare", "contrast", "differentiate between"
}

_HARD_MARKERS = {
    "prove", "disprove", "bayesian", "conditional probability",
    "subject to", "lagrangian", "dynamic programming",
    "integrate", "differentiate", "eigenvalue", "matrix",
    "virtue ethics", "deontological", "utilitarian", "perspectives", "frameworks",
    "counterfactual", "recursive", "second-order", "third-order",
    "uncertainty", "dilemma"
}

_VERY_HARD_MARKERS = {
    "schedule", "optimize", "allocate", "minimize makespan", 
    "maximize", "minimize", "optimal", "argmax", "argmin",
    "np-hard", "combinatorial", "traveling salesperson"
}

_DOMAIN_MARKERS = {
    "medical", "patient", "diagnosis", "clinical",
    "legal", "lawsuit", "jurisdiction", "statute",
    "financial", "portfolio", "investment", "interest rate"
}

_MATH_OPERATORS = re.compile(r"[\+\-\*×÷/\^=\(\)]")
_NUMERIC_PATTERN = re.compile(r"\d+\.?\d*")
_CONSTRAINT_PATTERN = re.compile(r"constraint|rule|condition|requirement")


class ComplexityEstimator:
    """
    Estimates query difficulty and hallucination risk using structural features,
    semantic embeddings via LightGBM regressor, and keyword-based complexity signals.

    Returns both a continuous difficulty score and a categorical DifficultyLevel.
    """

    def __init__(self, model_path: str = "/app/models/complexity_lgb_v2.txt"):
        self.semantic_dim = 384
        self.encoder = None
        try:
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            self.semantic_dim = self.encoder.get_sentence_embedding_dimension()
        except Exception as exc:
            print(f"Warning: SentenceTransformer model could not be loaded. Falling back to zeros. Error: {exc}")
        self.model_path = model_path

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        if not os.path.exists(self.model_path):
            self._train_baseline_model()

        self.booster = lgb.Booster(model_file=self.model_path)

    def extract_features(self, query: str) -> np.ndarray:
        """Extracts structural and semantic features from the query."""
        # 1. Structural features (normalized)
        length = min(len(query) / 300.0, 1.0)
        word_count = min(len(query.split()) / 40.0, 1.0)
        avg_word_len = min((len(query) / max(len(query.split()), 1)) / 10.0, 1.0)
        num_questions = min(query.count('?') / 3.0, 1.0)

        structural_features = np.array([length, word_count, avg_word_len, num_questions], dtype=np.float32)

        # 2. Semantic features (384 dimensions for MiniLM)
        if self.encoder is None:
            semantic_features = np.zeros(self.semantic_dim, dtype=np.float32)
        else:
            semantic_features = self.encoder.encode(query, convert_to_numpy=True)

        # Combine (4 + 384 = 388 features)
        return np.concatenate([structural_features, semantic_features])

    def _compute_keyword_signals(self, query: str) -> Tuple[float, float, float, float, bool]:
        """
        Compute complexity signals from query keywords.
        Returns (math_signal, multi_step_signal, hard_signal, very_hard_signal, is_paradox) each in [0, 1] plus a boolean paradox flag.
        """
        query_lower = query.lower()

        # Math presence: operators and numbers
        math_operator_count = len(_MATH_OPERATORS.findall(query))
        number_count = len(_NUMERIC_PATTERN.findall(query))
        math_signal = min(1.0, (math_operator_count + number_count * 0.5) / 6.0)

        # Multi-step reasoning keywords
        multi_step_count = sum(1 for kw in _MULTI_STEP_KEYWORDS if kw in query_lower)
        multi_step_signal = min(1.0, multi_step_count / 3.0)

        # Hard problem markers
        hard_count = sum(1 for kw in _HARD_MARKERS if kw in query_lower)
        
        # Self-referential or paradoxical
        is_paradox = False
        if "itself" in query_lower or "its own" in query_lower or "paradox" in query_lower or "self-reference" in query_lower or "flags itself" in query_lower or "if and only if" in query_lower:
            hard_count += 2
            is_paradox = True
            
        # Constraints >= 3
        constraint_count = len(_CONSTRAINT_PATTERN.findall(query_lower))
        if constraint_count >= 3:
            hard_count += 2
            
        # Domain tags + multi-variable (simulated by multi_step or constraints)
        domain_count = sum(1 for kw in _DOMAIN_MARKERS if kw in query_lower)
        if domain_count > 0 and (constraint_count > 0 or multi_step_count > 0):
            hard_count += 2

        hard_signal = min(1.0, hard_count / 2.0)
        
        # Very hard markers
        very_hard_count = sum(1 for kw in _VERY_HARD_MARKERS if kw in query_lower)
        very_hard_signal = min(1.0, very_hard_count / 1.0)

        return math_signal, multi_step_signal, hard_signal, very_hard_signal, is_paradox

    def classify_difficulty(self, difficulty_score: float, word_count: int = 0) -> DifficultyLevel:
        """Classify continuous difficulty score into categorical level."""
        if difficulty_score >= 0.85 or (difficulty_score >= 0.65 and word_count >= 80):
            return DifficultyLevel.VERY_HARD
        elif difficulty_score >= 0.65:
            return DifficultyLevel.HARD
        elif difficulty_score >= 0.25:
            return DifficultyLevel.MEDIUM
        else:
            return DifficultyLevel.EASY

    def classify_query_type(self, query: str) -> QueryTypeEnum:
        """
        Lightweight heuristic query type classifier.
        OPEN_ENDED: conceptual / explanatory queries
        DETERMINISTIC: math, calculations, factual lookups
        SEMI_STRUCTURED: everything else
        """
        q = query.lower().strip()

        open_ended_signals = [
            "explain", "understand", "why", "how does", "how do", "how can",
            "implications", "impact", "significance", "describe", "discuss",
            "compare and contrast", "what are the", "in what ways",
            "advantages", "disadvantages", "opinion", "perspective",
            "elaborate", "interpret", "analyze the role", "what role",
        ]
        deterministic_signals = [
            "calculate", "compute", "what is", "how much", "how many",
            "solve", "find the value", "find the probability", "evaluate",
            "equals", "sum of", "product of", "derivative", "integral",
        ]

        # Check for numbers, equations, math operators (strong deterministic signal)
        import re
        has_numbers = bool(re.search(r'\d+\.?\d*\s*[\+\-\*×÷/%\^=]', q))
        has_percent = '%' in q and bool(re.search(r'\d', q))

        open_score = sum(1 for kw in open_ended_signals if kw in q)
        det_score = sum(1 for kw in deterministic_signals if kw in q)
        if has_numbers:
            det_score += 2
        if has_percent:
            det_score += 1

        if det_score > open_score and det_score >= 1:
            return QueryTypeEnum.DETERMINISTIC
        if open_score > det_score and open_score >= 1:
            return QueryTypeEnum.OPEN_ENDED
        return QueryTypeEnum.SEMI_STRUCTURED

    def estimate(self, query: str) -> Tuple[float, float, DifficultyLevel]:
        """
        Returns (difficulty_score, hallucination_risk_estimate, difficulty_level).
        Scores are bounded [0.0, 1.0].
        """
        features = self.extract_features(query)

        # LightGBM prediction
        pred = self.booster.predict(features.reshape(1, -1))[0]
        base_score = float(np.clip(pred, 0.0, 1.0))

        # Keyword-based adjustment
        math_signal, multi_step_signal, hard_signal, very_hard_signal, is_paradox = self._compute_keyword_signals(query)

        # Combine LightGBM score with keyword signals
        difficulty = float(np.clip(
            base_score * 0.5 + math_signal * 0.15 + multi_step_signal * 0.2 + hard_signal * 0.15 + very_hard_signal * 0.3,
            0.0, 1.0
        ))
        
        # Bug 4 Override - Hard Ceiling for Paradoxes
        if is_paradox:
             difficulty = max(0.80, difficulty)

        # Risk: correlated with difficulty, spikes on short vague queries
        word_count = len(query.split())
        if word_count < 5:
            risk = float(np.clip(difficulty * 1.2, 0.0, 1.0))
        else:
            risk = float(np.clip(difficulty * 0.8, 0.0, 1.0))

        difficulty_level = self.classify_difficulty(difficulty, word_count)

        return difficulty, risk, difficulty_level

    def _train_baseline_model(self):
        """Trains a synthetic baseline model for infrastructure bootstrap."""
        print("Training baseline LightGBM complexity model...")
        X_train = np.random.rand(500, 388)
        X_train[:, 4:] = X_train[:, 4:] * 2.0 - 1.0
        y_train = np.clip(
            (X_train[:, 0] * 0.4) + (X_train[:, 1] * 0.5) + np.abs(X_train[:, 10] * 0.2),
            0.0, 1.0
        )

        train_data = lgb.Dataset(X_train, label=y_train)
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 15,
            'learning_rate': 0.05,
            'verbose': -1
        }

        booster = lgb.train(params, train_data, num_boost_round=50)
        booster.save_model(self.model_path)
        print(f"Baseline model saved to {self.model_path}")
