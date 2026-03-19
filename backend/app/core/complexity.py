import os
import numpy as np
import lightgbm as lgb
from typing import Tuple
from sentence_transformers import SentenceTransformer

class ComplexityEstimator:
    """
    Estimates query difficulty and hallucination risk using structural features and semantic embeddings via LightGBM regressor"""
    def __init__(self, model_path: str="/app/models/complexity_lgb_v2.txt"):
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
        # 1. Structural features (normalized for the synthetic baseline model)
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

    def estimate(self, query: str) -> Tuple[float, float]:
        """
        Returns (difficulty_score, hallucination_risk)
        Scores are bounded [0.0, 1.0]
        """
        features = self.extract_features(query)
        
        # Reshape for single prediction: (1, num_features)
        pred = self.booster.predict(features.reshape(1, -1))[0]
        
        # The baseline model outputs a combined logit. 
        # We split it deterministically for the Phase 2 simulation.
        # In Phase 5, this will be replaced by a multi-target or twin model.
        base_score = float(np.clip(pred, 0.0, 1.0))
        
        difficulty = base_score
        # Risk heavily correlates with difficulty, but spikes on short, vague queries
        risk = float(np.clip(base_score * 1.2 if len(query.split()) < 5 else base_score * 0.8, 0.0, 1.0))
        
        return difficulty, risk

    def _train_baseline_model(self):
        """
        Trains a synthetic baseline model so the infrastructure works flawlessly 
        before we gather real telemetry data in Phase 4/5.
        """
        print("Training baseline LightGBM complexity model...")
        # Generate 500 synthetic feature rows (388 features)
        X_train = np.random.rand(500, 388)
        
        # Ensure semantic features are roughly zero-centered to match real embeddings
        X_train[:, 4:] = X_train[:, 4:] * 2.0 - 1.0
        
        # Synthetic target: higher word count and length = higher difficulty
        # Weighted to ensure complex queries map to high difficulty scores (> 0.85)
        y_train = np.clip((X_train[:, 0] * 0.4) + (X_train[:, 1] * 0.5) + np.abs(X_train[:, 10] * 0.2), 0.0, 1.0)
        
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
