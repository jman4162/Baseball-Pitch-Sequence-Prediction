"""Baseline models: Logistic Regression and Random Forest."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from .base import BaseModel


class LogisticRegressionModel(BaseModel):
    """Logistic Regression baseline for tabular pitch data."""

    def __init__(self, config=None):
        config = config or {}
        self.C = config.get("C", 1.0)
        self.penalty = config.get("penalty", "l2")
        self.class_weight = config.get("class_weight", "balanced")
        self.max_iter = config.get("max_iter", 1000)
        self._model = None

    @property
    def name(self) -> str:
        return "Logistic Regression"

    @property
    def model_type(self) -> str:
        return "tabular"

    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        self._model = LogisticRegression(
            C=self.C,
            penalty=self.penalty,
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            solver="lbfgs",
            multi_class="multinomial",
        )
        self._model.fit(X_train, y_train)

    def predict(self, X) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        return self._model.predict_proba(X)

    def get_params(self) -> dict:
        return {"C": self.C, "penalty": self.penalty, "class_weight": self.class_weight}


class RandomForestModel(BaseModel):
    """Random Forest baseline for tabular pitch data."""

    def __init__(self, config=None):
        config = config or {}
        self.n_estimators = config.get("n_estimators", 200)
        self.max_depth = config.get("max_depth", 15)
        self.min_samples_split = config.get("min_samples_split", 5)
        self.class_weight = config.get("class_weight", "balanced")
        self._model = None

    @property
    def name(self) -> str:
        return "Random Forest"

    @property
    def model_type(self) -> str:
        return "tabular"

    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        self._model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            class_weight=self.class_weight,
            random_state=42,
            n_jobs=-1,
        )
        self._model.fit(X_train, y_train)

    def predict(self, X) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        return self._model.predict_proba(X)

    def get_params(self) -> dict:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
        }
