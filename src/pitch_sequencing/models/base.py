"""Abstract base model for all pitch prediction models."""

from abc import ABC, abstractmethod

import numpy as np


class BaseModel(ABC):
    """Unified interface for all pitch prediction models."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name for display."""

    @property
    @abstractmethod
    def model_type(self) -> str:
        """'tabular' or 'sequence' — determines data format expected."""

    @abstractmethod
    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """Train the model."""

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        """Return predicted class labels."""

    @abstractmethod
    def predict_proba(self, X) -> np.ndarray:
        """Return class probabilities (n_samples x n_classes)."""

    def get_params(self) -> dict:
        """Return hyperparameters for logging."""
        return {}
