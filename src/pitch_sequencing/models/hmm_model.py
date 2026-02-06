"""HMM model wrapper using hmmlearn CategoricalHMM."""

import numpy as np
from .base import BaseModel


class HMMModel(BaseModel):
    """Hidden Markov Model for pitch sequence prediction."""

    def __init__(self, config=None):
        config = config or {}
        self.min_components = config.get("min_components", 1)
        self.max_components = config.get("max_components", 8)
        self.n_iter = config.get("n_iter", 100)
        self._model = None
        self._best_n = None

    @property
    def name(self) -> str:
        return "HMM"

    @property
    def model_type(self) -> str:
        return "sequence"

    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """Train HMM by sweeping n_components and picking best by validation accuracy.

        For HMM, X_train is expected to be a flat 2D array of shape (n_samples, 1)
        with encoded pitch types (the HMM uses its own flat encoding, not windowed).
        y_train is the same flat array (self-supervised next-token prediction).
        """
        from hmmlearn import hmm as hmmlearn_hmm

        best_accuracy = 0
        best_model = None

        for n_components in range(self.min_components, self.max_components + 1):
            model = hmmlearn_hmm.CategoricalHMM(
                n_components=n_components,
                n_iter=self.n_iter,
                random_state=42,
            )
            model.fit(X_train)

            if X_val is not None:
                predicted = model.predict(X_val)
                actual = X_val.flatten()
                accuracy = np.mean(predicted == actual)
            else:
                predicted = model.predict(X_train)
                actual = X_train.flatten()
                accuracy = np.mean(predicted == actual)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                self._best_n = n_components

        self._model = best_model

    def predict(self, X) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        """Return emission probabilities for each sample given predicted state."""
        states = self._model.predict(X)
        emission = self._model.emissionprob_
        return emission[states]

    def get_params(self) -> dict:
        return {
            "n_components": self._best_n,
            "min_components": self.min_components,
            "max_components": self.max_components,
            "n_iter": self.n_iter,
        }
