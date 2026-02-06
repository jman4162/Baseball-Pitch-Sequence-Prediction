"""AutoGluon TabularPredictor wrapper."""

import numpy as np
import pandas as pd

from .base import BaseModel


class AutoGluonModel(BaseModel):
    """AutoGluon TabularPredictor for tabular pitch data."""

    def __init__(self, config=None):
        config = config or {}
        self.preset = config.get("preset", "good_quality")
        self.time_limit = config.get("time_limit", None)
        self.models_dir = config.get("models_dir", "autogluon_pitchtype_models")
        self._predictor = None
        self._label = None

    @property
    def name(self) -> str:
        return "AutoGluon"

    @property
    def model_type(self) -> str:
        return "tabular"

    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        from autogluon.tabular import TabularDataset, TabularPredictor

        self._label = y_train.name if hasattr(y_train, "name") else "target"
        train_df = pd.DataFrame(X_train).copy()
        train_df[self._label] = y_train.values if hasattr(y_train, "values") else y_train
        train_data = TabularDataset(train_df)

        fit_kwargs = {"presets": self.preset}
        if self.time_limit is not None:
            fit_kwargs["time_limit"] = self.time_limit

        self._predictor = TabularPredictor(
            label=self._label, path=self.models_dir
        ).fit(train_data, **fit_kwargs)

    def predict(self, X) -> np.ndarray:
        from autogluon.tabular import TabularDataset

        test_df = pd.DataFrame(X)
        return self._predictor.predict(TabularDataset(test_df)).values

    def predict_proba(self, X) -> np.ndarray:
        from autogluon.tabular import TabularDataset

        test_df = pd.DataFrame(X)
        proba = self._predictor.predict_proba(TabularDataset(test_df))
        return proba.values

    def get_params(self) -> dict:
        return {"preset": self.preset, "time_limit": self.time_limit}
