"""LSTM model for pitch sequence prediction."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .base import BaseModel
from .torch_utils import PitchSequenceDataset, train_torch_model, predict_torch_model


class PitchPredictor(nn.Module):
    """2-layer LSTM for pitch type classification."""

    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.dropout(h_n[-1])
        return self.fc(out)


class LSTMModel(BaseModel):
    """LSTM wrapper implementing BaseModel interface."""

    def __init__(self, config=None):
        config = config or {}
        self.hidden_size = config.get("hidden_size", 64)
        self.num_layers = config.get("num_layers", 2)
        self.dropout = config.get("dropout", 0.3)
        self.epochs = config.get("epochs", 20)
        self.lr = config.get("learning_rate", 0.001)
        self.batch_size = config.get("batch_size", 256)
        self._model = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._num_classes = None
        self._history = None

    @property
    def name(self) -> str:
        return "LSTM"

    @property
    def model_type(self) -> str:
        return "sequence"

    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        input_size = X_train.shape[2]
        self._num_classes = len(np.unique(y_train))

        self._model = PitchPredictor(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_classes=self._num_classes,
            dropout=self.dropout,
        )

        train_ds = PitchSequenceDataset(X_train, y_train)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        if X_val is not None and y_val is not None:
            val_ds = PitchSequenceDataset(X_val, y_val)
        else:
            # Use last 20% of training data as validation
            split = int(len(X_train) * 0.8)
            val_ds = PitchSequenceDataset(X_train[split:], y_train[split:])
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        self._history = train_torch_model(
            self._model, train_loader, val_loader,
            epochs=self.epochs, lr=self.lr, device=self._device,
        )

    def predict(self, X) -> np.ndarray:
        ds = PitchSequenceDataset(X, np.zeros(len(X), dtype=np.int64))
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False)
        preds, _ = predict_torch_model(self._model, loader, self._device)
        return preds

    def predict_proba(self, X) -> np.ndarray:
        ds = PitchSequenceDataset(X, np.zeros(len(X), dtype=np.int64))
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False)
        _, probs = predict_torch_model(self._model, loader, self._device)
        return probs

    def get_params(self) -> dict:
        return {
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "epochs": self.epochs,
            "learning_rate": self.lr,
            "batch_size": self.batch_size,
        }
