"""Transformer model for pitch sequence prediction."""

import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .base import BaseModel
from .torch_utils import PitchSequenceDataset, train_torch_model, predict_torch_model


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 32):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class PitchTransformer(nn.Module):
    """Transformer encoder for pitch sequences.

    Architecture:
        Input: (batch, window_size, n_features)
        -> Linear(n_features, d_model) — input projection
        -> + sinusoidal PositionalEncoding
        -> TransformerEncoderLayer x num_layers
        -> Mean pool across sequence dimension
        -> Dropout -> Linear(d_model, num_classes)
    """

    def __init__(
        self,
        input_features: int,
        num_classes: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # mean pool across sequence
        x = self.dropout(x)
        return self.fc(x)


class TransformerModel(BaseModel):
    """Transformer wrapper implementing BaseModel interface."""

    def __init__(self, config=None):
        config = config or {}
        self.d_model = config.get("d_model", 64)
        self.nhead = config.get("nhead", 4)
        self.num_layers = config.get("num_layers", 2)
        self.dim_feedforward = config.get("dim_feedforward", 128)
        self.dropout = config.get("dropout", 0.2)
        self.epochs = config.get("epochs", 30)
        self.lr = config.get("learning_rate", 0.0005)
        self.batch_size = config.get("batch_size", 256)
        self._model = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._history = None

    @property
    def name(self) -> str:
        return "Transformer"

    @property
    def model_type(self) -> str:
        return "sequence"

    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        input_features = X_train.shape[2]
        num_classes = len(np.unique(y_train))

        self._model = PitchTransformer(
            input_features=input_features,
            num_classes=num_classes,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
        )

        train_ds = PitchSequenceDataset(X_train, y_train)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        if X_val is not None and y_val is not None:
            val_ds = PitchSequenceDataset(X_val, y_val)
        else:
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
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_layers": self.num_layers,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout,
            "epochs": self.epochs,
            "learning_rate": self.lr,
        }
