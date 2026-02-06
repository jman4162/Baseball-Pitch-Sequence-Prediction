"""1D-CNN model for pitch sequence prediction."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .base import BaseModel
from .torch_utils import PitchSequenceDataset, train_torch_model, predict_torch_model


class PitchCNN1D(nn.Module):
    """1D Convolutional network for pitch sequences.

    Architecture:
        Input: (batch, window_size, n_features)
        -> Transpose to (batch, n_features, window_size) for Conv1d
        -> Conv1d(in, 64, k=3) + ReLU + BatchNorm
        -> Conv1d(64, 128, k=3) + ReLU + BatchNorm
        -> Conv1d(128, 64, k=3) + ReLU + BatchNorm
        -> AdaptiveMaxPool1d(1) -> squeeze
        -> Dropout -> Linear(64, num_classes)
    """

    def __init__(self, input_features, num_classes, filters=None, kernel_size=3, dropout=0.3):
        super().__init__()
        if filters is None:
            filters = [64, 128, 64]

        layers = []
        in_channels = input_features
        for out_channels in filters:
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
                nn.BatchNorm1d(out_channels),
            ])
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(filters[-1], num_classes)

    def forward(self, x):
        # x: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)
        x = self.conv_layers(x)
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)


class CNN1DModel(BaseModel):
    """1D-CNN wrapper implementing BaseModel interface."""

    def __init__(self, config=None):
        config = config or {}
        self.filters = config.get("filters", [64, 128, 64])
        self.kernel_size = config.get("kernel_size", 3)
        self.dropout = config.get("dropout", 0.3)
        self.epochs = config.get("epochs", 30)
        self.lr = config.get("learning_rate", 0.001)
        self.batch_size = config.get("batch_size", 256)
        self._model = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._history = None

    @property
    def name(self) -> str:
        return "1D-CNN"

    @property
    def model_type(self) -> str:
        return "sequence"

    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        input_features = X_train.shape[2]
        num_classes = len(np.unique(y_train))

        self._model = PitchCNN1D(
            input_features=input_features,
            num_classes=num_classes,
            filters=self.filters,
            kernel_size=self.kernel_size,
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
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "dropout": self.dropout,
            "epochs": self.epochs,
            "learning_rate": self.lr,
        }
