"""Shared PyTorch training utilities for sequence models."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Optional, Tuple


class PitchSequenceDataset(Dataset):
    """Wraps numpy arrays as a PyTorch Dataset."""

    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.LongTensor(targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


def train_torch_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 20,
    lr: float = 0.001,
    patience: int = 5,
    device: Optional[torch.device] = None,
) -> Dict:
    """Train a PyTorch model with Adam, LR scheduling, early stopping, and gradient clipping.

    Returns:
        Dictionary with train_losses, val_losses, val_accuracies.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        # Train
        model.train()
        total_train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validate
        model.eval()
        total_val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                total_val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_acc = correct / total
        val_accuracies.append(val_acc)

        scheduler.step(avg_val_loss)

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
    }


def predict_torch_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference and return (predictions, probabilities).

    Returns:
        (predictions, probabilities) where predictions has shape (n,)
        and probabilities has shape (n, num_classes).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    all_preds = []
    all_probs = []
    with torch.no_grad():
        for batch_X, _ in data_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_preds), np.concatenate(all_probs)
