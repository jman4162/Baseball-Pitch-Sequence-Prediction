"""Data loading utilities for pitch sequence prediction."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Optional, Tuple


def load_pitch_data(path: str, filter_none_prev: bool = True) -> pd.DataFrame:
    """Load the main pitch dataset.

    Args:
        path: Path to baseball_pitch_data.csv.
        filter_none_prev: If True, drop rows where PreviousPitchType is 'None'.

    Returns:
        DataFrame with pitch data.
    """
    df = pd.read_csv(path)
    if filter_none_prev:
        df = df[df["PreviousPitchType"] != "None"].reset_index(drop=True)
    return df


def create_sequences(
    df: pd.DataFrame,
    window_size: int = 8,
    feature_cols: Optional[List[str]] = None,
    target_col: str = "PitchType_enc",
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Create sliding-window sequences respecting game boundaries.

    Game boundaries are detected via PitchNumber resets (the raw column must
    be present or reconstructable). The function expects that categorical
    columns have already been encoded (e.g. PitchType_enc, PitcherType_enc).

    Args:
        df: DataFrame with encoded features.
        window_size: Number of previous timesteps per sample.
        feature_cols: Columns to include as features in each timestep.
        target_col: Column to predict.

    Returns:
        (X, y, game_starts) where X has shape (n_samples, window_size, n_features),
        y has shape (n_samples,), and game_starts lists the indices where new games start.
    """
    if feature_cols is None:
        feature_cols = [
            "PitchType_enc", "Balls", "Strikes", "PitcherType_enc",
            "PitchNumber", "RunnersOn", "ScoreDiff",
        ]

    features = df[feature_cols].values
    targets = df[target_col].values

    # Detect game boundaries using AtBatNumber resets (drops from high to low).
    # Falls back to PitchNumber drops if AtBatNumber is not available.
    if "AtBatNumber_raw" in df.columns:
        boundary_col = df["AtBatNumber_raw"].values
    elif "AtBatNumber" in df.columns:
        boundary_col = df["AtBatNumber"].values
    elif "PitchNumber_raw" in df.columns:
        boundary_col = df["PitchNumber_raw"].values
    else:
        boundary_col = df["PitchNumber"].values
    game_starts = set(np.where(np.diff(boundary_col, prepend=boundary_col[0] + 1) < 0)[0])

    X_sequences = []
    y_targets = []

    for i in range(window_size, len(features)):
        window_range = range(i - window_size + 1, i + 1)
        if any(idx in game_starts for idx in window_range):
            continue
        X_sequences.append(features[i - window_size:i])
        y_targets.append(targets[i])

    X = np.array(X_sequences, dtype=np.float32)
    y = np.array(y_targets, dtype=np.int64)
    return X, y, sorted(game_starts)


def load_hmm_sequences(path: str) -> Tuple[np.ndarray, LabelEncoder]:
    """Load the HMM synthetic pitch sequences dataset.

    Args:
        path: Path to synthetic_pitch_sequences.csv.

    Returns:
        (flat_sequences, encoder) where flat_sequences is shape (n_total, 1)
        of encoded pitch types, and encoder can invert labels.
    """
    data = pd.read_csv(path).dropna()
    encoder = LabelEncoder()
    encoded = data.apply(encoder.fit_transform)
    flat = encoded.values.flatten().reshape(-1, 1)
    return flat, encoder
