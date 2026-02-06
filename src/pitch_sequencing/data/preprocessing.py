"""Preprocessing utilities for encoding, normalization, and splitting."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder


def encode_categoricals(
    df: pd.DataFrame,
    columns: List[str],
    encoders: Optional[Dict[str, LabelEncoder]] = None,
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """Label-encode categorical columns.

    Args:
        df: Input DataFrame.
        columns: Columns to encode.
        encoders: Pre-fitted encoders to reuse (for test data).

    Returns:
        (df_encoded, encoders_dict) — DataFrame with new *_enc columns and
        the fitted encoders.
    """
    df = df.copy()
    if encoders is None:
        encoders = {}
    for col in columns:
        enc_col = f"{col}_enc"
        if col not in encoders:
            enc = LabelEncoder()
            df[enc_col] = enc.fit_transform(df[col])
            encoders[col] = enc
        else:
            df[enc_col] = encoders[col].transform(df[col])
    return df, encoders


def normalize_numericals(
    df: pd.DataFrame,
    columns: List[str],
    stats: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
    """Z-score normalize numerical columns.

    Also stores the raw PitchNumber (before normalization) as PitchNumber_raw
    so that game-boundary detection still works downstream.

    Args:
        df: Input DataFrame.
        columns: Columns to normalize.
        stats: Pre-computed (mean, std) per column (for test data).

    Returns:
        (df_normalized, stats_dict)
    """
    df = df.copy()
    if stats is None:
        stats = {}

    # Preserve raw columns used for game boundary detection
    for raw_col in ["PitchNumber", "AtBatNumber"]:
        raw_name = f"{raw_col}_raw"
        if raw_col in columns and raw_name not in df.columns:
            df[raw_name] = df[raw_col].values.copy()

    for col in columns:
        if col not in stats:
            mean = df[col].mean()
            std = df[col].std() + 1e-8
            stats[col] = (mean, std)
        else:
            mean, std = stats[col]
        df[col] = (df[col] - mean) / std
    return df, stats


def create_splits(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    n_folds: int = 5,
    stratify: bool = True,
    random_state: int = 42,
    temporal: bool = False,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create train/test splits: either a single split or k-fold CV.

    Args:
        X: Feature array.
        y: Target array.
        test_size: Fraction for test set (single split mode).
        n_folds: Number of CV folds. If 1, performs a single split.
        stratify: Whether to stratify by y.
        random_state: Random seed.
        temporal: If True, use temporal (ordered) splits instead of random.

    Returns:
        List of (train_indices, test_indices) tuples.
    """
    n = len(X)

    if n_folds <= 1:
        if temporal:
            split_idx = int(n * (1 - test_size))
            return [(np.arange(split_idx), np.arange(split_idx, n))]
        strat = y if stratify else None
        train_idx, test_idx = train_test_split(
            np.arange(n), test_size=test_size, stratify=strat, random_state=random_state
        )
        return [(train_idx, test_idx)]

    if temporal:
        fold_size = n // n_folds
        folds = []
        for i in range(n_folds):
            test_start = i * fold_size
            test_end = test_start + fold_size if i < n_folds - 1 else n
            test_idx = np.arange(test_start, test_end)
            train_idx = np.concatenate([np.arange(0, test_start), np.arange(test_end, n)])
            folds.append((train_idx, test_idx))
        return folds

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    return [(train_idx, test_idx) for train_idx, test_idx in skf.split(X, y)]
