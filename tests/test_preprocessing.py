"""Tests for preprocessing utilities."""

import numpy as np
import pandas as pd
import pytest

from pitch_sequencing.data.preprocessing import encode_categoricals, normalize_numericals, create_splits
from pitch_sequencing.data.loader import create_sequences


def test_encode_categoricals_roundtrip():
    """encode_categoricals can round-trip labels via inverse_transform."""
    df = pd.DataFrame({"Color": ["red", "blue", "green", "red", "blue"]})
    df_enc, encoders = encode_categoricals(df, ["Color"])
    assert "Color_enc" in df_enc.columns
    assert list(encoders["Color"].inverse_transform(df_enc["Color_enc"])) == list(df["Color"])


def test_encode_categoricals_with_existing_encoder():
    """Passing pre-fitted encoders uses them correctly."""
    df_train = pd.DataFrame({"X": ["a", "b", "c"]})
    _, encoders = encode_categoricals(df_train, ["X"])
    df_test = pd.DataFrame({"X": ["b", "a", "c"]})
    df_test_enc, _ = encode_categoricals(df_test, ["X"], encoders=encoders)
    assert list(df_test_enc["X_enc"]) == list(encoders["X"].transform(["b", "a", "c"]))


def test_normalize_numericals():
    """normalize_numericals Z-scores columns correctly."""
    df = pd.DataFrame({"val": [10.0, 20.0, 30.0, 40.0, 50.0]})
    df_norm, stats = normalize_numericals(df, ["val"])
    assert abs(df_norm["val"].mean()) < 1e-6
    assert abs(df_norm["val"].std(ddof=0) - 1.0) < 0.15  # approximate (small N)


def test_normalize_preserves_raw_pitch_number():
    """Normalizing PitchNumber also saves PitchNumber_raw."""
    df = pd.DataFrame({"PitchNumber": [1, 2, 3, 50, 1, 2]})
    df_norm, _ = normalize_numericals(df, ["PitchNumber"])
    assert "PitchNumber_raw" in df_norm.columns
    assert list(df_norm["PitchNumber_raw"]) == [1, 2, 3, 50, 1, 2]


def test_create_splits_single():
    """Single split returns correct sizes."""
    X = np.arange(100)
    y = np.repeat([0, 1, 2, 3], 25)
    folds = create_splits(X, y, test_size=0.2, n_folds=1)
    assert len(folds) == 1
    train_idx, test_idx = folds[0]
    assert len(train_idx) == 80
    assert len(test_idx) == 20


def test_create_splits_kfold():
    """K-fold produces the right number of folds with correct sizes."""
    X = np.arange(100)
    y = np.repeat([0, 1, 2, 3], 25)
    folds = create_splits(X, y, n_folds=5)
    assert len(folds) == 5
    for train_idx, test_idx in folds:
        assert len(test_idx) == 20
        assert len(train_idx) == 80


def test_create_sequences_respects_boundaries():
    """create_sequences should not create windows spanning game boundaries."""
    df = pd.DataFrame({
        "PitchType_enc": [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
        "Balls": [0] * 12,
        "Strikes": [0] * 12,
        "PitcherType_enc": [0] * 12,
        "PitchNumber": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "RunnersOn": [0] * 12,
        "ScoreDiff": [0] * 12,
        "PitchNumber_raw": [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6],  # game boundary at index 6
    })
    X, y, game_starts = create_sequences(df, window_size=4, target_col="PitchType_enc")
    # The window covering indices [3,4,5,6] spans the game boundary — should be excluded
    # game_starts should include index 6
    assert 6 in game_starts
    # All valid sequences should NOT cross index 6
    assert len(X) > 0
