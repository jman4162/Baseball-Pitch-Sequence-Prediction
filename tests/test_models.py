"""Tests for model instantiation and basic fit/predict."""

import numpy as np
import pytest

from pitch_sequencing.models import MODEL_REGISTRY, get_model


def _make_tabular_data(n=50, n_features=8, n_classes=4):
    rng = np.random.RandomState(42)
    X = rng.randn(n, n_features).astype(np.float32)
    y = rng.randint(0, n_classes, n)
    return X, y


def _make_sequence_data(n=50, window=8, n_features=7, n_classes=4):
    rng = np.random.RandomState(42)
    X = rng.randn(n, window, n_features).astype(np.float32)
    y = rng.randint(0, n_classes, n)
    return X, y


def test_registry_has_all_models():
    """Model registry contains all 7 expected models."""
    expected = {"logistic_regression", "random_forest", "hmm", "autogluon", "lstm", "cnn1d", "transformer"}
    assert expected == set(MODEL_REGISTRY.keys())


def test_get_model_unknown_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        get_model("nonexistent_model")


@pytest.mark.parametrize("model_name", ["logistic_regression", "random_forest"])
def test_tabular_model_fit_predict(model_name):
    """Tabular models can fit and predict on small data."""
    X, y = _make_tabular_data()
    model = get_model(model_name)
    assert model.model_type == "tabular"
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(X),)
    proba = model.predict_proba(X)
    assert proba.shape[0] == len(X)
    assert proba.shape[1] >= 2


@pytest.mark.parametrize("model_name", ["lstm", "cnn1d", "transformer"])
def test_sequence_model_fit_predict(model_name):
    """Sequence models can fit and predict on small data."""
    X, y = _make_sequence_data()
    config = {"epochs": 2, "batch_size": 16}
    model = get_model(model_name, config)
    assert model.model_type == "sequence"
    split = 40
    model.fit(X[:split], y[:split], X_val=X[split:], y_val=y[split:])
    preds = model.predict(X)
    assert preds.shape == (len(X),)
    proba = model.predict_proba(X)
    assert proba.shape[0] == len(X)


def test_hmm_model_fit_predict():
    """HMM model can fit and predict on flat sequence data."""
    rng = np.random.RandomState(42)
    X = rng.randint(0, 4, (200, 1))
    model = get_model("hmm", {"min_components": 1, "max_components": 3, "n_iter": 10})
    assert model.model_type == "sequence"
    model.fit(X[:160], X[:160].flatten(), X_val=X[160:], y_val=X[160:].flatten())
    preds = model.predict(X)
    assert preds.shape == (200,)
