"""Model registry for all pitch prediction models."""

from .baselines import LogisticRegressionModel, RandomForestModel
from .hmm_model import HMMModel
from .autogluon_model import AutoGluonModel
from .lstm import LSTMModel
from .cnn1d import CNN1DModel
from .transformer import TransformerModel

MODEL_REGISTRY = {
    "logistic_regression": LogisticRegressionModel,
    "random_forest": RandomForestModel,
    "hmm": HMMModel,
    "autogluon": AutoGluonModel,
    "lstm": LSTMModel,
    "cnn1d": CNN1DModel,
    "transformer": TransformerModel,
}


def get_model(name, config=None):
    """Instantiate a model by registry name.

    Args:
        name: Key in MODEL_REGISTRY (e.g. 'lstm', 'random_forest').
        config: Optional dict of hyperparameters.

    Returns:
        Instance of the model class.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](config)
