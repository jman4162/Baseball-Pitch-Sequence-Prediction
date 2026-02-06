# Models Overview

All 7 models implement the `BaseModel` abstract interface and are accessible through the `MODEL_REGISTRY`.

## Model Comparison

| Model | Type | Key Config | Strengths |
|-------|------|-----------|-----------|
| [Logistic Regression](logistic-regression.md) | Tabular | C=1.0, balanced weights | Fast, interpretable baseline |
| [Random Forest](random-forest.md) | Tabular | 200 trees, max_depth=15 | Handles non-linear relationships |
| [HMM](hmm.md) | Sequence | 1-8 hidden states | Captures latent pitch states |
| [AutoGluon](autogluon.md) | Tabular | good_quality preset | Automated model selection |
| [LSTM](lstm.md) | Sequence | 2-layer, hidden=64, window=8 | Long-range sequence dependencies |
| [1D-CNN](cnn1d.md) | Sequence | 3 conv layers, kernel=3 | Local pattern detection |
| [Transformer](transformer.md) | Sequence | d_model=64, 4 heads, 2 layers | Self-attention over sequences |

## BaseModel Interface

All models implement the following abstract interface:

```python
from pitch_sequencing.models.base import BaseModel

class BaseModel(ABC):
    @property
    def name(self) -> str:
        """Human-readable model name."""

    @property
    def model_type(self) -> str:
        """'tabular' or 'sequence' — determines input shape."""

    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """Train the model."""

    def predict(self, X) -> np.ndarray:
        """Return predicted class labels."""

    def predict_proba(self, X) -> np.ndarray:
        """Return class probabilities (n_samples x n_classes)."""

    def get_params(self) -> dict:
        """Return model hyperparameters."""
```

### Input Shapes

- **Tabular models** (`model_type = "tabular"`): Expect `(n_samples, n_features)` NumPy arrays or DataFrames
- **Sequence models** (`model_type = "sequence"`): Expect `(n_samples, window_size, n_features)` 3D arrays

## Model Registry

Models are accessed by name through the registry:

```python
from pitch_sequencing import get_model, MODEL_REGISTRY

# List all registered models
print(list(MODEL_REGISTRY.keys()))
# ['logistic_regression', 'random_forest', 'hmm', 'autogluon', 'lstm', 'cnn1d', 'transformer']

# Instantiate a model with config
model = get_model("lstm", {"hidden_size": 64, "num_layers": 2})
```
