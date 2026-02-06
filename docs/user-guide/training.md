# Training Models

## Model Types

Models fall into two categories based on their expected input format:

| Type | Input Shape | Models |
|------|------------|--------|
| **Tabular** | `(n_samples, n_features)` | Logistic Regression, Random Forest, AutoGluon |
| **Sequence** | `(n_samples, window_size, n_features)` | LSTM, CNN1D, Transformer, HMM |

## CLI Training

```bash
# Train with default config
pitch-train --model lstm

# Train with custom config
pitch-train --model lstm --config configs/models/lstm.yaml
```

Available model names: `logistic_regression`, `random_forest`, `hmm`, `autogluon`, `lstm`, `cnn1d`, `transformer`

## Python API Training

### Tabular Models

```python
from pitch_sequencing import get_model, load_pitch_data
from pitch_sequencing.data.preprocessing import encode_categoricals, normalize_numericals, create_splits

# Load and preprocess
df = load_pitch_data("data/baseball_pitch_data.csv")
df, encoders = encode_categoricals(df, ["PitchType", "Outcome", "PitcherType", "PreviousPitchType"])
df, stats = normalize_numericals(df, ["PitchNumber", "AtBatNumber", "RunnersOn", "ScoreDiff"])

# Prepare features
feature_cols = ["Balls", "Strikes", "PitcherType_enc", "PreviousPitchType_enc",
                "PitchNumber", "AtBatNumber", "RunnersOn", "ScoreDiff"]
X = df[feature_cols].values
y = df["PitchType_enc"].values

# Split
folds = create_splits(X, y, n_folds=5)
train_idx, test_idx = folds[0]

# Train
model = get_model("random_forest", {"n_estimators": 200, "max_depth": 15})
model.fit(X[train_idx], y[train_idx])

# Predict
predictions = model.predict(X[test_idx])
probabilities = model.predict_proba(X[test_idx])
```

### Sequence Models

```python
from pitch_sequencing import get_model
from pitch_sequencing.data.loader import create_sequences

# Create sequences from preprocessed DataFrame
X_seq, y_seq, _ = create_sequences(df, window_size=8,
    feature_cols=["Balls", "Strikes", "PitchType_enc", ...],
    target_col="PitchType_enc")

folds = create_splits(X_seq, y_seq, n_folds=5)
train_idx, test_idx = folds[0]

# Train LSTM
model = get_model("lstm", {"hidden_size": 64, "num_layers": 2, "epochs": 20})
model.fit(X_seq[train_idx], y_seq[train_idx],
          X_val=X_seq[test_idx], y_val=y_seq[test_idx])

predictions = model.predict(X_seq[test_idx])
```

## Model Configuration

Each model reads hyperparameters from a YAML config file or a dictionary:

```yaml
# configs/models/lstm.yaml
model_type: lstm
hidden_size: 64
num_layers: 2
dropout: 0.3
epochs: 20
learning_rate: 0.001
batch_size: 256
```

See the [Configuration](../configuration.md) page for all model configs.

## Custom Models

To add a new model, implement the `BaseModel` abstract class:

```python
from pitch_sequencing.models.base import BaseModel

class MyModel(BaseModel):
    @property
    def name(self) -> str:
        return "My Custom Model"

    @property
    def model_type(self) -> str:
        return "tabular"  # or "sequence"

    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        # Training logic
        pass

    def predict(self, X):
        # Return class labels
        pass

    def predict_proba(self, X):
        # Return (n_samples, n_classes) probability matrix
        pass
```

Then register it in `models/__init__.py`:

```python
MODEL_REGISTRY["my_model"] = MyModel
```
