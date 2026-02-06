# LSTM

A 2-layer Long Short-Term Memory neural network for sequence-based pitch prediction.

## Overview

- **Type**: Sequence
- **Library**: PyTorch
- **Registry name**: `lstm`
- **Class**: `LSTMModel`
- **Network**: `PitchPredictor`

## Architecture

```
Input (batch, window_size, n_features)
    → LSTM (2 layers, hidden_size=64, dropout=0.3)
    → Take last hidden state
    → Fully connected → num_classes
```

## Configuration

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

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_size` | 64 | LSTM hidden dimension |
| `num_layers` | 2 | Number of stacked LSTM layers |
| `dropout` | 0.3 | Dropout rate |
| `epochs` | 20 | Maximum training epochs |
| `learning_rate` | 0.001 | Adam optimizer learning rate |
| `batch_size` | 32 | Training batch size |
| `patience` | 5 | Early stopping patience |

## Usage

```python
from pitch_sequencing import get_model

model = get_model("lstm", {
    "hidden_size": 64,
    "num_layers": 2,
    "epochs": 20,
    "batch_size": 256
})

# X_train shape: (n_samples, window_size, n_features)
model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

## API Reference

::: pitch_sequencing.models.lstm.LSTMModel
