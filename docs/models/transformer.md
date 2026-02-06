# Transformer

A Transformer encoder network using self-attention for pitch sequence prediction.

## Overview

- **Type**: Sequence
- **Library**: PyTorch
- **Registry name**: `transformer`
- **Class**: `TransformerModel`
- **Network**: `PitchTransformer`

## Architecture

```
Input (batch, window_size, n_features)
    → Linear embedding → d_model dimensions
    → Sinusoidal positional encoding
    → TransformerEncoder (2 layers, 4 attention heads)
    → Mean pooling over sequence
    → Fully connected → num_classes
```

## Configuration

```yaml
# configs/models/transformer.yaml
model_type: transformer
d_model: 64
nhead: 4
num_layers: 2
dropout: 0.3
epochs: 20
learning_rate: 0.001
batch_size: 256
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_model` | 64 | Embedding dimension |
| `nhead` | 4 | Number of attention heads |
| `num_layers` | 2 | Number of Transformer encoder layers |
| `dropout` | 0.3 | Dropout rate |
| `epochs` | 20 | Maximum training epochs |
| `learning_rate` | 0.001 | Adam optimizer learning rate |
| `batch_size` | 256 | Training batch size |
| `patience` | 5 | Early stopping patience |

## Usage

```python
from pitch_sequencing import get_model

model = get_model("transformer", {
    "d_model": 64,
    "nhead": 4,
    "num_layers": 2,
    "epochs": 20
})

model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

## API Reference

::: pitch_sequencing.models.transformer.TransformerModel
