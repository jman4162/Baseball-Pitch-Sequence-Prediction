# 1D-CNN

A 3-layer 1D convolutional neural network for detecting local pitch sequence patterns.

## Overview

- **Type**: Sequence
- **Library**: PyTorch
- **Registry name**: `cnn1d`
- **Class**: `CNN1DModel`
- **Network**: `PitchCNN1D`

## Architecture

```
Input (batch, window_size, n_features)
    → Transpose to (batch, n_features, window_size)
    → Conv1d(n_features, 64, kernel=3) + ReLU + BatchNorm
    → Conv1d(64, 128, kernel=3) + ReLU + BatchNorm
    → Conv1d(128, 64, kernel=3) + ReLU + BatchNorm
    → Adaptive Max Pooling → Dropout
    → Fully connected → num_classes
```

## Configuration

```yaml
# configs/models/cnn1d.yaml
model_type: cnn1d
filters: [64, 128, 64]
kernel_size: 3
dropout: 0.3
epochs: 20
learning_rate: 0.001
batch_size: 256
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `filters` | `[64, 128, 64]` | Number of filters per conv layer |
| `kernel_size` | 3 | Convolution kernel size |
| `dropout` | 0.3 | Dropout rate |
| `epochs` | 20 | Maximum training epochs |
| `learning_rate` | 0.001 | Adam optimizer learning rate |
| `batch_size` | 256 | Training batch size |
| `patience` | 5 | Early stopping patience |

## Usage

```python
from pitch_sequencing import get_model

model = get_model("cnn1d", {
    "filters": [64, 128, 64],
    "kernel_size": 3,
    "epochs": 20
})

model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

## API Reference

::: pitch_sequencing.models.cnn1d.CNN1DModel
