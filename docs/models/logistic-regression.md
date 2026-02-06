# Logistic Regression

A scikit-learn logistic regression classifier serving as a baseline tabular model.

## Overview

- **Type**: Tabular
- **Library**: scikit-learn
- **Registry name**: `logistic_regression`
- **Class**: `LogisticRegressionModel`

## Configuration

```yaml
# configs/models/logistic.yaml
model_type: logistic_regression
C: 1.0
penalty: l2
class_weight: balanced
max_iter: 1000
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `C` | 1.0 | Inverse regularization strength |
| `penalty` | `l2` | Regularization type |
| `class_weight` | `balanced` | Adjusts weights inversely proportional to class frequencies |
| `max_iter` | 1000 | Maximum iterations for solver |

## Usage

```python
from pitch_sequencing import get_model

model = get_model("logistic_regression", {"C": 1.0, "class_weight": "balanced"})
model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

## API Reference

::: pitch_sequencing.models.baselines.LogisticRegressionModel
