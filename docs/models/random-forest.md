# Random Forest

A scikit-learn random forest ensemble classifier for tabular pitch data.

## Overview

- **Type**: Tabular
- **Library**: scikit-learn
- **Registry name**: `random_forest`
- **Class**: `RandomForestModel`

## Configuration

```yaml
# configs/models/random_forest.yaml
model_type: random_forest
n_estimators: 200
max_depth: 15
random_state: 42
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_estimators` | 200 | Number of trees in the forest |
| `max_depth` | 15 | Maximum tree depth |
| `random_state` | 42 | Random seed for reproducibility |

## Usage

```python
from pitch_sequencing import get_model

model = get_model("random_forest", {"n_estimators": 200, "max_depth": 15})
model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

## API Reference

::: pitch_sequencing.models.baselines.RandomForestModel
