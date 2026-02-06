# AutoGluon

An AutoML model using AutoGluon's TabularPredictor for automated model selection and ensembling.

## Overview

- **Type**: Tabular
- **Library**: AutoGluon
- **Registry name**: `autogluon`
- **Class**: `AutoGluonModel`

!!! note
    Requires the `autogluon` optional extra: `pip install -e ".[autogluon]"`

## Configuration

```yaml
# configs/models/autogluon.yaml
model_type: autogluon
preset: good_quality
time_limit: null
models_dir: autogluon_pitchtype_models
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `preset` | `good_quality` | AutoGluon quality preset |
| `time_limit` | `null` | Max training time in seconds (null = no limit) |
| `models_dir` | `autogluon_pitchtype_models` | Directory for AutoGluon model artifacts |

## Usage

```python
from pitch_sequencing import get_model

model = get_model("autogluon", {"preset": "good_quality"})
model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

## API Reference

::: pitch_sequencing.models.autogluon_model.AutoGluonModel
