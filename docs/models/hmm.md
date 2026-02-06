# Hidden Markov Model (HMM)

A sequence model using hmmlearn's CategoricalHMM to capture latent pitch states.

## Overview

- **Type**: Sequence
- **Library**: hmmlearn
- **Registry name**: `hmm`
- **Class**: `HMMModel`

!!! note
    Requires the `hmm` optional extra: `pip install -e ".[hmm]"`

## Configuration

```yaml
# configs/models/hmm.yaml
model_type: hmm
min_components: 1
max_components: 8
n_iter: 100
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_components` | 1 | Minimum hidden states to try |
| `max_components` | 8 | Maximum hidden states to try |
| `n_iter` | 100 | EM iterations per fit |

## How It Works

During `fit()`, the model sweeps the number of hidden states from `min_components` to `max_components`, training a CategoricalHMM for each value. The best model is selected by validation accuracy.

## Usage

```python
from pitch_sequencing import get_model

model = get_model("hmm", {"min_components": 1, "max_components": 8})
model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

predictions = model.predict(X_test)
```

## API Reference

::: pitch_sequencing.models.hmm_model.HMMModel
