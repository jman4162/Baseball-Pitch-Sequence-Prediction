# Quick Start

This guide walks you through generating data, training a model, and running the benchmark suite.

## Setup

```bash
python -m venv venv
source venv/bin/activate
make install  # pip install -e ".[all,dev]"
```

## 1. Generate Synthetic Data

```bash
make data
# or: pitch-generate --num-games 3000 --at-bats 35 --output-dir ./data
```

This produces ~384K pitch rows with realistic pitcher archetypes, sequence strategies, fatigue modeling, and game context.

## 2. Train a Single Model

```bash
make train MODEL=lstm
# or: pitch-train --model lstm
```

Available models: `logistic_regression`, `random_forest`, `hmm`, `autogluon`, `lstm`, `cnn1d`, `transformer`

## 3. Run the Full Benchmark

```bash
make benchmark
# or: pitch-benchmark
```

This runs all 7 models through 5-fold cross-validation and reports accuracy, F1, and other metrics with bootstrap confidence intervals.

## 4. Run Ablation Studies

```bash
make ablation TYPE=feature
# or: pitch-ablation --type feature --model lstm
```

Ablation types: `feature`, `architecture`, `data`, `hyperparam`

## 5. View Results in MLflow

```bash
make mlflow
# Opens at http://localhost:5000
```

## Python API

You can also use the package programmatically:

```python
from pitch_sequencing import load_pitch_data, get_model, MODEL_REGISTRY

# Load data
df = load_pitch_data("data/baseball_pitch_data.csv")

# List available models
print(list(MODEL_REGISTRY.keys()))

# Create and train a model
model = get_model("random_forest", {"n_estimators": 200, "max_depth": 15})
# ... prepare X_train, y_train ...
model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

See the [User Guide](../user-guide/data-pipeline.md) for a complete walkthrough of the data pipeline, training, and evaluation.
