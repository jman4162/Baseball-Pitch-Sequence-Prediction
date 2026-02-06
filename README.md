# Baseball Pitch Sequence Prediction

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-F7931E.svg)](https://scikit-learn.org/)
[![MLflow](https://img.shields.io/badge/MLflow-2.10%2B-0194E2.svg)](https://mlflow.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A professional-grade Python package for baseball pitch sequence prediction using 7 ML models, with benchmarking, ablation studies, and MLflow experiment tracking.

## Overview

This project generates synthetic baseball pitch data with realistic pitcher archetypes, pitch sequence strategies, fatigue modeling, and game situation context — then trains and compares multiple models for predicting the next pitch type.

### Models

| Model | Type | Description |
|-------|------|-------------|
| **Logistic Regression** | Tabular | Baseline linear classifier |
| **Random Forest** | Tabular | Ensemble of decision trees |
| **HMM** | Sequence | Hidden Markov Model (hmmlearn) |
| **AutoGluon** | Tabular | AutoML with model ensembling |
| **LSTM** | Sequence | 2-layer LSTM neural network |
| **1D-CNN** | Sequence | 3-layer convolutional network |
| **Transformer** | Sequence | Self-attention encoder |

All models share a unified interface (`fit`, `predict`, `predict_proba`) and are benchmarked via k-fold cross-validation with bootstrap confidence intervals and paired statistical tests.

## Installation

```bash
# From source (development)
pip install -e ".[all,dev]"

# With all optional dependencies (AutoGluon + hmmlearn)
pip install pitch-sequencing[all]

# After install, generate training data:
pitch-generate --output-dir ./data
```

## Quick Start

```bash
# Set up environment
python -m venv venv
source venv/bin/activate
make install            # pip install -e ".[all,dev]"

# Generate synthetic data
make data               # or: pitch-generate

# Train a single model
make train MODEL=lstm   # or: pitch-train --model lstm

# Run full benchmark (all 7 models, 5-fold CV)
make benchmark          # or: pitch-benchmark

# Run ablation studies
make ablation           # or: pitch-ablation --type feature --model lstm

# Launch MLflow UI
make mlflow             # opens at http://localhost:5000

# Run tests
make test
```

### CLI Commands

After installation, these commands are available on your PATH:

| Command | Description |
|---------|-------------|
| `pitch-generate` | Generate synthetic pitch datasets |
| `pitch-train --model <name>` | Train a single model |
| `pitch-benchmark` | Run full benchmark suite |
| `pitch-ablation --type <type>` | Run ablation studies |

## Project Structure

```
├── pyproject.toml              # PEP 621 packaging + CLI entry points
├── Makefile                    # Common commands
├── src/pitch_sequencing/       # Main package
│   ├── __init__.py             # Public API (get_model, load_pitch_data, etc.)
│   ├── cli.py                  # CLI entry points (pitch-generate, etc.)
│   ├── config.py               # Config loading + dataclasses
│   ├── paths.py                # Config/data path resolution
│   ├── configs/                # Bundled YAML configs (ship with pip install)
│   │   ├── data.yaml
│   │   ├── benchmark.yaml
│   │   ├── ablation.yaml
│   │   └── models/             # Per-model hyperparameters
│   ├── data/                   # Data loading, preprocessing, simulation
│   ├── models/                 # All 7 model implementations
│   └── evaluation/             # Metrics, benchmarking, ablation, visualization
├── scripts/                    # Thin CLI wrappers (for make targets)
├── configs/                    # Dev-time config copies (mirrored in package)
├── notebooks/                  # Original Jupyter notebooks
├── data/                       # Generated datasets (not packaged)
├── experiments/                # MLflow artifacts (gitignored)
└── tests/                      # pytest test suite
```

## Synthetic Data

The simulator generates ~384K pitch rows per run with:

- **Pitcher archetypes**: power, finesse, slider_specialist, balanced — each with distinct pitch distributions
- **Sequence strategies**: 8 multi-pitch patterns (e.g., FB-FB→CH, SL-SL→FB) that create learnable sequential dependencies
- **Count-dependent outcomes**: Hit rates from 5-6% (pitcher's counts) to 19-23% (hitter's counts)
- **Fatigue modeling**: Pitch selection degrades after archetype-specific thresholds (80-95 pitches)
- **Game situation**: Runners on base and score differential affect pitch selection

### Dataset Columns

`Balls, Strikes, PitchType, Outcome, PitcherType, PitchNumber, AtBatNumber, RunnersOn, ScoreDiff, PreviousPitchType`

## Configuration

All settings are YAML-driven:

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

## Evaluation

- **Metrics**: Accuracy, balanced accuracy, macro precision/recall/F1, log loss, per-class metrics
- **Benchmarking**: k-fold CV with bootstrap 95% confidence intervals
- **Statistical tests**: Paired t-tests and Cohen's d effect sizes between models
- **Ablation studies**: Feature importance, architecture variants, data scaling, hyperparameter sensitivity
- **MLflow tracking**: All experiments logged with parameters, metrics, and artifacts

## Notebooks

Original exploratory notebooks are preserved in `notebooks/` and can be run via Jupyter or Google Colab. They now import from the `pitch_sequencing` package.

## License

See [LICENSE](LICENSE) file.
