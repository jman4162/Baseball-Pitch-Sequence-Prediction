# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository explores baseball pitch sequence prediction using multiple ML approaches. It is structured as a Python package (`pitch_sequencing`) with 7 models, a benchmarking framework, ablation tooling, and MLflow experiment tracking. Original notebooks are preserved in `notebooks/`.

## Quick Start

```bash
source venv/bin/activate
make install        # pip install -e ".[all,dev]"
make data           # regenerate synthetic data
make benchmark      # run all 7 models through 5-fold CV
make test           # run pytest
make mlflow         # launch MLflow UI at localhost:5000
```

## Package Structure

```
src/pitch_sequencing/
├── __init__.py         # Public API (get_model, load_pitch_data, etc.)
├── cli.py              # CLI entry points (pitch-generate, pitch-train, etc.)
├── config.py           # YAML loader + dataclasses
├── paths.py            # Config/data path resolution (importlib.resources)
├── configs/            # Bundled YAML configs (shipped with pip install)
│   ├── data.yaml
│   ├── benchmark.yaml
│   ├── ablation.yaml
│   └── models/         # Per-model hyperparameters
├── data/
│   ├── loader.py       # load_pitch_data(), create_sequences()
│   ├── preprocessing.py # encode, normalize, split
│   └── simulator.py    # BaseballPitchSimulator + generate_dataset()
├── models/
│   ├── __init__.py     # MODEL_REGISTRY + get_model()
│   ├── base.py         # BaseModel ABC (fit/predict/predict_proba)
│   ├── baselines.py    # LogisticRegression, RandomForest
│   ├── hmm_model.py    # HMM (hmmlearn)
│   ├── autogluon_model.py # AutoGluon TabularPredictor
│   ├── lstm.py         # LSTM (PyTorch)
│   ├── cnn1d.py        # 1D-CNN (PyTorch)
│   ├── transformer.py  # Transformer (PyTorch)
│   └── torch_utils.py  # Shared training loop, dataset class
└── evaluation/
    ├── metrics.py      # compute_metrics(), bootstrap_ci(), t-test
    ├── benchmark.py    # BenchmarkRunner (k-fold CV across all models)
    ├── ablation.py     # AblationRunner (feature, arch, data, hyperparam)
    └── visualization.py # Confusion matrix, bar charts, learning curves
```

## Models (7 total)

All models implement `BaseModel` ABC with `fit()`, `predict()`, `predict_proba()`, `get_params()`.

| Model | Type | Key Config |
|-------|------|-----------|
| Logistic Regression | tabular | C=1.0, balanced weights |
| Random Forest | tabular | 200 trees, max_depth=15 |
| HMM | sequence | 1-8 hidden states sweep |
| AutoGluon | tabular | good_quality preset |
| LSTM | sequence | 2-layer, hidden=64, window=8 |
| 1D-CNN | sequence | 3 conv layers [64,128,64], k=3 |
| Transformer | sequence | d_model=64, 4 heads, 2 layers |

`model_type = "tabular"`: expects (X_df, y) with tabular features.
`model_type = "sequence"`: expects (X_3d, y) with shape (n, window_size, n_features).

## Configuration

Configs are bundled YAML files in `src/pitch_sequencing/configs/` (also mirrored in `configs/` for dev):
- `data.yaml` — data paths, features, split ratios, window size
- `benchmark.yaml` — which models, k-folds, metrics
- `ablation.yaml` — ablation study configurations
- `models/*.yaml` — per-model hyperparameters

Path resolution uses `paths.py` (`get_default_config()`, `get_default_data_dir()`).

## CLI Commands

```bash
pitch-generate [--num-games 3000] [--at-bats 35] [--output-dir ./data]
pitch-train --model lstm [--config path/to/lstm.yaml]
pitch-benchmark [--config path/to/benchmark.yaml]
pitch-ablation --type feature|architecture|data|hyperparam [--model lstm]
```

Scripts in `scripts/` are thin wrappers calling `pitch_sequencing.cli`.

## Datasets

- **`data/baseball_pitch_data.csv`** — ~384K rows: Balls, Strikes, PitchType, Outcome, PitcherType, PitchNumber, AtBatNumber, RunnersOn, ScoreDiff, PreviousPitchType
- **`data/synthetic_pitch_sequences.csv`** — 2,500 x 100 pitch sequences for HMM

## Simulator Features

- **Pitcher archetypes**: power (55% FB), finesse (25% FB, 30% CB), slider_specialist (40% SL), balanced (even)
- **Sequence strategies**: 8 pitch patterns boosting follow-up probability by 15-25%
- **Count-dependent outcomes**: Hit rate 5-6% in pitcher's counts to 19-23% in hitter's counts
- **Fatigue modeling**: After threshold (80-95 pitches), shift toward fastballs and balls
- **Game situation**: Runners on base, score differential affect selection

## Key Dependencies

- `torch`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`
- `mlflow` (experiment tracking)
- `pyyaml` (configuration)
- Optional: `autogluon` (AutoML), `hmmlearn` (HMM)

## Pitch Types and Outcomes

- **Pitch types**: Fastball, Slider, Curveball, Changeup
- **Outcomes**: ball, strike, hit
- **Count states**: (balls, strikes) where balls in [0,3], strikes in [0,2]

## Notebooks

Original notebooks are in `notebooks/` and serve as demonstrations:
1. `Baseball_Pitch_Sequence_Simulator.ipynb` — data generation
2. `HMM_Pitch_Predictor.ipynb` — HMM training
3. `AutoGluon_Baseball_Pitch_Prediction.ipynb` — pitch type prediction
4. `AutoGluon_Baseball_Pitch_Outcome_Prediction.ipynb` — outcome prediction
5. `LSTM_Pitch_Predictor.ipynb` — LSTM training
