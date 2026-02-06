# Baseball Pitch Sequence Prediction

A professional-grade Python package for baseball pitch sequence prediction using 7 ML models, with benchmarking, ablation studies, and MLflow experiment tracking.

## Overview

This project generates synthetic baseball pitch data with realistic pitcher archetypes, pitch sequence strategies, fatigue modeling, and game situation context — then trains and compares multiple models for predicting the next pitch type.

## Models

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

## Key Features

- **Realistic synthetic data** with pitcher archetypes, fatigue, and game context
- **7 prediction models** spanning tabular and sequence architectures
- **Comprehensive benchmarking** with k-fold CV and bootstrap confidence intervals
- **Ablation studies** for feature importance, architecture, data scaling, and hyperparameters
- **MLflow tracking** for experiment management and comparison
- **YAML-driven configuration** for all settings
- **CLI commands** for data generation, training, benchmarking, and ablation

## Quick Links

- [Installation](getting-started/installation.md) — Get up and running
- [Quick Start](getting-started/quickstart.md) — Generate data and train your first model
- [Models Overview](models/index.md) — Compare all 7 models
- [CLI Reference](cli-reference.md) — Command-line interface
- [API Reference](api/index.md) — Python API documentation
