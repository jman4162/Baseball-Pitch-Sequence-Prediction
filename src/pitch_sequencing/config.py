"""Configuration loading and dataclasses for the pitch sequencing package."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .paths import get_default_data_dir


def load_config(path: str) -> dict:
    """Read a YAML config file and return a dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def _default_data_path() -> str:
    return str(get_default_data_dir() / "baseball_pitch_data.csv")


def _default_hmm_path() -> str:
    return str(get_default_data_dir() / "synthetic_pitch_sequences.csv")


@dataclass
class DataConfig:
    data_path: str = field(default_factory=_default_data_path)
    hmm_data_path: str = field(default_factory=_default_hmm_path)
    target_col: str = "PitchType"
    outcome_col: str = "Outcome"
    test_size: float = 0.2
    n_folds: int = 5
    random_state: int = 42
    window_size: int = 8
    tabular_features: List[str] = field(default_factory=lambda: [
        "Balls", "Strikes", "PitcherType", "PitchNumber",
        "AtBatNumber", "RunnersOn", "ScoreDiff", "PreviousPitchType",
    ])
    sequence_features: List[str] = field(default_factory=lambda: [
        "PitchType_enc", "Balls", "Strikes", "PitcherType_enc",
        "PitchNumber", "RunnersOn", "ScoreDiff",
    ])
    categorical_features: List[str] = field(default_factory=lambda: [
        "PitchType", "PitcherType", "PreviousPitchType", "Outcome",
    ])
    numerical_features: List[str] = field(default_factory=lambda: [
        "Balls", "Strikes", "PitchNumber", "AtBatNumber", "ScoreDiff",
    ])

    @classmethod
    def from_yaml(cls, path: str) -> "DataConfig":
        cfg = load_config(path)
        return cls(**{k: v for k, v in cfg.items() if k in cls.__dataclass_fields__})


@dataclass
class ModelConfig:
    model_type: str = "lstm"
    hyperparameters: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str) -> "ModelConfig":
        cfg = load_config(path)
        model_type = cfg.pop("model_type", "lstm")
        return cls(model_type=model_type, hyperparameters=cfg)


@dataclass
class BenchmarkConfig:
    experiment_name: str = "pitch_type_benchmark"
    models: List[str] = field(default_factory=lambda: [
        "logistic_regression", "random_forest", "hmm",
        "autogluon", "lstm", "cnn1d", "transformer",
    ])
    n_folds: int = 5
    metrics: List[str] = field(default_factory=lambda: [
        "accuracy", "balanced_accuracy", "macro_f1",
        "macro_precision", "macro_recall", "log_loss",
    ])
    statistical_tests: bool = True
    confidence_level: float = 0.95

    @classmethod
    def from_yaml(cls, path: str) -> "BenchmarkConfig":
        cfg = load_config(path)
        return cls(**{k: v for k, v in cfg.items() if k in cls.__dataclass_fields__})


@dataclass
class AblationConfig:
    default_model: str = "lstm"
    feature_groups: Dict[str, List[str]] = field(default_factory=dict)
    data_fractions: List[float] = field(default_factory=lambda: [0.1, 0.25, 0.5, 0.75, 1.0])
    architecture_variants: Dict[str, Dict[str, List]] = field(default_factory=dict)
    hyperparameter_sensitivity: Dict[str, List] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str) -> "AblationConfig":
        cfg = load_config(path)
        return cls(**{k: v for k, v in cfg.items() if k in cls.__dataclass_fields__})
