"""Benchmark runner: k-fold cross-validation across all models."""

import os
from pathlib import Path
from typing import Dict, List, Optional

import mlflow
import numpy as np
import pandas as pd

from ..config import BenchmarkConfig, DataConfig, ModelConfig, load_config
from ..data.loader import load_pitch_data, create_sequences, load_hmm_sequences
from ..data.preprocessing import encode_categoricals, normalize_numericals, create_splits
from ..models import get_model
from ..paths import get_default_config
from .metrics import bootstrap_confidence_interval, compute_metrics, paired_t_test, compute_effect_size


class BenchmarkRunner:
    """Run all models through k-fold CV and produce comparison results."""

    def __init__(
        self,
        benchmark_config: BenchmarkConfig,
        data_config: DataConfig,
        models_config_dir: str = None,
    ):
        if models_config_dir is None:
            models_config_dir = str(get_default_config("models"))
        self.cfg = benchmark_config
        self.data_cfg = data_config
        self.models_config_dir = models_config_dir
        self._results: Optional[pd.DataFrame] = None

    def run(self) -> pd.DataFrame:
        """Run all models through k-fold CV, return results DataFrame."""
        # Set up MLflow
        mlflow.set_tracking_uri(f"file://{os.path.abspath('experiments')}")
        mlflow.set_experiment(self.cfg.experiment_name)

        # Load and preprocess data (shared across all models)
        df = load_pitch_data(self.data_cfg.data_path)
        df, encoders = encode_categoricals(
            df, [c for c in self.data_cfg.categorical_features if c in df.columns]
        )
        df, norm_stats = normalize_numericals(df, self.data_cfg.numerical_features)

        # Prepare tabular data
        tabular_feature_cols = []
        for col in self.data_cfg.tabular_features:
            enc_col = f"{col}_enc"
            if enc_col in df.columns:
                tabular_feature_cols.append(enc_col)
            elif col in df.columns:
                tabular_feature_cols.append(col)

        X_tab = df[tabular_feature_cols].values
        y_tab = df[f"{self.data_cfg.target_col}_enc"].values

        # Prepare sequence data
        X_seq, y_seq, _ = create_sequences(
            df,
            window_size=self.data_cfg.window_size,
            feature_cols=self.data_cfg.sequence_features,
            target_col=f"{self.data_cfg.target_col}_enc",
        )

        # Prepare HMM data
        hmm_flat, hmm_encoder = load_hmm_sequences(self.data_cfg.hmm_data_path)

        # Create folds
        tab_folds = create_splits(X_tab, y_tab, n_folds=self.cfg.n_folds,
                                  random_state=self.data_cfg.random_state)
        seq_folds = create_splits(X_seq, y_seq, n_folds=self.cfg.n_folds,
                                  random_state=self.data_cfg.random_state)
        hmm_folds = create_splits(hmm_flat, hmm_flat.flatten(), n_folds=self.cfg.n_folds,
                                  random_state=self.data_cfg.random_state)

        all_results = {}

        for model_name in self.cfg.models:
            print(f"\n{'='*60}")
            print(f"Running: {model_name}")
            print(f"{'='*60}")

            # Load model config
            config_path = os.path.join(self.models_config_dir, f"{model_name.replace('logistic_regression', 'logistic')}.yaml")
            if not os.path.exists(config_path):
                config_path = os.path.join(self.models_config_dir, f"{model_name}.yaml")
            model_cfg = load_config(config_path) if os.path.exists(config_path) else {}

            fold_metrics = {m: [] for m in self.cfg.metrics}

            # Determine data and folds
            model_cls = get_model(model_name, model_cfg).__class__
            temp_model = model_cls(model_cfg)

            if model_name == "hmm":
                data_X, data_y, folds = hmm_flat, hmm_flat.flatten(), hmm_folds
            elif temp_model.model_type == "sequence":
                data_X, data_y, folds = X_seq, y_seq, seq_folds
            else:
                data_X, data_y, folds = X_tab, y_tab, tab_folds

            for fold_idx, (train_idx, test_idx) in enumerate(folds):
                print(f"  Fold {fold_idx + 1}/{self.cfg.n_folds}...", end=" ", flush=True)

                model = get_model(model_name, model_cfg)

                X_train, y_train = data_X[train_idx], data_y[train_idx]
                X_test, y_test = data_X[test_idx], data_y[test_idx]

                with mlflow.start_run(run_name=f"{model_name}_fold{fold_idx}"):
                    mlflow.log_params(model.get_params())
                    mlflow.log_param("model_name", model_name)
                    mlflow.log_param("fold", fold_idx)

                    model.fit(X_train, y_train, X_val=X_test, y_val=y_test)

                    y_pred = model.predict(X_test)
                    try:
                        y_proba = model.predict_proba(X_test)
                    except Exception:
                        y_proba = None

                    metrics = compute_metrics(y_test, y_pred, y_proba)

                    for m in self.cfg.metrics:
                        val = metrics.get(m, float("nan"))
                        fold_metrics[m].append(val)
                        mlflow.log_metric(m, val)

                print(f"acc={metrics['accuracy']:.4f}")

            # Aggregate fold metrics
            model_result = {"model": model_name}
            for m in self.cfg.metrics:
                scores = fold_metrics[m]
                mean, ci_low, ci_high = bootstrap_confidence_interval(scores, self.cfg.confidence_level)
                model_result[f"{m}_mean"] = mean
                model_result[f"{m}_ci_low"] = ci_low
                model_result[f"{m}_ci_high"] = ci_high
                model_result[f"{m}_scores"] = scores

            all_results[model_name] = model_result

        self._results = pd.DataFrame([v for v in all_results.values()])

        # Statistical tests between all pairs
        if self.cfg.statistical_tests and len(self.cfg.models) > 1:
            print("\nStatistical Tests (paired t-test on accuracy):")
            models = list(all_results.keys())
            for i in range(len(models)):
                for j in range(i + 1, len(models)):
                    a_scores = all_results[models[i]]["accuracy_scores"]
                    b_scores = all_results[models[j]]["accuracy_scores"]
                    t_stat, p_val = paired_t_test(a_scores, b_scores)
                    d = compute_effect_size(a_scores, b_scores)
                    sig = "*" if p_val < 0.05 else ""
                    print(f"  {models[i]} vs {models[j]}: t={t_stat:.3f}, p={p_val:.4f}{sig}, d={d:.3f}")

        return self._results

    def summary_table(self) -> str:
        """Return a markdown-formatted comparison table."""
        if self._results is None:
            return "No results yet. Run benchmark first."

        lines = []
        header_metrics = [m for m in self.cfg.metrics if m != "log_loss"]
        header = "| Model | " + " | ".join(m.replace("_", " ").title() for m in header_metrics) + " |"
        sep = "|" + "---|" * (len(header_metrics) + 1)
        lines.append(header)
        lines.append(sep)

        for _, row in self._results.iterrows():
            cells = [row["model"]]
            for m in header_metrics:
                mean = row.get(f"{m}_mean", float("nan"))
                ci_low = row.get(f"{m}_ci_low", float("nan"))
                ci_high = row.get(f"{m}_ci_high", float("nan"))
                half_width = (ci_high - ci_low) / 2
                cells.append(f"{mean:.3f} +/- {half_width:.3f}")
            lines.append("| " + " | ".join(cells) + " |")

        return "\n".join(lines)
