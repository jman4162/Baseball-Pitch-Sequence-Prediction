"""Ablation study runner for feature, architecture, data size, and hyperparameter studies."""

import os
from typing import Dict, List, Optional

import mlflow
import numpy as np
import pandas as pd

from ..config import AblationConfig, DataConfig, load_config
from ..data.loader import load_pitch_data, create_sequences
from ..data.preprocessing import encode_categoricals, normalize_numericals, create_splits
from ..models import get_model
from ..paths import get_default_config
from .metrics import bootstrap_confidence_interval, compute_metrics


class AblationRunner:
    """Run ablation studies with MLflow logging."""

    def __init__(
        self,
        ablation_config: AblationConfig,
        data_config: DataConfig,
        models_config_dir: str = None,
    ):
        if models_config_dir is None:
            models_config_dir = str(get_default_config("models"))
        self.cfg = ablation_config
        self.data_cfg = data_config
        self.models_config_dir = models_config_dir

    def _load_and_prepare(self):
        """Load data and return (df, encoders, norm_stats)."""
        df = load_pitch_data(self.data_cfg.data_path)
        df, encoders = encode_categoricals(
            df, [c for c in self.data_cfg.categorical_features if c in df.columns]
        )
        df, norm_stats = normalize_numericals(df, self.data_cfg.numerical_features)
        return df, encoders, norm_stats

    def _get_model_config(self, model_name: str) -> dict:
        config_path = os.path.join(
            self.models_config_dir,
            f"{model_name.replace('logistic_regression', 'logistic')}.yaml",
        )
        if not os.path.exists(config_path):
            config_path = os.path.join(self.models_config_dir, f"{model_name}.yaml")
        return load_config(config_path) if os.path.exists(config_path) else {}

    def _train_and_evaluate(self, model_name, model_cfg, X, y, n_folds=3):
        """Train model with CV and return list of accuracy scores."""
        folds = create_splits(X, y, n_folds=n_folds, random_state=self.data_cfg.random_state)
        scores = []
        for train_idx, test_idx in folds:
            model = get_model(model_name, model_cfg)
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            model.fit(X_train, y_train, X_val=X_test, y_val=y_test)
            y_pred = model.predict(X_test)
            scores.append(compute_metrics(y_test, y_pred)["accuracy"])
        return scores

    def run_feature_ablation(self, model_name: Optional[str] = None) -> pd.DataFrame:
        """Remove each feature group one-at-a-time, measure accuracy drop."""
        model_name = model_name or self.cfg.default_model
        model_cfg = self._get_model_config(model_name)
        df, encoders, norm_stats = self._load_and_prepare()

        mlflow.set_tracking_uri(f"file://{os.path.abspath('experiments')}")
        mlflow.set_experiment(f"ablation_feature_{model_name}")

        # Determine base model type
        temp_model = get_model(model_name, model_cfg)
        is_sequence = temp_model.model_type == "sequence"

        results = []
        for group_name, features in self.cfg.feature_groups.items():
            print(f"  Feature group: {group_name}")

            if is_sequence:
                # For sequence models, select which features to include
                seq_features = []
                for f in features:
                    enc_col = f"{f}_enc"
                    if enc_col in df.columns:
                        seq_features.append(enc_col)
                    elif f in df.columns:
                        seq_features.append(f)
                if not seq_features:
                    continue
                X, y, _ = create_sequences(
                    df,
                    window_size=self.data_cfg.window_size,
                    feature_cols=seq_features,
                    target_col=f"{self.data_cfg.target_col}_enc",
                )
            else:
                tab_features = []
                for f in features:
                    enc_col = f"{f}_enc"
                    if enc_col in df.columns:
                        tab_features.append(enc_col)
                    elif f in df.columns:
                        tab_features.append(f)
                if not tab_features:
                    continue
                X = df[tab_features].values
                y = df[f"{self.data_cfg.target_col}_enc"].values

            with mlflow.start_run(run_name=f"feature_{group_name}"):
                scores = self._train_and_evaluate(model_name, model_cfg, X, y)
                mean, ci_low, ci_high = bootstrap_confidence_interval(scores)
                mlflow.log_metric("accuracy_mean", mean)
                mlflow.log_param("feature_group", group_name)

            results.append({
                "variant": group_name,
                "accuracy": mean,
                "ci_low": ci_low,
                "ci_high": ci_high,
            })

        return pd.DataFrame(results)

    def run_architecture_ablation(self, model_name: Optional[str] = None) -> pd.DataFrame:
        """For neural models, test architecture variants."""
        model_name = model_name or self.cfg.default_model
        base_cfg = self._get_model_config(model_name)
        df, encoders, norm_stats = self._load_and_prepare()

        mlflow.set_tracking_uri(f"file://{os.path.abspath('experiments')}")
        mlflow.set_experiment(f"ablation_architecture_{model_name}")

        variants = self.cfg.architecture_variants.get(model_name, {})
        if not variants:
            print(f"No architecture variants defined for {model_name}")
            return pd.DataFrame()

        X, y, _ = create_sequences(
            df,
            window_size=self.data_cfg.window_size,
            feature_cols=self.data_cfg.sequence_features,
            target_col=f"{self.data_cfg.target_col}_enc",
        )

        results = []
        for param_name, param_values in variants.items():
            for val in param_values:
                variant_cfg = dict(base_cfg)
                variant_cfg[param_name] = val
                variant_cfg["epochs"] = min(variant_cfg.get("epochs", 10), 10)
                label = f"{param_name}={val}"
                print(f"  Architecture variant: {label}")

                with mlflow.start_run(run_name=f"arch_{label}"):
                    scores = self._train_and_evaluate(model_name, variant_cfg, X, y)
                    mean, ci_low, ci_high = bootstrap_confidence_interval(scores)
                    mlflow.log_metric("accuracy_mean", mean)
                    mlflow.log_param("variant", label)

                results.append({
                    "variant": label,
                    "accuracy": mean,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                })

        return pd.DataFrame(results)

    def run_data_ablation(self, model_name: Optional[str] = None) -> pd.DataFrame:
        """Train on varying fractions of data, plot learning curves."""
        model_name = model_name or self.cfg.default_model
        model_cfg = self._get_model_config(model_name)
        df, encoders, norm_stats = self._load_and_prepare()

        mlflow.set_tracking_uri(f"file://{os.path.abspath('experiments')}")
        mlflow.set_experiment(f"ablation_data_{model_name}")

        temp_model = get_model(model_name, model_cfg)
        is_sequence = temp_model.model_type == "sequence"

        if is_sequence:
            X, y, _ = create_sequences(
                df,
                window_size=self.data_cfg.window_size,
                feature_cols=self.data_cfg.sequence_features,
                target_col=f"{self.data_cfg.target_col}_enc",
            )
        else:
            tab_features = []
            for col in self.data_cfg.tabular_features:
                enc_col = f"{col}_enc"
                if enc_col in df.columns:
                    tab_features.append(enc_col)
                elif col in df.columns:
                    tab_features.append(col)
            X = df[tab_features].values
            y = df[f"{self.data_cfg.target_col}_enc"].values

        results = []
        for frac in self.cfg.data_fractions:
            n_samples = int(len(X) * frac)
            X_sub, y_sub = X[:n_samples], y[:n_samples]
            label = f"{frac*100:.0f}%"
            print(f"  Data fraction: {label} ({n_samples} samples)")

            with mlflow.start_run(run_name=f"data_{label}"):
                scores = self._train_and_evaluate(model_name, model_cfg, X_sub, y_sub)
                mean, ci_low, ci_high = bootstrap_confidence_interval(scores)
                mlflow.log_metric("accuracy_mean", mean)
                mlflow.log_param("data_fraction", frac)

            results.append({
                "variant": label,
                "accuracy": mean,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n_samples": n_samples,
            })

        return pd.DataFrame(results)

    def run_hyperparameter_sensitivity(
        self, model_name: Optional[str] = None, param_name: Optional[str] = None, values: Optional[list] = None
    ) -> pd.DataFrame:
        """Vary one hyperparameter, keep others at defaults."""
        model_name = model_name or self.cfg.default_model
        base_cfg = self._get_model_config(model_name)
        df, encoders, norm_stats = self._load_and_prepare()

        mlflow.set_tracking_uri(f"file://{os.path.abspath('experiments')}")
        mlflow.set_experiment(f"ablation_hyperparam_{model_name}")

        if param_name is None and values is None:
            # Run all configured sensitivity studies
            all_results = []
            for pname, pvalues in self.cfg.hyperparameter_sensitivity.items():
                for val in pvalues:
                    variant_cfg = dict(base_cfg)
                    variant_cfg[pname] = val
                    variant_cfg["epochs"] = min(variant_cfg.get("epochs", 10), 10)
                    label = f"{pname}={val}"
                    print(f"  Hyperparam: {label}")

                    temp_model = get_model(model_name, variant_cfg)
                    is_sequence = temp_model.model_type == "sequence"
                    if is_sequence:
                        ws = variant_cfg.get("window_size", self.data_cfg.window_size)
                        if pname == "window_size":
                            ws = val
                        X, y, _ = create_sequences(
                            df, window_size=ws,
                            feature_cols=self.data_cfg.sequence_features,
                            target_col=f"{self.data_cfg.target_col}_enc",
                        )
                    else:
                        tab_features = []
                        for col in self.data_cfg.tabular_features:
                            enc_col = f"{col}_enc"
                            if enc_col in df.columns:
                                tab_features.append(enc_col)
                            elif col in df.columns:
                                tab_features.append(col)
                        X = df[tab_features].values
                        y = df[f"{self.data_cfg.target_col}_enc"].values

                    with mlflow.start_run(run_name=f"hp_{label}"):
                        scores = self._train_and_evaluate(model_name, variant_cfg, X, y)
                        mean, ci_low, ci_high = bootstrap_confidence_interval(scores)
                        mlflow.log_metric("accuracy_mean", mean)

                    all_results.append({
                        "variant": label,
                        "accuracy": mean,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                    })
            return pd.DataFrame(all_results)

        # Single param sweep
        results = []
        for val in values:
            variant_cfg = dict(base_cfg)
            variant_cfg[param_name] = val
            label = f"{param_name}={val}"
            print(f"  Hyperparam: {label}")

            temp_model = get_model(model_name, variant_cfg)
            is_sequence = temp_model.model_type == "sequence"
            if is_sequence:
                X, y, _ = create_sequences(
                    df,
                    window_size=self.data_cfg.window_size,
                    feature_cols=self.data_cfg.sequence_features,
                    target_col=f"{self.data_cfg.target_col}_enc",
                )
            else:
                tab_features = []
                for col in self.data_cfg.tabular_features:
                    enc_col = f"{col}_enc"
                    if enc_col in df.columns:
                        tab_features.append(enc_col)
                    elif col in df.columns:
                        tab_features.append(col)
                X = df[tab_features].values
                y = df[f"{self.data_cfg.target_col}_enc"].values

            with mlflow.start_run(run_name=f"hp_{label}"):
                scores = self._train_and_evaluate(model_name, variant_cfg, X, y)
                mean, ci_low, ci_high = bootstrap_confidence_interval(scores)
                mlflow.log_metric("accuracy_mean", mean)

            results.append({
                "variant": label,
                "accuracy": mean,
                "ci_low": ci_low,
                "ci_high": ci_high,
            })

        return pd.DataFrame(results)
