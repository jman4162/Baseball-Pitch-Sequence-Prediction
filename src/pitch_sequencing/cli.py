"""CLI entry points for the pitch-sequencing package.

These functions are referenced by ``[project.scripts]`` in pyproject.toml and
can also be called from the thin wrapper scripts in ``scripts/``.
"""

import argparse
import os
from pathlib import Path

from .paths import get_default_config, get_default_data_dir


# ---------------------------------------------------------------------------
# pitch-generate
# ---------------------------------------------------------------------------

def generate_main():
    """Generate synthetic baseball pitch datasets."""
    parser = argparse.ArgumentParser(description="Generate synthetic baseball pitch data")
    parser.add_argument("--num-games", type=int, default=3000, help="Number of games to simulate")
    parser.add_argument("--at-bats", type=int, default=35, help="At-bats per game")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory")
    args = parser.parse_args()

    from .data.simulator import generate_dataset, generate_hmm_sequences

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating main dataset ({args.num_games} games, {args.at_bats} at-bats each)...")
    df = generate_dataset(num_games=args.num_games, at_bats_per_game=args.at_bats, seed=args.seed)
    path = output_dir / "baseball_pitch_data.csv"
    df.to_csv(path, index=False)
    print(f"  Saved {len(df)} rows to {path}")
    print(f"  Pitch distribution:\n{df['PitchType'].value_counts(normalize=True).round(3).to_string()}")

    print("\nGenerating HMM sequences dataset...")
    hmm_df = generate_hmm_sequences(num_sequences=2500, sequence_length=100, seed=args.seed)
    hmm_path = output_dir / "synthetic_pitch_sequences.csv"
    hmm_df.to_csv(hmm_path, index=False)
    print(f"  Saved {len(hmm_df)} sequences to {hmm_path}")

    print("\nDone!")


# ---------------------------------------------------------------------------
# pitch-train
# ---------------------------------------------------------------------------

def train_main():
    """Train a single pitch prediction model."""
    parser = argparse.ArgumentParser(description="Train a single pitch prediction model")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g. lstm, random_forest)")
    parser.add_argument("--config", type=str, default=None, help="Path to model config YAML")
    parser.add_argument("--data-config", type=str, default=None, help="Path to data config")
    args = parser.parse_args()

    import mlflow
    import numpy as np

    from .config import DataConfig, load_config
    from .data.loader import load_pitch_data, create_sequences, load_hmm_sequences
    from .data.preprocessing import encode_categoricals, normalize_numericals
    from .models import get_model
    from .evaluation.metrics import compute_metrics

    data_config_path = args.data_config or str(get_default_config("data.yaml"))
    data_cfg = DataConfig.from_yaml(data_config_path)

    # Load model config
    if args.config:
        model_cfg = load_config(args.config)
    else:
        default_path = str(get_default_config(f"models/{args.model.replace('logistic_regression', 'logistic')}.yaml"))
        if os.path.exists(default_path):
            model_cfg = load_config(default_path)
        else:
            model_cfg = {}

    # Load and prepare data
    print(f"Loading data from {data_cfg.data_path}...")
    df = load_pitch_data(data_cfg.data_path)
    df, encoders = encode_categoricals(
        df, [c for c in data_cfg.categorical_features if c in df.columns]
    )
    df, norm_stats = normalize_numericals(df, data_cfg.numerical_features)

    model = get_model(args.model, model_cfg)
    print(f"Training {model.name} (type={model.model_type})...")

    if args.model == "hmm":
        from sklearn.model_selection import train_test_split
        hmm_flat, hmm_enc = load_hmm_sequences(data_cfg.hmm_data_path)
        X_train, X_test = train_test_split(hmm_flat, test_size=data_cfg.test_size, random_state=data_cfg.random_state)
        model.fit(X_train, X_train.flatten(), X_val=X_test, y_val=X_test.flatten())
        y_pred = model.predict(X_test)
        y_test = X_test.flatten()
    elif model.model_type == "sequence":
        X, y, _ = create_sequences(
            df, window_size=data_cfg.window_size,
            feature_cols=data_cfg.sequence_features,
            target_col=f"{data_cfg.target_col}_enc",
        )
        split = int(len(X) * (1 - data_cfg.test_size))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        model.fit(X_train, y_train, X_val=X_test, y_val=y_test)
        y_pred = model.predict(X_test)
    else:
        tab_features = []
        for col in data_cfg.tabular_features:
            enc_col = f"{col}_enc"
            if enc_col in df.columns:
                tab_features.append(enc_col)
            elif col in df.columns:
                tab_features.append(col)
        X = df[tab_features].values
        y = df[f"{data_cfg.target_col}_enc"].values
        split = int(len(X) * (1 - data_cfg.test_size))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    metrics = compute_metrics(y_test, y_pred)
    print(f"\nResults for {model.name}:")
    print(f"  Accuracy:          {metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"  Macro F1:          {metrics['macro_f1']:.4f}")
    print(f"  Macro Precision:   {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall:      {metrics['macro_recall']:.4f}")

    # Log to MLflow
    mlflow.set_tracking_uri(f"file://{os.path.abspath('experiments')}")
    mlflow.set_experiment(f"train_{args.model}")
    with mlflow.start_run(run_name=f"{args.model}_single"):
        mlflow.log_params(model.get_params())
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, v)
    print("\nLogged to MLflow.")


# ---------------------------------------------------------------------------
# pitch-benchmark
# ---------------------------------------------------------------------------

def benchmark_main():
    """Run the full benchmark suite across all models."""
    parser = argparse.ArgumentParser(description="Run pitch prediction benchmark")
    parser.add_argument("--config", type=str, default=None, help="Benchmark config path")
    parser.add_argument("--data-config", type=str, default=None, help="Data config path")
    parser.add_argument("--models-dir", type=str, default=None, help="Models config directory")
    args = parser.parse_args()

    from .config import BenchmarkConfig, DataConfig
    from .evaluation.benchmark import BenchmarkRunner

    bench_config_path = args.config or str(get_default_config("benchmark.yaml"))
    data_config_path = args.data_config or str(get_default_config("data.yaml"))
    models_dir = args.models_dir or str(get_default_config("models"))

    bench_cfg = BenchmarkConfig.from_yaml(bench_config_path)
    data_cfg = DataConfig.from_yaml(data_config_path)

    runner = BenchmarkRunner(bench_cfg, data_cfg, models_config_dir=models_dir)
    results = runner.run()

    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(runner.summary_table())

    os.makedirs("experiments", exist_ok=True)
    results.to_csv("experiments/benchmark_results.csv", index=False)
    print("\nResults saved to experiments/benchmark_results.csv")


# ---------------------------------------------------------------------------
# pitch-ablation
# ---------------------------------------------------------------------------

def ablation_main():
    """Run ablation studies."""
    parser = argparse.ArgumentParser(description="Run ablation studies")
    parser.add_argument("--type", type=str, required=True,
                        choices=["feature", "architecture", "data", "hyperparam"],
                        help="Type of ablation study")
    parser.add_argument("--model", type=str, default=None, help="Model name (default: from config)")
    parser.add_argument("--config", type=str, default=None, help="Ablation config path")
    parser.add_argument("--data-config", type=str, default=None, help="Data config path")
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    from .config import AblationConfig, DataConfig
    from .evaluation.ablation import AblationRunner
    from .evaluation.visualization import plot_ablation_results

    abl_config_path = args.config or str(get_default_config("ablation.yaml"))
    data_config_path = args.data_config or str(get_default_config("data.yaml"))

    abl_cfg = AblationConfig.from_yaml(abl_config_path)
    data_cfg = DataConfig.from_yaml(data_config_path)

    runner = AblationRunner(abl_cfg, data_cfg)

    if args.type == "feature":
        print("Running feature ablation...")
        results = runner.run_feature_ablation(args.model)
    elif args.type == "architecture":
        print("Running architecture ablation...")
        results = runner.run_architecture_ablation(args.model)
    elif args.type == "data":
        print("Running data size ablation...")
        results = runner.run_data_ablation(args.model)
    elif args.type == "hyperparam":
        print("Running hyperparameter sensitivity...")
        results = runner.run_hyperparameter_sensitivity(args.model)

    if not results.empty:
        print("\nResults:")
        print(results.to_string(index=False))

        os.makedirs("experiments", exist_ok=True)
        fig = plot_ablation_results(results, args.type)
        output_path = f"experiments/ablation_{args.type}.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved to {output_path}")
        plt.close(fig)

        results.to_csv(f"experiments/ablation_{args.type}.csv", index=False)
        print(f"Results saved to experiments/ablation_{args.type}.csv")
