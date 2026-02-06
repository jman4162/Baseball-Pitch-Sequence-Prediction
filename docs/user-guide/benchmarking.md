# Benchmarking

The benchmark suite runs all models through k-fold cross-validation and computes metrics with bootstrap confidence intervals and statistical tests.

## CLI Usage

```bash
# Run with default config
pitch-benchmark

# Run with custom config
pitch-benchmark --config configs/benchmark.yaml
```

## Benchmark Configuration

```yaml
# configs/benchmark.yaml
experiment_name: pitch_benchmark
models:
  - logistic_regression
  - random_forest
  - hmm
  - autogluon
  - lstm
  - cnn1d
  - transformer
n_folds: 5
metrics:
  - accuracy
  - balanced_accuracy
  - macro_f1
  - log_loss
```

## Python API

```python
from pitch_sequencing.config import DataConfig, BenchmarkConfig
from pitch_sequencing.evaluation.benchmark import BenchmarkRunner

data_cfg = DataConfig.from_yaml("configs/data.yaml")
bench_cfg = BenchmarkConfig(
    experiment_name="my_benchmark",
    models=["lstm", "random_forest", "transformer"],
    n_folds=5,
    metrics=["accuracy", "macro_f1"]
)

runner = BenchmarkRunner(bench_cfg, data_cfg, models_config_dir="configs/models")
results_df = runner.run()
print(results_df)
```

## Output

The benchmark produces:

- **Per-fold metrics** for each model
- **Bootstrap confidence intervals** (95% CI by default, 1000 bootstrap samples)
- **Paired t-tests** between model pairs with p-values
- **Cohen's d effect sizes** for pairwise comparisons
- **MLflow experiment logs** with parameters, metrics, and artifacts

## Metrics

| Metric | Description |
|--------|-------------|
| `accuracy` | Overall accuracy |
| `balanced_accuracy` | Average per-class recall |
| `macro_precision` | Macro-averaged precision |
| `macro_recall` | Macro-averaged recall |
| `macro_f1` | Macro-averaged F1 score |
| `log_loss` | Logarithmic loss (requires `predict_proba`) |

Per-class precision, recall, and F1 are also computed for each pitch type (Fastball, Slider, Curveball, Changeup).

## Statistical Comparisons

After k-fold CV, models are compared pairwise:

- **Paired t-test**: Tests whether the difference in fold scores is statistically significant
- **Cohen's d**: Measures the effect size (small: 0.2, medium: 0.5, large: 0.8)

## MLflow Integration

All benchmark runs are logged to MLflow under the experiment name. See [MLflow Tracking](mlflow.md) for details on viewing and comparing runs.
