# MLflow Tracking

All experiments are tracked with MLflow for reproducibility and comparison.

## Starting the UI

```bash
make mlflow
# or: mlflow ui --backend-store-uri experiments
```

This opens the MLflow UI at [http://localhost:5000](http://localhost:5000).

## Tracking URI

Experiments are stored locally in the `experiments/` directory (gitignored):

```
file://./experiments/
```

## What Gets Logged

### Benchmark Runs

Each model's k-fold CV run logs:

- **Parameters**: model name, hyperparameters, n_folds, data config
- **Metrics**: accuracy, balanced_accuracy, macro_f1, log_loss (per fold and averaged)
- **Artifacts**: results DataFrame, confusion matrices

### Ablation Runs

Each ablation variant logs:

- **Parameters**: ablation type, model, variant description
- **Metrics**: performance under each ablation condition
- **Tags**: ablation type for easy filtering

## Experiment Names

| Run Type | Default Experiment Name |
|----------|------------------------|
| Benchmark | `pitch_benchmark` |
| Ablation | `pitch_ablation` |
| Single training | `pitch_train` |

## Comparing Runs

In the MLflow UI:

1. Select an experiment from the sidebar
2. Check the runs you want to compare
3. Click **Compare** to see side-by-side metrics and parameters
4. Use the **Chart** view to visualize metric distributions

## Programmatic Access

```python
import mlflow

mlflow.set_tracking_uri("file://./experiments")

# List experiments
for exp in mlflow.search_experiments():
    print(exp.name, exp.experiment_id)

# Query runs
runs = mlflow.search_runs(experiment_names=["pitch_benchmark"])
print(runs[["params.model", "metrics.accuracy", "metrics.macro_f1"]])
```

## Cleaning Up

Experiment artifacts are stored in `experiments/` and are gitignored. To clean up:

```bash
make clean  # removes experiments/ and other build artifacts
```
