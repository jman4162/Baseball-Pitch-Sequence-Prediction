# CLI Reference

After installing the package (`pip install -e ".[all,dev]"`), the following commands are available on your PATH.

## pitch-generate

Generate synthetic baseball pitch datasets.

```bash
pitch-generate [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--num-games` | 3000 | Number of games to simulate |
| `--at-bats` | 35 | At-bats per game |
| `--seed` | 42 | Random seed |
| `--output-dir` | `data` | Output directory for CSV files |

**Example:**

```bash
# Default: 3000 games, ~384K pitch rows
pitch-generate

# Custom: smaller dataset
pitch-generate --num-games 500 --at-bats 30 --output-dir ./my-data
```

**Output files:**

- `<output-dir>/baseball_pitch_data.csv` — Main pitch dataset
- `<output-dir>/synthetic_pitch_sequences.csv` — HMM training sequences

---

## pitch-train

Train a single model.

```bash
pitch-train --model MODEL [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | *(required)* | Model name from registry |
| `--config` | *(auto-detected)* | Path to model config YAML |

**Available models:** `logistic_regression`, `random_forest`, `hmm`, `autogluon`, `lstm`, `cnn1d`, `transformer`

**Example:**

```bash
# Train with default config
pitch-train --model lstm

# Train with custom config
pitch-train --model lstm --config configs/models/lstm.yaml
```

---

## pitch-benchmark

Run the full benchmark suite across all models.

```bash
pitch-benchmark [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | *(auto-detected)* | Path to benchmark config YAML |

**Example:**

```bash
# Run all 7 models through 5-fold CV
pitch-benchmark

# Custom benchmark config
pitch-benchmark --config my_benchmark.yaml
```

---

## pitch-ablation

Run ablation studies on a specific model.

```bash
pitch-ablation --type TYPE [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--type` | *(required)* | Ablation type |
| `--model` | *(from config)* | Model to ablate |

**Ablation types:** `feature`, `architecture`, `data`, `hyperparam`

**Example:**

```bash
# Feature importance ablation
pitch-ablation --type feature --model lstm

# Data scaling ablation
pitch-ablation --type data --model transformer
```

---

## Make Targets

The Makefile provides shortcuts for common commands:

| Target | Command |
|--------|---------|
| `make install` | `pip install -e ".[all,dev]"` |
| `make data` | `pitch-generate` |
| `make train MODEL=lstm` | `pitch-train --model lstm` |
| `make benchmark` | `pitch-benchmark` |
| `make ablation TYPE=feature` | `pitch-ablation --type feature` |
| `make mlflow` | `mlflow ui --backend-store-uri experiments` |
| `make test` | `pytest tests/` |
| `make docs` | `mkdocs build --strict` |
| `make docs-serve` | `mkdocs serve` |
| `make clean` | Remove build artifacts |
