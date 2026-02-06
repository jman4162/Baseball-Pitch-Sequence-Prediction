# Ablation Studies

Ablation studies measure the contribution of individual components by systematically removing or varying them. The `AblationRunner` supports four types of ablation.

## CLI Usage

```bash
# Feature ablation
pitch-ablation --type feature --model lstm

# Architecture ablation
pitch-ablation --type architecture --model lstm

# Data scaling ablation
pitch-ablation --type data --model lstm

# Hyperparameter ablation
pitch-ablation --type hyperparam --model lstm
```

## Ablation Types

### Feature Ablation

Measures the impact of each input feature by training the model with one feature removed at a time.

```bash
pitch-ablation --type feature --model lstm
```

Output: Accuracy drop when each feature is removed, showing which features contribute most.

### Architecture Ablation

Varies architectural parameters (e.g., number of layers, hidden size) to understand their impact.

```bash
pitch-ablation --type architecture --model lstm
```

### Data Scaling Ablation

Trains the model on increasing fractions of the dataset to measure how performance scales with data size.

```bash
pitch-ablation --type data --model lstm
```

### Hyperparameter Ablation

Sweeps key hyperparameters to understand sensitivity.

```bash
pitch-ablation --type hyperparam --model lstm
```

## Configuration

```yaml
# configs/ablation.yaml
feature_ablation:
  model: lstm
  features_to_drop:
    - Balls
    - Strikes
    - PitcherType_enc
    - PreviousPitchType_enc

data_ablation:
  model: lstm
  fractions: [0.1, 0.25, 0.5, 0.75, 1.0]

architecture_ablation:
  model: lstm
  variants:
    - hidden_size: 32
    - hidden_size: 64
    - hidden_size: 128

hyperparam_ablation:
  model: lstm
  params:
    learning_rate: [0.0001, 0.001, 0.01]
    dropout: [0.1, 0.3, 0.5]
```

## Python API

```python
from pitch_sequencing.config import DataConfig, AblationConfig
from pitch_sequencing.evaluation.ablation import AblationRunner

data_cfg = DataConfig.from_yaml("configs/data.yaml")
ablation_cfg = AblationConfig(...)  # Load from YAML or construct

runner = AblationRunner(ablation_cfg, data_cfg, models_config_dir="configs/models")
# Run specific ablation type
```

## MLflow Logging

All ablation runs are logged to MLflow with parameters identifying the ablation type and variant. Use the MLflow UI to compare results across ablation conditions.
