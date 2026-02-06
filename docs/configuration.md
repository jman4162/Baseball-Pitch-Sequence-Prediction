# Configuration

All settings are driven by YAML config files. Bundled configs ship with the package in `src/pitch_sequencing/configs/` and are mirrored in `configs/` for development.

## Data Configuration

```yaml
# configs/data.yaml
data_path: data/baseball_pitch_data.csv
hmm_data_path: data/synthetic_pitch_sequences.csv
target_col: PitchType
outcome_col: Outcome

test_size: 0.2
n_folds: 5
random_state: 42
window_size: 8

tabular_features:
  - Balls
  - Strikes
  - PitcherType
  - PreviousPitchType
  - PitchNumber
  - AtBatNumber
  - RunnersOn
  - ScoreDiff

sequence_features:
  - Balls
  - Strikes
  - PitchType_enc
  - Outcome_enc
  - PitcherType_enc
  - PitchNumber
  - AtBatNumber
  - RunnersOn
  - ScoreDiff
  - PreviousPitchType_enc

categorical_features:
  - PitchType
  - Outcome
  - PitcherType
  - PreviousPitchType

numerical_features:
  - PitchNumber
  - AtBatNumber
  - RunnersOn
  - ScoreDiff
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

## Model Configurations

### Logistic Regression

```yaml
# configs/models/logistic.yaml
model_type: logistic_regression
C: 1.0
penalty: l2
class_weight: balanced
max_iter: 1000
```

### Random Forest

```yaml
# configs/models/random_forest.yaml
model_type: random_forest
n_estimators: 200
max_depth: 15
random_state: 42
```

### HMM

```yaml
# configs/models/hmm.yaml
model_type: hmm
min_components: 1
max_components: 8
n_iter: 100
```

### AutoGluon

```yaml
# configs/models/autogluon.yaml
model_type: autogluon
preset: good_quality
time_limit: null
models_dir: autogluon_pitchtype_models
```

### LSTM

```yaml
# configs/models/lstm.yaml
model_type: lstm
hidden_size: 64
num_layers: 2
dropout: 0.3
epochs: 20
learning_rate: 0.001
batch_size: 256
```

### 1D-CNN

```yaml
# configs/models/cnn1d.yaml
model_type: cnn1d
filters: [64, 128, 64]
kernel_size: 3
dropout: 0.3
epochs: 20
learning_rate: 0.001
batch_size: 256
```

### Transformer

```yaml
# configs/models/transformer.yaml
model_type: transformer
d_model: 64
nhead: 4
num_layers: 2
dropout: 0.3
epochs: 20
learning_rate: 0.001
batch_size: 256
```

## Ablation Configuration

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

## Path Resolution

Configs are resolved using `pitch_sequencing.paths`:

```python
from pitch_sequencing.paths import get_default_config, get_default_data_dir

# Get bundled config path
config_path = get_default_config("data.yaml")
model_config = get_default_config("models/lstm.yaml")

# Get default data directory
data_dir = get_default_data_dir()
```

The path resolution prefers local `./configs/` and `./data/` directories (for development) and falls back to bundled package resources (for installed usage).
