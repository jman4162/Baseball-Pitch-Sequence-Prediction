# Data Pipeline

The data pipeline generates synthetic baseball pitch data, loads it, preprocesses features, and creates sequences for model training.

## Data Generation

The simulator (`pitch_sequencing.data.simulator`) generates realistic pitch-by-pitch data.

### Pitcher Archetypes

Each simulated pitcher is assigned one of four archetypes:

| Archetype | Fastball % | Slider % | Curveball % | Changeup % | Fatigue Threshold |
|-----------|-----------|----------|-------------|------------|-------------------|
| Power | 55% | 20% | 10% | 15% | 95 pitches |
| Finesse | 25% | 15% | 30% | 30% | 80 pitches |
| Slider Specialist | 20% | 40% | 20% | 20% | 85 pitches |
| Balanced | 30% | 25% | 25% | 20% | 90 pitches |

Archetype blending uses 60% archetype bias and 40% count-based probabilities.

### Sequence Strategies

Eight pitch patterns create learnable sequential dependencies by boosting follow-up pitch probability by 15-25%:

- Fastball → Fastball → Changeup
- Slider → Slider → Fastball
- Curveball → Fastball (and more)

### Count-Dependent Outcomes

Hit rates vary from 5-6% in pitcher's counts (0-2, 1-2) to 19-23% in hitter's counts (3-0, 3-1).

### Fatigue Modeling

After an archetype-specific threshold (80-95 pitches), pitchers shift toward fastballs and more balls.

### Game Situation

Runners on base and score differential affect pitch selection probabilities.

### CLI Usage

```bash
pitch-generate --num-games 3000 --at-bats 35 --seed 42 --output-dir ./data
```

### Python Usage

```python
from pitch_sequencing.data.simulator import generate_dataset, generate_hmm_sequences

# Main dataset (~384K rows)
df = generate_dataset(num_games=3000, at_bats_per_game=35, seed=42)

# HMM sequences (2500 x 100)
hmm_df = generate_hmm_sequences(num_sequences=2500, sequence_length=100, seed=42)
```

## Data Loading

```python
from pitch_sequencing.data.loader import load_pitch_data, create_sequences

df = load_pitch_data("data/baseball_pitch_data.csv", filter_none_prev=True)
```

### Dataset Columns

| Column | Type | Description |
|--------|------|-------------|
| Balls | int | Current ball count (0-3) |
| Strikes | int | Current strike count (0-2) |
| PitchType | str | Fastball, Slider, Curveball, Changeup |
| Outcome | str | ball, strike, hit |
| PitcherType | str | power, finesse, slider_specialist, balanced |
| PitchNumber | int | Cumulative per-game pitch count |
| AtBatNumber | int | At-bat number within game (1-35) |
| RunnersOn | int | Number of runners on base (0-3) |
| ScoreDiff | int | Score differential |
| PreviousPitchType | str | Previous pitch thrown |

!!! note
    `PitchNumber` is the same value for all pitches within an at-bat. It is a cumulative per-game pitch count, not sequential per-pitch.

## Preprocessing

### Encoding Categoricals

```python
from pitch_sequencing.data.preprocessing import encode_categoricals

df, encoders = encode_categoricals(df, ["PitchType", "Outcome", "PitcherType", "PreviousPitchType"])
# Creates PitchType_enc, Outcome_enc, etc.
```

### Normalizing Numericals

```python
from pitch_sequencing.data.preprocessing import normalize_numericals

df, stats = normalize_numericals(df, ["PitchNumber", "AtBatNumber", "RunnersOn", "ScoreDiff"])
# Saves PitchNumber_raw, AtBatNumber_raw for boundary detection
```

### Creating Sequences

For sequence models (LSTM, CNN1D, Transformer):

```python
from pitch_sequencing.data.loader import create_sequences

X_seq, y_seq, game_starts = create_sequences(
    df, window_size=8,
    feature_cols=["Balls", "Strikes", "PitchType_enc", ...],
    target_col="PitchType_enc"
)
# X_seq shape: (n_samples, 8, n_features)
# y_seq shape: (n_samples,)
```

Game boundaries are detected using AtBatNumber resets (drops from ~35 back to 1).

### Creating Train/Test Splits

```python
from pitch_sequencing.data.preprocessing import create_splits

folds = create_splits(X, y, n_folds=5, stratify=True, random_state=42)
for train_idx, test_idx in folds:
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
```
