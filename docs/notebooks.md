# Notebooks

Original exploratory Jupyter notebooks are preserved in `notebooks/` and can be run via Jupyter or Google Colab. They now import from the `pitch_sequencing` package.

## Available Notebooks

### 1. Baseball Pitch Sequence Simulator

**File:** `notebooks/Baseball_Pitch_Sequence_Simulator.ipynb`

Demonstrates the synthetic data generation pipeline, including pitcher archetypes, sequence strategies, count-dependent outcomes, fatigue modeling, and game context. Generates both the main pitch dataset and HMM training sequences.

### 2. HMM Pitch Predictor

**File:** `notebooks/HMM_Pitch_Predictor.ipynb`

Trains a Hidden Markov Model on synthetic pitch sequences. Sweeps the number of hidden states (1-8) and evaluates prediction accuracy. Shows transition matrices and emission probabilities.

### 3. AutoGluon Baseball Pitch Prediction

**File:** `notebooks/AutoGluon_Baseball_Pitch_Prediction.ipynb`

Uses AutoGluon's TabularPredictor to predict the next pitch type from tabular features. Demonstrates automated model selection and ensembling.

### 4. AutoGluon Baseball Pitch Outcome Prediction

**File:** `notebooks/AutoGluon_Baseball_Pitch_Outcome_Prediction.ipynb`

Predicts pitch outcomes (ball, strike, hit) rather than pitch types. Uses the same AutoGluon approach with outcome-specific features.

### 5. LSTM Pitch Predictor

**File:** `notebooks/LSTM_Pitch_Predictor.ipynb`

Trains a 2-layer LSTM on windowed pitch sequences. Includes data preprocessing, sequence creation, model training with early stopping, and evaluation with confusion matrices.

## Running Notebooks

### Local Jupyter

```bash
source venv/bin/activate
pip install jupyter
jupyter notebook notebooks/
```

### Google Colab

Upload any notebook to Google Colab and add this cell at the top:

```python
!pip install git+https://github.com/jman4162/Baseball-Pitch-Sequence-Prediction.git
```
