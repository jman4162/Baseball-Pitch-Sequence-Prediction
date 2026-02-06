# Installation

## Prerequisites

- Python 3.9 or later
- pip (included with Python)
- Git (for cloning the repository)

## Install from Source

```bash
git clone https://github.com/jman4162/Baseball-Pitch-Sequence-Prediction.git
cd Baseball-Pitch-Sequence-Prediction
```

### Development Install (Recommended)

Includes all optional dependencies and development tools:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[all,dev]"
```

Or using Make:

```bash
make install
```

### Minimal Install

Core dependencies only (no AutoGluon or hmmlearn):

```bash
pip install -e .
```

### Optional Extras

| Extra | Contents |
|-------|----------|
| `all` | AutoGluon + hmmlearn |
| `autogluon` | AutoGluon TabularPredictor |
| `hmm` | hmmlearn for HMM model |
| `docs` | MkDocs + Material theme + mkdocstrings |
| `dev` | pytest + build + docs |

Install specific extras:

```bash
pip install -e ".[hmm]"         # Just HMM support
pip install -e ".[autogluon]"   # Just AutoGluon
pip install -e ".[docs]"        # Documentation tools
```

## Verify Installation

```bash
# Check the package is installed
python -c "import pitch_sequencing; print(pitch_sequencing.__version__)"

# Check CLI commands are available
pitch-generate --help
pitch-train --help
pitch-benchmark --help
pitch-ablation --help
```

## Generate Training Data

After installation, generate the synthetic dataset:

```bash
pitch-generate --output-dir ./data
```

This creates two files in `data/`:

- `baseball_pitch_data.csv` (~384K rows)
- `synthetic_pitch_sequences.csv` (2,500 sequences)
