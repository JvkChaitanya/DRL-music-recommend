# Sequential Music Recommendation with Deep Reinforcement Learning

A music recommendation system combining collaborative filtering baselines with deep reinforcement learning for sequential, personalized recommendations.

## Models Implemented

### Baseline Models (4)
| Model | File | Description |
|-------|------|-------------|
| **Popularity** | `src/models/popularity.py` | Recommends most popular items regardless of user |
| **Item-CF** | `src/models/item_cf.py` | Item-based Collaborative Filtering using cosine similarity |
| **User-CF** | `src/models/user_cf.py` | User-based Collaborative Filtering |
| **SASRec** | `src/models/sasrec.py` | Self-Attentive Sequential Recommendation (Transformer-based) |

### RL Model
| Model | File | Description |
|-------|------|-------------|
| **Item-CF + RL** | `src/models/rl_agent.py` | Actor-Critic agent using Item-CF as backbone for candidate generation |

Based on performance comparison, **Item-CF** was selected as the backbone for the RL agent due to its strong baseline performance.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Training Pipeline                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Phase 1: Baseline Training                                     │
│   ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │
│   │  Popularity  │ │   Item-CF    │ │   User-CF    │            │
│   └──────────────┘ └──────┬───────┘ └──────────────┘            │
│                           │                                      │
│   Phase 2: RL Training    │                                      │
│   ┌───────────────────────▼─────────────────────────┐           │
│   │         Actor-Critic RL Agent                    │           │
│   │   ┌─────────────┐     ┌─────────────┐           │           │
│   │   │    Actor    │ ←── │   Critic    │           │           │
│   │   │ (Policy π)  │     │  (Value V)  │           │           │
│   │   └──────┬──────┘     └─────────────┘           │           │
│   │          │                                       │           │
│   │   Item-CF Candidates → RL Re-ranking → Action   │           │
│   └─────────────────────────────────────────────────┘           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
sequential-music-rec/
├── run.py                      # Main executable (CLI)
├── config.yaml                 # Configuration
├── requirements.txt            # Dependencies
├── README.md
│
├── src/
│   ├── data/
│   │   ├── preprocess.py       # Data preprocessing
│   │   └── dataset.py          # PyTorch Dataset
│   │
│   ├── models/
│   │   ├── popularity.py       # Popularity baseline
│   │   ├── item_cf.py          # Item-based CF (RL backbone)
│   │   ├── user_cf.py          # User-based CF
│   │   ├── sasrec.py           # SASRec model
│   │   ├── rl_agent.py         # Actor-Critic RL agent
│   │   └── environment.py      # RL environment
│   │
│   ├── train_baselines.py      # Train all baselines
│   ├── train_sasrec.py         # Train SASRec
│   ├── train_rl_itemcf.py      # Train RL agent
│   └── evaluate_all_models.py  # Evaluate all models
│
├── data/
│   ├── raw/                    # Raw Last.fm-1K data
│   └── processed/              # Preprocessed sequences
│
└── checkpoints/                # Trained model weights
```

## Installation

```bash
# Clone the repository
git clone https://github.com/YourUsername/sequential-music-rec.git
cd sequential-music-rec

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Requirements
- Python >= 3.8
- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- pandas >= 2.0.0
- scikit-learn >= 1.3.0
- tqdm >= 4.65.0
- PyYAML >= 6.0

## Quick Start

```bash
# 1. Preprocess data
python run.py preprocess

# 2. Train all 4 baseline models (Popularity, Item-CF, User-CF, SASRec)
python run.py train-baselines

# 3. Train RL agent (uses Item-CF as backbone)
python run.py train-rl

# 4. Evaluate all models
python run.py evaluate
```

## Usage

### Command Line Interface

```bash
python run.py <command>
```

| Command | Description |
|---------|-------------|
| `preprocess` | Preprocess raw Last.fm data into sequences |
| `train-baselines` | Train all 4 baselines (Popularity, Item-CF, User-CF, SASRec) |
| `train-rl` | Train RL agent with Item-CF backbone |
| `evaluate` | Evaluate and compare all models |


### Configuration

Edit `config.yaml` to adjust hyperparameters:

```yaml
data:
  raw_path: data/raw
  processed_path: data/processed

sasrec:
  embedding_dim: 64
  num_heads: 2
  num_layers: 2

rl:
  actor_lr: 0.0001
  critic_lr: 0.001
  gamma: 0.99
  episodes: 10000
```

## Dataset

**Last.fm-1K** dataset:
- 19.1M listening events
- 992 users
- 1M+ unique tracks
- Timestamped for sequential modeling

## Evaluation Metrics

- **Hit@K**: Whether the target item is in top-K recommendations
- **NDCG@K**: Normalized Discounted Cumulative Gain at K
- **MRR**: Mean Reciprocal Rank

## Citations

If you use this code, please cite:

### SASRec
```bibtex
@inproceedings{kang2018self,
  title={Self-Attentive Sequential Recommendation},
  author={Kang, Wang-Cheng and McAuley, Julian},
  booktitle={ICDM},
  year={2018}
}
```

### Last.fm Dataset
```bibtex
@inproceedings{celma2010music,
  title={Music Recommendation and Discovery: The Long Tail, Long Fail, and Long Play in the Digital Music Space},
  author={Celma, Oscar},
  year={2010},
  publisher={Springer}
}
```

### Actor-Critic RL
```bibtex
@article{mnih2016asynchronous,
  title={Asynchronous Methods for Deep Reinforcement Learning},
  author={Mnih, Volodymyr and others},
  journal={ICML},
  year={2016}
}
```

## License

MIT License
