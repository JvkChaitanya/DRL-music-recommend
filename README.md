# Sequential Music Recommendation with RL

A two-phase music recommendation system combining content-aware sequential modeling (SASRec) with reinforcement learning for long-term user engagement optimization.

## Architecture

```
Phase 1: SASRec (Supervised) → Learns sequential patterns + content features
Phase 2: RL Policy (Actor-Critic) → Optimizes for engagement & diversity
```

## Project Structure

```
sequential-music-rec/
├── data/
│   ├── raw/                    # Raw Last.fm-1K data
│   └── processed/              # Preprocessed sequences
├── src/
│   ├── data/
│   │   ├── preprocess.py       # Data preprocessing
│   │   └── dataset.py          # PyTorch Dataset
│   ├── models/
│   │   ├── sasrec.py           # SASRec model
│   │   ├── rl_agent.py         # Actor-Critic agent
│   │   └── environment.py      # Simulated environment
│   ├── train_sasrec.py         # Phase 1 training
│   ├── train_rl.py             # Phase 2 training
│   └── evaluate.py             # Evaluation metrics
├── config.yaml                 # Hyperparameters
└── requirements.txt
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Preprocess data
python src/data/preprocess.py

# Train SASRec (Phase 1)
python src/train_sasrec.py

# Train RL (Phase 2)
python src/train_rl.py
```

## Dataset

Using Last.fm-1K dataset:
- 19.1M listening events
- 992 users
- 1M+ unique tracks
- Timestamped for sequential modeling
