"""
Data preprocessing for Last.fm-1K dataset.
Converts raw listening events into training sequences.
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import timedelta
from collections import defaultdict
from tqdm import tqdm
import yaml


def load_config():
    """Load configuration from yaml file."""
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load raw Last.fm-1K listening events."""
    print("Loading raw data...")
    df = pd.read_csv(
        filepath,
        sep='\t',
        names=['user_id', 'timestamp', 'artist_id', 'artist_name', 'track_id', 'track_name'],
        on_bad_lines='skip'
    )
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp', 'track_name'])
    print(f"Loaded {len(df):,} events")
    return df


def build_vocabulary(df: pd.DataFrame, min_count: int = 5) -> dict:
    """Build track vocabulary mapping track names to indices."""
    print("Building vocabulary...")
    track_counts = df['track_name'].value_counts()
    valid_tracks = track_counts[track_counts >= min_count].index.tolist()
    
    # Special tokens
    vocab = {
        '<PAD>': 0,
        '<UNK>': 1,
    }
    
    for i, track in enumerate(valid_tracks, start=2):
        vocab[track] = i
    
    print(f"Vocabulary size: {len(vocab):,} tracks")
    return vocab


def create_sequences(
    df: pd.DataFrame,
    vocab: dict,
    min_seq_length: int = 5,
    max_seq_length: int = 50,
    session_gap_minutes: int = 30
) -> list:
    """
    Create training sequences from listening history.
    Splits by session (time gap > threshold) and creates sliding windows.
    """
    print("Creating sequences...")
    sequences = []
    session_gap = timedelta(minutes=session_gap_minutes)
    
    # Group by user and sort by timestamp
    grouped = df.groupby('user_id')
    
    for user_id, user_df in tqdm(grouped, desc="Processing users"):
        user_df = user_df.sort_values('timestamp')
        
        # Split into sessions
        sessions = []
        current_session = []
        prev_time = None
        
        for _, row in user_df.iterrows():
            track = row['track_name']
            timestamp = row['timestamp']
            
            # Map to vocabulary
            track_id = vocab.get(track, vocab['<UNK>'])
            if track_id == vocab['<UNK>']:
                continue
            
            if prev_time is not None and (timestamp - prev_time) > session_gap:
                # New session
                if len(current_session) >= min_seq_length:
                    sessions.append(current_session)
                current_session = []
            
            current_session.append(track_id)
            prev_time = timestamp
        
        # Don't forget last session
        if len(current_session) >= min_seq_length:
            sessions.append(current_session)
        
        # Create sliding window sequences from sessions
        for session in sessions:
            for i in range(min_seq_length, len(session) + 1):
                seq = session[max(0, i - max_seq_length):i]
                if len(seq) >= min_seq_length:
                    sequences.append({
                        'user_id': user_id,
                        'sequence': seq[:-1],  # Input sequence
                        'target': seq[-1]       # Next item to predict
                    })
    
    print(f"Created {len(sequences):,} sequences")
    return sequences


def split_data(sequences: list, train_ratio: float = 0.8, val_ratio: float = 0.1):
    """Split sequences into train/val/test sets."""
    np.random.shuffle(sequences)
    n = len(sequences)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train = sequences[:train_end]
    val = sequences[train_end:val_end]
    test = sequences[val_end:]
    
    print(f"Train: {len(train):,}, Val: {len(val):,}, Test: {len(test):,}")
    return train, val, test


def save_processed_data(vocab, train, val, test, output_dir: str):
    """Save processed data to disk."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / 'vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    
    with open(output_path / 'train.pkl', 'wb') as f:
        pickle.dump(train, f)
    
    with open(output_path / 'val.pkl', 'wb') as f:
        pickle.dump(val, f)
    
    with open(output_path / 'test.pkl', 'wb') as f:
        pickle.dump(test, f)
    
    print(f"Saved processed data to {output_path}")


def main():
    config = load_config()
    
    # Load raw data
    df = load_raw_data(config['data']['raw_path'])
    
    # Build vocabulary
    vocab = build_vocabulary(df)
    
    # Create sequences
    sequences = create_sequences(
        df,
        vocab,
        min_seq_length=config['data']['min_seq_length'],
        max_seq_length=config['data']['max_seq_length'],
        session_gap_minutes=config['data']['session_gap_minutes']
    )
    
    # Split data
    train, val, test = split_data(sequences)
    
    # Save
    save_processed_data(vocab, train, val, test, config['data']['processed_path'])
    
    print("\nPreprocessing complete!")


if __name__ == "__main__":
    main()
