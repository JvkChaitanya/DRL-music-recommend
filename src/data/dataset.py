"""
PyTorch Dataset for sequential music recommendation.
"""
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from pathlib import Path
import numpy as np


class MusicSequenceDataset(Dataset):
    """Dataset for music sequence prediction."""
    
    def __init__(self, data_path: str, max_seq_length: int = 50):
        self.max_seq_length = max_seq_length
        
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        sequence = sample['sequence']
        target = sample['target']
        
        # Pad or truncate sequence
        if len(sequence) >= self.max_seq_length:
            sequence = sequence[-self.max_seq_length:]
        else:
            # Pad at the beginning
            padding = [0] * (self.max_seq_length - len(sequence))
            sequence = padding + sequence
        
        return {
            'sequence': torch.tensor(sequence, dtype=torch.long),
            'target': torch.tensor(target, dtype=torch.long),
            'seq_length': torch.tensor(len(sample['sequence']), dtype=torch.long)
        }


def create_dataloaders(
    processed_dir: str,
    batch_size: int = 128,
    max_seq_length: int = 50,
    num_workers: int = 0
) -> tuple:
    """Create train, validation, and test dataloaders."""
    processed_path = Path(processed_dir)
    
    train_dataset = MusicSequenceDataset(
        processed_path / 'train.pkl',
        max_seq_length
    )
    val_dataset = MusicSequenceDataset(
        processed_path / 'val.pkl',
        max_seq_length
    )
    test_dataset = MusicSequenceDataset(
        processed_path / 'test.pkl',
        max_seq_length
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader


def load_vocab(processed_dir: str) -> dict:
    """Load vocabulary mapping."""
    with open(Path(processed_dir) / 'vocab.pkl', 'rb') as f:
        return pickle.load(f)
