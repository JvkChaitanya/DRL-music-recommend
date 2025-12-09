"""
Training script for all baseline models.
Trains: Popularity, Item-CF, User-CF, and SASRec.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import yaml
import pickle
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent))

from data.dataset import load_vocab, create_dataloaders
from models.popularity import PopularityBaseline
from models.item_cf import ItemBasedCF
from models.user_cf import UserBasedCF
from models.sasrec import SASRec


def save_baselines(popularity, item_cf, user_cf, output_dir: str):
    """Save trained baseline models."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / 'popularity_baseline.pkl', 'wb') as f:
        pickle.dump(popularity, f)
    
    with open(output_path / 'item_cf.pkl', 'wb') as f:
        pickle.dump(item_cf, f)
    
    with open(output_path / 'user_cf.pkl', 'wb') as f:
        pickle.dump(user_cf, f)
    
    print(f"Saved baselines to {output_path}")


def load_config():
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_data(processed_dir: str):
    """Load preprocessed train/val/test data."""
    processed_path = Path(processed_dir)
    
    with open(processed_path / 'train.pkl', 'rb') as f:
        train = pickle.load(f)
    with open(processed_path / 'val.pkl', 'rb') as f:
        val = pickle.load(f)
    with open(processed_path / 'test.pkl', 'rb') as f:
        test = pickle.load(f)
    
    return train, val, test


def evaluate_baseline(model, test_data: list, num_items: int, top_k_values: list, quiet: bool = True):
    """Evaluate a baseline model (Popularity, Item-CF, User-CF)."""
    hits = {k: 0 for k in top_k_values}
    ndcg = {k: 0.0 for k in top_k_values}
    mrr_sum = 0.0
    n_samples = len(test_data)
    
    for sample in tqdm(test_data, desc="Evaluating", disable=quiet, ncols=80):
        sequence = sample['sequence']
        target = sample['target']
        
        scores = model.predict_scores(sequence, num_items)
        sorted_items = np.argsort(scores)[::-1]
        
        rank = np.where(sorted_items == target)[0]
        if len(rank) > 0:
            rank = rank[0] + 1
            mrr_sum += 1.0 / rank
            
            for k in top_k_values:
                if rank <= k:
                    hits[k] += 1
                    ndcg[k] += 1.0 / np.log2(rank + 1)
    
    metrics = {}
    for k in top_k_values:
        metrics[f'hit@{k}'] = hits[k] / n_samples
        metrics[f'ndcg@{k}'] = ndcg[k] / n_samples
    metrics['mrr'] = mrr_sum / n_samples
    
    return metrics


# ==================== SASRec Training Functions ====================

def train_sasrec_epoch(model, dataloader, optimizer, criterion, device, quiet=True, grad_clip=1.0):
    """Train SASRec for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="SASRec Training", disable=quiet, ncols=80)
    for batch in pbar:
        sequence = batch['sequence'].to(device)
        target = batch['target'].to(device)
        
        optimizer.zero_grad()
        logits = model(sequence)
        loss = criterion(logits, target)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({'loss': total_loss / num_batches})
    
    return total_loss / num_batches


def validate_sasrec(model, dataloader, criterion, device):
    """Validate SASRec model."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            sequence = batch['sequence'].to(device)
            target = batch['target'].to(device)
            
            logits = model(sequence)
            loss = criterion(logits, target)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def evaluate_sasrec(model, test_data, num_items, top_k_values, max_seq_length, device, quiet=True):
    """Evaluate SASRec model."""
    model.eval()
    hits = {k: 0 for k in top_k_values}
    ndcg = {k: 0.0 for k in top_k_values}
    mrr_sum = 0.0
    n_samples = len(test_data)
    
    for sample in tqdm(test_data, desc="SASRec Eval", disable=quiet, ncols=80):
        sequence = sample['sequence']
        target = sample['target']
        
        # Pad/truncate sequence
        seq = sequence[-max_seq_length:]
        if len(seq) < max_seq_length:
            seq = [0] * (max_seq_length - len(seq)) + seq
        
        seq_tensor = torch.tensor([seq], dtype=torch.long, device=device)
        
        with torch.no_grad():
            logits = model(seq_tensor).squeeze(0)
        
        # Mask sequence items
        for item in sequence:
            if item < len(logits):
                logits[item] = -float('inf')
        
        sorted_items = torch.argsort(logits, descending=True).cpu().numpy()
        
        rank_arr = np.where(sorted_items == target)[0]
        if len(rank_arr) > 0:
            rank = rank_arr[0] + 1
            mrr_sum += 1.0 / rank
            
            for k in top_k_values:
                if rank <= k:
                    hits[k] += 1
                    ndcg[k] += 1.0 / np.log2(rank + 1)
    
    metrics = {}
    for k in top_k_values:
        metrics[f'hit@{k}'] = hits[k] / n_samples
        metrics[f'ndcg@{k}'] = ndcg[k] / n_samples
    metrics['mrr'] = mrr_sum / n_samples
    
    return metrics


def train_sasrec(config, num_items, train_loader, val_loader, test_data, checkpoint_dir, device, quiet=True):
    """Train SASRec model."""
    print("\n" + "="*50)
    print("Training SASRec...")
    print("="*50)
    
    model = SASRec(
        num_items=num_items,
        embedding_dim=config['sasrec']['embedding_dim'],
        num_heads=config['sasrec']['num_heads'],
        num_layers=config['sasrec']['num_layers'],
        max_seq_length=config['sasrec']['max_seq_length'],
        dropout=config['sasrec']['dropout']
    ).to(device)
    
    print(f"SASRec parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    grad_clip = config.get('advanced', {}).get('gradient_clip', 1.0)
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config['training']['sasrec_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['training']['sasrec_epochs']}")
        
        train_loss = train_sasrec_epoch(model, train_loader, optimizer, criterion, device, quiet, grad_clip)
        val_loss = validate_sasrec(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_dir / "sasrec_best.pt")
            print("Saved best SASRec model!")
        else:
            patience_counter += 1
            if patience_counter >= config['training']['early_stopping_patience']:
                print("Early stopping triggered!")
                break
    
    # Evaluate on test set
    model.load_state_dict(torch.load(checkpoint_dir / "sasrec_best.pt"))
    max_seq_length = config['sasrec']['max_seq_length']
    sasrec_metrics = evaluate_sasrec(model, test_data[:2000], num_items, config['eval']['top_k'], 
                                      max_seq_length, device, quiet)
    
    print("\nSASRec Results:")
    for k in config['eval']['top_k']:
        print(f"  Hit@{k}: {sasrec_metrics[f'hit@{k}']:.4f} | NDCG@{k}: {sasrec_metrics[f'ndcg@{k}']:.4f}")
    print(f"  MRR: {sasrec_metrics['mrr']:.4f}")
    
    return sasrec_metrics


# ==================== Main Function ====================

def main():
    config = load_config()
    quiet = config['data'].get('quiet_mode', True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading data...")
    vocab = load_vocab(config['data']['processed_path'])
    num_items = len(vocab)
    print(f"Vocabulary size: {num_items:,}")
    
    train, val, test = load_data(config['data']['processed_path'])
    print(f"Train: {len(train):,}, Val: {len(val):,}, Test: {len(test):,}")
    
    # Sample for faster evaluation
    eval_sample_size = 10000
    test_sample = test[:eval_sample_size] if len(test) > eval_sample_size else test
    print(f"Using {len(test_sample):,} samples for evaluation")
    
    train_sample_size = 100000
    train_sample = train[:train_sample_size] if len(train) > train_sample_size else train
    print(f"Using {len(train_sample):,} samples for training CF models")
    
    checkpoint_dir = Path(__file__).parent.parent / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    # ========== 1. Popularity Baseline ==========
    print("\n" + "="*50)
    print("Training Popularity Baseline...")
    print("="*50)
    
    popularity = PopularityBaseline()
    popularity.fit(train_sample)
    
    pop_metrics = evaluate_baseline(popularity, test_sample, num_items, config['eval']['top_k'], quiet)
    print("\nPopularity Baseline Results:")
    for k in config['eval']['top_k']:
        print(f"  Hit@{k}: {pop_metrics[f'hit@{k}']:.4f} | NDCG@{k}: {pop_metrics[f'ndcg@{k}']:.4f}")
    print(f"  MRR: {pop_metrics['mrr']:.4f}")
    
    # ========== 2. Item-based CF ==========
    print("\n" + "="*50)
    print("Training Item-based CF...")
    print("="*50)
    
    item_cf = ItemBasedCF(k_neighbors=30)
    item_cf.fit(train_sample, num_items)
    
    item_metrics = evaluate_baseline(item_cf, test_sample[:2000], num_items, config['eval']['top_k'], quiet)
    print("\nItem-based CF Results:")
    for k in config['eval']['top_k']:
        print(f"  Hit@{k}: {item_metrics[f'hit@{k}']:.4f} | NDCG@{k}: {item_metrics[f'ndcg@{k}']:.4f}")
    print(f"  MRR: {item_metrics['mrr']:.4f}")
    
    # ========== 3. User-based CF ==========
    print("\n" + "="*50)
    print("Training User-based CF...")
    print("="*50)
    
    user_cf = UserBasedCF(k_neighbors=30)
    user_cf.fit(train_sample, num_items)
    
    user_metrics = evaluate_baseline(user_cf, test_sample[:2000], num_items, config['eval']['top_k'], quiet)
    print("\nUser-based CF Results:")
    for k in config['eval']['top_k']:
        print(f"  Hit@{k}: {user_metrics[f'hit@{k}']:.4f} | NDCG@{k}: {user_metrics[f'ndcg@{k}']:.4f}")
    print(f"  MRR: {user_metrics['mrr']:.4f}")
    
    # ========== 4. SASRec ==========
    train_loader, val_loader, _ = create_dataloaders(
        config['data']['processed_path'],
        batch_size=config['training']['batch_size'],
        max_seq_length=config['sasrec']['max_seq_length']
    )
    
    sasrec_metrics = train_sasrec(config, num_items, train_loader, val_loader, 
                                   test_sample, checkpoint_dir, device, quiet)
    
    # ========== Summary ==========
    print("\n" + "="*60)
    print("BASELINE COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Model':<20} {'Hit@10':<10} {'NDCG@10':<10} {'MRR':<10}")
    print("-"*60)
    print(f"{'Popularity':<20} {pop_metrics['hit@10']:<10.4f} {pop_metrics['ndcg@10']:<10.4f} {pop_metrics['mrr']:<10.4f}")
    print(f"{'Item-based CF':<20} {item_metrics['hit@10']:<10.4f} {item_metrics['ndcg@10']:<10.4f} {item_metrics['mrr']:<10.4f}")
    print(f"{'User-based CF':<20} {user_metrics['hit@10']:<10.4f} {user_metrics['ndcg@10']:<10.4f} {user_metrics['mrr']:<10.4f}")
    print(f"{'SASRec':<20} {sasrec_metrics['hit@10']:<10.4f} {sasrec_metrics['ndcg@10']:<10.4f} {sasrec_metrics['mrr']:<10.4f}")
    
    # Save models
    save_baselines(popularity, item_cf, user_cf, str(checkpoint_dir))
    
    # Save metrics
    all_metrics = {
        'popularity': pop_metrics,
        'item_cf': item_metrics,
        'user_cf': user_metrics,
        'sasrec': sasrec_metrics
    }
    with open(checkpoint_dir / 'baseline_metrics.pkl', 'wb') as f:
        pickle.dump(all_metrics, f)
    
    print("\nAll baseline training complete!")


if __name__ == "__main__":
    main()
