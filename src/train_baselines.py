"""
Training and evaluation script for baseline models.
Compares Popularity, Item-CF, and User-CF baselines.
"""
import torch
import numpy as np
from pathlib import Path
import yaml
import pickle
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent))

from data.dataset import load_vocab
from models.baselines import PopularityBaseline, ItemBasedCF, UserBasedCF, save_baselines


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


def evaluate_baseline(model, test_data: list, num_items: int, top_k_values: list):
    """
    Evaluate a baseline model.
    
    Returns:
        metrics: Dict of metric values
    """
    hits = {k: 0 for k in top_k_values}
    ndcg = {k: 0.0 for k in top_k_values}
    mrr_sum = 0.0
    n_samples = len(test_data)
    
    for sample in tqdm(test_data, desc="Evaluating"):
        sequence = sample['sequence']
        target = sample['target']
        
        # Get predictions
        scores = model.predict_scores(sequence, num_items)
        
        # Get ranking
        sorted_items = np.argsort(scores)[::-1]
        
        # Find rank of target
        rank = np.where(sorted_items == target)[0]
        if len(rank) > 0:
            rank = rank[0] + 1  # 1-indexed
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


def main():
    config = load_config()
    
    print("Loading data...")
    vocab = load_vocab(config['data']['processed_path'])
    num_items = len(vocab)
    print(f"Vocabulary size: {num_items:,}")
    
    train, val, test = load_data(config['data']['processed_path'])
    print(f"Train: {len(train):,}, Val: {len(val):,}, Test: {len(test):,}")
    
    # Sample for faster evaluation (use 10K samples)
    eval_sample_size = 10000
    test_sample = test[:eval_sample_size] if len(test) > eval_sample_size else test
    print(f"Using {len(test_sample):,} samples for evaluation")
    
    # Use smaller training sample for CF models (too slow otherwise)
    train_sample_size = 100000
    train_sample = train[:train_sample_size] if len(train) > train_sample_size else train
    print(f"Using {len(train_sample):,} samples for training CF models")
    
    # ========== Popularity Baseline ==========
    print("\n" + "="*50)
    print("Training Popularity Baseline...")
    print("="*50)
    
    popularity = PopularityBaseline()
    popularity.fit(train_sample)
    
    pop_metrics = evaluate_baseline(popularity, test_sample, num_items, config['eval']['top_k'])
    print("\nPopularity Baseline Results:")
    for k in config['eval']['top_k']:
        print(f"  Hit@{k}: {pop_metrics[f'hit@{k}']:.4f} | NDCG@{k}: {pop_metrics[f'ndcg@{k}']:.4f}")
    print(f"  MRR: {pop_metrics['mrr']:.4f}")
    
    # ========== Item-based CF ==========
    print("\n" + "="*50)
    print("Training Item-based CF...")
    print("="*50)
    
    item_cf = ItemBasedCF(k_neighbors=30)
    item_cf.fit(train_sample, num_items)
    
    item_metrics = evaluate_baseline(item_cf, test_sample[:2000], num_items, config['eval']['top_k'])
    print("\nItem-based CF Results:")
    for k in config['eval']['top_k']:
        print(f"  Hit@{k}: {item_metrics[f'hit@{k}']:.4f} | NDCG@{k}: {item_metrics[f'ndcg@{k}']:.4f}")
    print(f"  MRR: {item_metrics['mrr']:.4f}")
    
    # ========== User-based CF ==========
    print("\n" + "="*50)
    print("Training User-based CF...")
    print("="*50)
    
    user_cf = UserBasedCF(k_neighbors=30)
    user_cf.fit(train_sample, num_items)
    
    user_metrics = evaluate_baseline(user_cf, test_sample[:2000], num_items, config['eval']['top_k'])
    print("\nUser-based CF Results:")
    for k in config['eval']['top_k']:
        print(f"  Hit@{k}: {user_metrics[f'hit@{k}']:.4f} | NDCG@{k}: {user_metrics[f'ndcg@{k}']:.4f}")
    print(f"  MRR: {user_metrics['mrr']:.4f}")
    
    # ========== Summary ==========
    print("\n" + "="*50)
    print("BASELINE COMPARISON SUMMARY")
    print("="*50)
    print(f"{'Model':<20} {'Hit@10':<10} {'NDCG@10':<10} {'MRR':<10}")
    print("-"*50)
    print(f"{'Popularity':<20} {pop_metrics['hit@10']:<10.4f} {pop_metrics['ndcg@10']:<10.4f} {pop_metrics['mrr']:<10.4f}")
    print(f"{'Item-based CF':<20} {item_metrics['hit@10']:<10.4f} {item_metrics['ndcg@10']:<10.4f} {item_metrics['mrr']:<10.4f}")
    print(f"{'User-based CF':<20} {user_metrics['hit@10']:<10.4f} {user_metrics['ndcg@10']:<10.4f} {user_metrics['mrr']:<10.4f}")
    
    # Save models
    checkpoint_dir = Path(__file__).parent.parent / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    save_baselines(popularity, item_cf, user_cf, str(checkpoint_dir))
    
    # Save metrics
    all_metrics = {
        'popularity': pop_metrics,
        'item_cf': item_metrics,
        'user_cf': user_metrics
    }
    with open(checkpoint_dir / 'baseline_metrics.pkl', 'wb') as f:
        pickle.dump(all_metrics, f)
    
    print("\nBaseline training complete!")


if __name__ == "__main__":
    main()
