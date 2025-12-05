"""
Quick baseline evaluation using sampled data.
Faster version that loads incrementally.
"""
import numpy as np
from collections import defaultdict
import pickle
from pathlib import Path
from tqdm import tqdm
import yaml


def load_config():
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


class PopularityBaseline:
    """Simple popularity-based baseline."""
    
    def __init__(self):
        self.item_counts = defaultdict(int)
        self.popular_items = None
    
    def fit(self, data: list):
        print("Fitting Popularity Baseline...")
        for sample in tqdm(data, desc="Processing"):
            for item in sample['sequence']:
                self.item_counts[item] += 1
            self.item_counts[sample['target']] += 1
        
        self.popular_items = sorted(
            self.item_counts.keys(),
            key=lambda x: self.item_counts[x],
            reverse=True
        )
        print(f"Fitted on {len(self.popular_items)} unique items")
    
    def predict_rank(self, sequence: list, target: int) -> int:
        """Get rank of target in predictions."""
        seen = set(sequence)
        rank = 1
        for item in self.popular_items:
            if item not in seen:
                if item == target:
                    return rank
                rank += 1
        return rank


def evaluate_baseline(model, test_data: list, top_k_values: list):
    """Evaluate baseline with Hit@K, NDCG@K, MRR."""
    hits = {k: 0 for k in top_k_values}
    ndcg = {k: 0.0 for k in top_k_values}
    mrr_sum = 0.0
    n_samples = len(test_data)
    
    for sample in tqdm(test_data, desc="Evaluating"):
        rank = model.predict_rank(sample['sequence'], sample['target'])
        
        if rank <= max(top_k_values):
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
    processed_path = Path(config['data']['processed_path'])
    
    # Load only 100K training samples
    print("Loading training data (first 100K samples)...")
    with open(processed_path / 'train.pkl', 'rb') as f:
        train_full = pickle.load(f)
    train = train_full[:100000]
    del train_full  # Free memory
    print(f"Loaded {len(train):,} training samples")
    
    # Load test data (first 10K)
    print("Loading test data (first 10K samples)...")
    with open(processed_path / 'test.pkl', 'rb') as f:
        test_full = pickle.load(f)
    test = test_full[:10000]
    del test_full
    print(f"Loaded {len(test):,} test samples")
    
    # Train and evaluate Popularity baseline
    print("\n" + "="*50)
    print("POPULARITY BASELINE")
    print("="*50)
    
    popularity = PopularityBaseline()
    popularity.fit(train)
    
    metrics = evaluate_baseline(popularity, test, config['eval']['top_k'])
    
    print("\n=== Results ===")
    for k in config['eval']['top_k']:
        print(f"Hit@{k}: {metrics[f'hit@{k}']:.4f} | NDCG@{k}: {metrics[f'ndcg@{k}']:.4f}")
    print(f"MRR: {metrics['mrr']:.4f}")
    
    # Save metrics
    checkpoint_dir = Path(__file__).parent.parent / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    with open(checkpoint_dir / 'popularity_metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)
    
    print("\nPopularity baseline complete!")
    print("\nNote: Item-CF and User-CF are computationally expensive.")
    print("Run SASRec next for a stronger neural baseline.")


if __name__ == "__main__":
    main()
