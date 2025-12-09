"""
Fast evaluation script comparing all models on a sample of test data.
Models: Popularity, User-CF, Item-CF, SASRec (if available), RL (if available)
"""
import torch
import numpy as np
import pickle
from pathlib import Path
import yaml
from tqdm import tqdm
from typing import Dict, List, Optional
import sys
import traceback

sys.path.insert(0, str(Path(__file__).parent))

from data.dataset import load_vocab
from models.popularity import PopularityBaseline
from models.item_cf import ItemBasedCF
from models.user_cf import UserBasedCF
from models.sasrec import SASRec
from models.rl_agent import ActorCriticAgent


def load_baselines(output_dir: str):
    """Load trained baseline models."""
    output_path = Path(output_dir)
    
    with open(output_path / 'popularity_baseline.pkl', 'rb') as f:
        popularity = pickle.load(f)
    
    with open(output_path / 'item_cf.pkl', 'rb') as f:
        item_cf = pickle.load(f)
    
    with open(output_path / 'user_cf.pkl', 'rb') as f:
        user_cf = pickle.load(f)
    
    return popularity, item_cf, user_cf


def load_config():
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_test_data(processed_path: str, sample_size: Optional[int] = None):
    """Load test dataset with optional sampling."""
    with open(Path(processed_path) / 'test.pkl', 'rb') as f:
        data = pickle.load(f)
    
    if sample_size and sample_size < len(data):
        # Random sample
        indices = np.random.choice(len(data), sample_size, replace=False)
        data = [data[i] for i in indices]
        print(f"Sampled {sample_size} test examples (from {len(indices)} total)")
    
    return data


def compute_metrics(predictions: List[int], target: int, k_values: List[int]) -> Dict:
    """Compute Hit@K and NDCG@K metrics."""
    metrics = {}
    
    for k in k_values:
        top_k = predictions[:k]
        
        # Hit@K
        hit = 1 if target in top_k else 0
        metrics[f'hit@{k}'] = hit
        
        # NDCG@K
        if target in top_k:
            position = top_k.index(target) + 1
            ndcg = 1.0 / np.log2(position + 1)
        else:
            ndcg = 0.0
        metrics[f'ndcg@{k}'] = ndcg
    
    return metrics


def evaluate_model(model, test_data, num_items, k_values, model_name,
                   rl_agent=None, item_cf=None, sasrec=None, max_seq_length=200, device='cpu'):
    """
    Generic evaluation function for all models.
    """
    print(f"\n{'='*60}")
    print(f"EVALUATING {model_name.upper()}")
    print(f"{'='*60}")
    
    all_metrics = {f'hit@{k}': [] for k in k_values}
    all_metrics.update({f'ndcg@{k}': [] for k in k_values})
    
    state_dim = 128
    similarity_size = item_cf.item_similarity.shape[0] if item_cf else 0
    
    for sample in tqdm(test_data, desc=f"{model_name} Eval", ncols=80):
        sequence = sample['sequence']
        target = sample['target']
        
        try:
            if model_name == "RL":
                # RL re-ranks Item-CF candidates
                item_scores = item_cf.predict_scores(sequence, num_items)
                for item in sequence:
                    if item < len(item_scores):
                        item_scores[item] = -np.inf
                
                top_candidates_np = np.argsort(item_scores)[-100:][::-1].copy()
                top_candidates = torch.tensor(top_candidates_np, dtype=torch.long, device=device)
                
                # State embedding
                recent_items = [item for item in sequence[-10:] if item > 0]
                state_embedding = torch.zeros(state_dim, device=device)
                
                if len(recent_items) > 0:
                    weights = np.exp(np.linspace(-1, 0, len(recent_items)))
                    weights = weights / weights.sum()
                    valid_embeddings, valid_weights = [], []
                    
                    for idx, item in enumerate(recent_items):
                        if 0 < item < similarity_size:
                            sim_vec = item_cf.item_similarity[item].toarray().flatten()
                            if len(sim_vec) >= state_dim:
                                top_k_indices = np.argpartition(sim_vec, -state_dim)[-state_dim:]
                                embedding = sim_vec[top_k_indices]
                            else:
                                embedding = np.zeros(state_dim)
                                embedding[:len(sim_vec)] = sim_vec
                            valid_embeddings.append(embedding)
                            valid_weights.append(weights[idx])
                    
                    if valid_embeddings:
                        valid_weights = np.array(valid_weights) / np.array(valid_weights).sum()
                        weighted_emb = np.average(valid_embeddings, axis=0, weights=valid_weights)
                        state_embedding = torch.tensor(weighted_emb, dtype=torch.float32, device=device)
                
                # Candidate embeddings
                candidate_embeddings = torch.zeros(len(top_candidates), state_dim, device=device)
                for i, cand_item in enumerate(top_candidates):
                    cand_item = int(cand_item.item())
                    if 0 < cand_item < similarity_size:
                        sim_vec = item_cf.item_similarity[cand_item].toarray().flatten()
                        if len(sim_vec) >= state_dim:
                            top_k_indices = np.argpartition(sim_vec, -state_dim)[-state_dim:]
                            embedding = sim_vec[top_k_indices]
                        else:
                            embedding = np.zeros(state_dim)
                            embedding[:len(sim_vec)] = sim_vec
                        candidate_embeddings[i] = torch.tensor(embedding, dtype=torch.float32, device=device)
                
                # RL ranking
                with torch.no_grad():
                    action_pref = rl_agent.actor(state_embedding.unsqueeze(0)).squeeze(0)
                    scores = torch.matmul(candidate_embeddings, action_pref)
                    ranked_indices = torch.argsort(scores, descending=True)
                
                predictions = [top_candidates[idx].item() for idx in ranked_indices[:max(k_values)]]
            
            elif model_name == "SASRec":
                # SASRec uses neural network forward pass
                # Pad/truncate sequence to max_seq_length
                seq = sequence[-max_seq_length:]
                if len(seq) < max_seq_length:
                    padding = [0] * (max_seq_length - len(seq))
                    seq = padding + seq
                
                seq_tensor = torch.tensor([seq], dtype=torch.long, device=device)
                
                with torch.no_grad():
                    logits = sasrec(seq_tensor).squeeze(0)  # (num_items,)
                
                # Remove items in sequence
                for item in sequence:
                    if item < len(logits):
                        logits[item] = -float('inf')
                
                # Get top-K predictions
                top_k_indices = torch.argsort(logits, descending=True)[:max(k_values)]
                predictions = top_k_indices.cpu().tolist()
            
            else:
                # Regular baseline models (Popularity, User-CF, Item-CF)
                item_scores = model.predict_scores(sequence, num_items)
                
                # Remove items in sequence
                for item in sequence:
                    if item < len(item_scores):
                        item_scores[item] = -np.inf
                
                # Get top-K predictions
                top_k_indices = np.argsort(item_scores)[-max(k_values):][::-1].copy()
                predictions = top_k_indices.tolist()
            
            # Compute metrics
            metrics = compute_metrics(predictions, target, k_values)
            for key, value in metrics.items():
                all_metrics[key].append(value)
        
        except Exception as e:
            print(f"\n{'='*60}")
            print(f"ERROR in {model_name} evaluation:")
            print(f"Sample: sequence length={len(sequence)}, target={target}")
            print(f"Exception: {e}")
            print("Full traceback:")
            traceback.print_exc()
            print(f"{'='*60}\n")
            # Add zeros for failed samples
            for key in all_metrics:
                all_metrics[key].append(0.0)
    
    # Average metrics
    avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
    return avg_metrics


def print_comparison_table(results: Dict[str, Dict]):
    """Print comprehensive comparison table."""
    print("\n" + "="*100)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*100)
    
    # Get all metrics
    all_metrics = sorted(list(results[list(results.keys())[0]].keys()))
    
    # Header
    header = f"{'Metric':<12}"
    for model_name in results.keys():
        header += f"{model_name:<20}"
    print(header)
    print("-"*100)
    
    # Rows
    for metric in all_metrics:
        row = f"{metric:<12}"
        values = []
        for model_name in results.keys():
            value = results[model_name][metric]
            values.append(value)
            row += f"{value:<20.4f}"
        
        # Highlight best
        best_idx = np.argmax(values)
        print(row + f"  ← BEST" if values[best_idx] > 0 else row)
    
    print("="*100)
    
    # Summary
    print("\nSUMMARY:")
    for model_name, metrics in results.items():
        avg_hit = np.mean([v for k, v in metrics.items() if 'hit' in k])
        avg_ndcg = np.mean([v for k, v in metrics.items() if 'ndcg' in k])
        print(f"  {model_name:<20} Avg Hit: {avg_hit:.4f}  Avg NDCG: {avg_ndcg:.4f}")


def main():
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Configuration
    SAMPLE_SIZE = 1000  # Evaluate on 1K samples (change to None for full test set)
    k_values = config['rl'].get('top_k', [5, 10, 20])
    
    # Load data
    vocab = load_vocab(config['data']['processed_path'])
    num_items = len(vocab)
    test_data = load_test_data(config['data']['processed_path'], SAMPLE_SIZE)
    
    print(f"Vocabulary size: {num_items:,}")
    print(f"Test samples: {len(test_data):,}")
    print(f"Metrics: Hit@K and NDCG@K for K = {k_values}\n")
    
    checkpoint_dir = Path(__file__).parent.parent / "checkpoints"
    results = {}
    
    # Load baseline models using the built-in loader
    # This loads all three: popularity, user_cf, item_cf
    popularity = None
    user_cf = None
    item_cf = None  # Initialize to ensure it's available for RL
    
    try:
        popularity, item_cf, user_cf = load_baselines(str(checkpoint_dir))  # Fixed: correct order!
        print("✓ All baseline models loaded successfully")
    except Exception as e:
        print(f"✗ Could not load all baselines together: {e}")
        print("Trying to load models individually...")
        
        # Fallback: Try loading individually
        try:
            with open(checkpoint_dir / 'popularity_baseline.pkl', 'rb') as f:
                popularity = pickle.load(f)
            print("  ✓ Popularity baseline loaded")
        except Exception as e2:
            print(f"  ✗ Popularity baseline failed: {e2}")
        
        try:
            with open(checkpoint_dir / 'user_cf.pkl', 'rb') as f:
                user_cf = pickle.load(f)
            print("  ✓ User-CF loaded")
        except Exception as e2:
            print(f"  ✗ User-CF failed: {e2}")
        
        try:
            with open(checkpoint_dir / 'item_cf.pkl', 'rb') as f:
                item_cf = pickle.load(f)
            print("  ✓ Item-CF loaded")
        except Exception as e2:
            print(f"  ✗ Item-CF failed: {e2}")
    
    # Evaluate each baseline if loaded
    if popularity is not None:
        try:
            results['Popularity'] = evaluate_model(
                popularity, test_data, num_items, k_values, "Popularity"
            )
        except Exception as e:
            print(f"✗ Popularity evaluation failed: {e}")
            traceback.print_exc()
    
    if user_cf is not None:
        try:
            results['User-CF'] = evaluate_model(
                user_cf, test_data, num_items, k_values, "User-CF"
            )
        except Exception as e:
            print(f"✗ User-CF evaluation failed: {e}")
            traceback.print_exc()
    
    if item_cf is not None:
        try:
            results['Item-CF'] = evaluate_model(
                item_cf, test_data, num_items, k_values, "Item-CF"
            )
        except Exception as e:
            print(f"✗ Item-CF evaluation failed: {e}")
            traceback.print_exc()

    
    # Try to load SASRec
    try:
        sasrec_path = checkpoint_dir / "sasrec_best.pt"
        if sasrec_path.exists():
            sasrec = SASRec(
                num_items=num_items,
                embedding_dim=config['sasrec']['embedding_dim'],
                num_heads=config['sasrec']['num_heads'],
                num_layers=config['sasrec']['num_layers'],
                max_seq_length=config['sasrec']['max_seq_length'],
                dropout=config['sasrec']['dropout']
            ).to(device)
            
            checkpoint = torch.load(sasrec_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                sasrec.load_state_dict(checkpoint['model_state_dict'])
            else:
                sasrec.load_state_dict(checkpoint)
            sasrec.eval()
            print("\n✓ SASRec model loaded")
            
            results['SASRec'] = evaluate_model(
                None, test_data, num_items, k_values, "SASRec",
                sasrec=sasrec, max_seq_length=config['sasrec']['max_seq_length'], device=device
            )
    except Exception as e:
        print(f"\nSASRec not available: {e}")
    
    # Try to load RL agent (requires Item-CF)
    if item_cf is not None:
        try:
            rl_agent = ActorCriticAgent(state_dim=128, hidden_dim=128, device=str(device))
            rl_agent.load(str(checkpoint_dir / "rl_agent_itemcf.pt"))
            print("\n✓ RL agent loaded")
            
            results['Item-CF+RL'] = evaluate_model(
                None, test_data, num_items, k_values, "RL",
                rl_agent=rl_agent, item_cf=item_cf, device=device
            )
        except Exception as e:
            print(f"\n✗ RL agent not available: {e}")
            traceback.print_exc()
    else:
        print("\n✗ RL agent evaluation skipped (Item-CF not loaded)")
    
    # Print comparison table
    print_comparison_table(results)
    
    # Save results
    results_data = {
        'results': results,
        'test_samples': len(test_data),
        'sample_size': SAMPLE_SIZE,
        'k_values': k_values
    }
    
    results_path = checkpoint_dir / "evaluation_comparison.pkl"
    with open(results_path, 'wb') as f:
        pickle.dump(results_data, f)
    print(f"\n✓ Results saved to: {results_path}")


if __name__ == "__main__":
    main()
