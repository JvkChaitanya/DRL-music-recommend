"""
GPU-accelerated baseline models using PyTorch.
"""
import torch
import torch.nn.functional as F
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


class PopularityBaselineGPU:
    """GPU-accelerated popularity baseline."""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.popularity_scores = None
    
    def fit(self, data: list, num_items: int):
        print(f"Fitting Popularity Baseline on {self.device}...")
        counts = torch.zeros(num_items, device=self.device)
        
        for sample in tqdm(data, desc="Counting"):
            for item in sample['sequence']:
                if 0 < item < num_items:
                    counts[item] += 1
            if 0 < sample['target'] < num_items:
                counts[sample['target']] += 1
        
        self.popularity_scores = counts
        print(f"Fitted on {(counts > 0).sum().item()} unique items")
    
    def predict(self, sequences: torch.Tensor, k: int = 10) -> torch.Tensor:
        """
        Predict top-k items for batch of sequences.
        
        Args:
            sequences: (batch, seq_len) tensor
            k: number of items to return
            
        Returns:
            top_k: (batch, k) tensor of item indices
        """
        batch_size = sequences.size(0)
        scores = self.popularity_scores.unsqueeze(0).expand(batch_size, -1).clone()
        
        # Mask items already in sequence
        for i in range(batch_size):
            seq_items = sequences[i][sequences[i] > 0]
            scores[i, seq_items] = -float('inf')
        
        _, top_k = scores.topk(k, dim=1)
        return top_k


class ItemCFGPU:
    """GPU-accelerated Item-based Collaborative Filtering with Batch Processing."""
    
    def __init__(self, k_neighbors: int = 50, device='cuda'):
        self.k_neighbors = k_neighbors
        self.device = device
        self.item_similarity_indices = None # Store as top-k indices to save memory
        self.item_similarity_values = None  # Store as top-k values
        self.active_item_map = None
        self.reverse_item_map = None
        self.num_active_items = 0
    
    def fit(self, data: list, num_items: int):
        print(f"Fitting Item-CF on {self.device} (Batched Sparse Approach)...")
        
        # 1. Identify active items and map them
        print("Identifying active items...")
        unique_items = set()
        for sample in data:
            unique_items.update(sample['sequence'])
            unique_items.add(sample['target'])
        
        if 0 in unique_items: unique_items.remove(0)
        
        sorted_items = sorted(list(unique_items))
        self.num_active_items = len(sorted_items)
        print(f"Active items: {self.num_active_items:,}")
        
        self.active_item_map = {item: idx for idx, item in enumerate(sorted_items)}
        self.reverse_item_map = {idx: item for idx, item in enumerate(sorted_items)}
        
        # 2. Build Sparse Item-Session Matrix R
        # Shape: (Num_Items, Num_Sessions)
        # R[i, j] = 1 if item i is in session j
        print("Building sparse interaction matrix...")
        
        indices = []
        values = []
        num_sessions = len(data)
        
        # Pre-compute session norms for cosine similarity later? 
        # Actually standard ItemCF uses cosine between item vectors.
        # Item vector = column in User-Item matrix. Here Session-Item.
        # So we want cosine sim between rows of R.
        
        item_counts = torch.zeros(self.num_active_items, device=self.device)
        
        for session_idx, sample in enumerate(tqdm(data, desc="Building Indices")):
            items = list(set(sample['sequence'] + [sample['target']]))
            for item in items:
                if item in self.active_item_map:
                    item_idx = self.active_item_map[item]
                    indices.append([item_idx, session_idx])
                    values.append(1.0)
                    item_counts[item_idx] += 1
        
        indices = torch.tensor(indices, dtype=torch.long).t().to(self.device)
        values = torch.tensor(values, dtype=torch.float32).to(self.device)
        
        # Create Sparse Matrix R
        R = torch.sparse_coo_tensor(
            indices, values, 
            (self.num_active_items, num_sessions),
            device=self.device
        )
        
        # 3. Compute Similarity in Batches
        # Sim[i, j] = (R[i] . R[j]) / (|R[i]| |R[j]|)
        # We process batches of items i
        
        print("Computing top-k similarities in batches...")
        self.item_similarity_indices = torch.zeros((self.num_active_items, self.k_neighbors), dtype=torch.long, device=self.device)
        self.item_similarity_values = torch.zeros((self.num_active_items, self.k_neighbors), dtype=torch.float32, device=self.device)
        
        norms = torch.sqrt(item_counts + 1e-8)
        
        batch_size = 128 # Process 128 query items at a time
        
        # We need R in CSR format for faster matmul usually, but updated PyTorch handles COO well too.
        # Ideally: SIM = R @ R.T
        # We compute batch: SIM_Batch = R[batch] @ R.T
        # But slicing sparse matrix R[batch] is hard.
        # Instead, we construct R_dense_batch manually?
        # No, better to iterate query items, densify them, multiply by Sparse R.T? No R is (I, S).
        # We want (I, S) @ (S, I).
        
        # Let's iterate batches of query items
        # For batch of items B, extract their session vectors as Dense (B, S)
        # Then (B, S) @ R.T (S, I) -> (B, I)
        # But R.T is potentially large? R.T is (10k, 100k).
        # PyTorch sparse mm: (Sparse) @ (Dense).
        # So we want R (Sparse I, S) @ Q.T (Dense S, B) -> (I, B).
        # This gives similarities of ALL items vs Batch items.
        
        # We iterate over columns of the resulting similarity matrix?
        # Yes. We take a batch of items Q. Construct Q (Dense S, B).
        # Sim = R @ Q.
        # Then Sim[i, j] is similarity between item i and query item j.
        # We want top-k for each query item j.
        
        # Wait, if we iterate queries, we build Q.
        # Q should be (S, B).
        # Columns of Q are session vectors for query items.
        # R @ Q -> (I, S) @ (S, B) -> (I, B).
        
        # To construct Q (S, B) efficiently:
        # We can just extract rows from R?
        # R is coalesced.
        
        # Alternative: Convert R to CSR. Slicing rows is efficient-ish?
        # Or Just use the indices list we built.
        
        # Let's map item_id -> list of sessions (adjacency list) on CPU for fast batch construction.
        item_to_sessions = defaultdict(list)
        cpu_indices = indices.cpu().numpy()
        for k in range(indices.shape[1]):
            i, s = cpu_indices[0, k], cpu_indices[1, k]
            item_to_sessions[i].append(s)
            
        for start_idx in tqdm(range(0, self.num_active_items, batch_size), desc="Computing Batches"):
            end_idx = min(start_idx + batch_size, self.num_active_items)
            current_batch_size = end_idx - start_idx
            
            # Construct Dense Q (S, B)
            Q = torch.zeros((num_sessions, current_batch_size), device=self.device)
            
            # Fill Q
            for i in range(current_batch_size):
                item_id = start_idx + i
                sessions = item_to_sessions.get(item_id, [])
                if sessions:
                    # Q[sessions, i] = 1.0 (but vectorized)
                    # Use scatter or index put
                     # Just looping here is fast enough for small number of sessions per item (~50)
                     for s in sessions:
                         Q[s, i] = 1.0
            
            # Compute Raw Dot Products
            # Result: (Num_Items, Batch)
            # R (Sparse I, S) @ Q (Dense S, B)
            batch_sims = torch.mm(R, Q)
            
            # Normalize
            # Sim[i, b] = Dot[i, b] / (Norm[i] * Norm[b])
            # Norm[b] correspond to items start_idx ... end_idx
            
            batch_norms = norms[start_idx:end_idx] # (B,)
            
            # (I, B) / (I, 1) -> (I, B)
            batch_sims = batch_sims / (norms.unsqueeze(1) + 1e-8)
            # (I, B) / (1, B) -> (I, B)
            batch_sims = batch_sims / (batch_norms.unsqueeze(0) + 1e-8)
            
            # Mask self-similarity (set to -1)
            # Batch item 'j' (0..B-1) corresponds to global item 'start_idx + j'
            # We want batch_sims[start_idx + j, j] = -1
            for j in range(current_batch_size):
                batch_sims[start_idx + j, j] = -1.0
            
            # Top-K
            # We want top-k for each column (each query item)
            # values, indices = topk(k, dim=0) -> (K, B)
            vals, inds = batch_sims.topk(self.k_neighbors, dim=0)
            
            # Transpose to (B, K) and store
            self.item_similarity_values[start_idx:end_idx] = vals.t()
            self.item_similarity_indices[start_idx:end_idx] = inds.t()
            
            del Q, batch_sims, vals, inds
            
        print("Item-CF fitted!")
    
    def predict(self, sequences: torch.Tensor, k: int = 10) -> torch.Tensor:
        """Predict top-k items."""
        batch_size = sequences.size(0)
        
        # Output container
        scores = torch.zeros(batch_size, self.num_active_items, device=self.device)
        
        sequences_cpu = sequences.cpu().numpy()
        
        for i in range(batch_size):
            seq = sequences_cpu[i]
            # Filter active
            active_inds = [self.active_item_map[x] for x in seq if x in self.active_item_map]
            if not active_inds:
                continue
                
            active_inds_tensor = torch.tensor(active_inds, device=self.device, dtype=torch.long)
            
            # Aggregate similarity scores
            # Retrieve neighbors and scores for items in history
            # Indices: (Len_Seq, K), Values: (Len_Seq, K)
            neighbor_inds = self.item_similarity_indices[active_inds_tensor]
            neighbor_vals = self.item_similarity_values[active_inds_tensor]
            
            # Scatter add to scores
            # We flatten to (Len_Seq * K)
            flat_inds = neighbor_inds.view(-1)
            flat_vals = neighbor_vals.view(-1)
            
            # Accumulate scores
            # scores[i].index_add_(0, flat_inds, flat_vals)
            # Note: accumulating directly into the row
            scores[i].scatter_add_(0, flat_inds, flat_vals)
            
            # Mask already seen
            scores[i, active_inds_tensor] = -float('inf')
            
        # Top-K
        _, dense_top_k = scores.topk(k, dim=1)
        
        # Remap
        dense_top_k_cpu = dense_top_k.cpu().numpy()
        global_top_k = np.zeros((batch_size, k), dtype=np.int64)
        
        for i in range(batch_size):
            for j in range(k):
                idx = dense_top_k_cpu[i, j]
                global_top_k[i, j] = self.reverse_item_map[idx]
        
        return torch.tensor(global_top_k, device=self.device)


def evaluate_gpu(model, test_data: list, num_items: int, top_k_values: list, device: str, batch_size: int = 256):
    """Batch evaluation on GPU."""
    print(f"Evaluating on {device}...")
    
    max_k = max(top_k_values)
    hits = {k: 0 for k in top_k_values}
    ndcg = {k: 0.0 for k in top_k_values}
    mrr_sum = 0.0
    
    # Process in batches
    n_samples = len(test_data)
    max_seq_len = max(len(s['sequence']) for s in test_data)
    
    for start in tqdm(range(0, n_samples, batch_size), desc="Evaluating"):
        end = min(start + batch_size, n_samples)
        batch = test_data[start:end]
        
        # Pad sequences
        sequences = torch.zeros(len(batch), max_seq_len, dtype=torch.long, device=device)
        targets = torch.zeros(len(batch), dtype=torch.long, device=device)
        
        for i, sample in enumerate(batch):
            seq = sample['sequence']
            sequences[i, :len(seq)] = torch.tensor(seq, device=device)
            targets[i] = sample['target']
        
        # Get predictions
        top_k_preds = model.predict(sequences, k=max_k)
        
        # Compute metrics
        for i in range(len(batch)):
            target = targets[i].item()
            preds = top_k_preds[i].tolist()
            
            if target in preds:
                rank = preds.index(target) + 1
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
    
    # Check GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    processed_path = Path(config['data']['processed_path'])
    
    # Load vocab
    print("Loading vocabulary...")
    with open(processed_path / 'vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    num_items = len(vocab)
    print(f"Vocabulary size: {num_items:,}")
    
    # Load data samples
    print("Loading training data...")
    with open(processed_path / 'train.pkl', 'rb') as f:
        train = pickle.load(f)
    train = train[:100000]  # Use 100K for speed
    print(f"Using {len(train):,} training samples")
    
    print("Loading test data...")
    with open(processed_path / 'test.pkl', 'rb') as f:
        test = pickle.load(f)
    test = test[:10000]  # Use 10K for speed
    print(f"Using {len(test):,} test samples")
    
    # ========== Popularity Baseline ==========
    print("\n" + "="*50)
    print("POPULARITY BASELINE (GPU)")
    print("="*50)
    
    popularity = PopularityBaselineGPU(device=device)
    popularity.fit(train, num_items)
    
    pop_metrics = evaluate_gpu(popularity, test, num_items, config['eval']['top_k'], device)
    print("\nResults:")
    for k in config['eval']['top_k']:
        print(f"  Hit@{k}: {pop_metrics[f'hit@{k}']:.4f} | NDCG@{k}: {pop_metrics[f'ndcg@{k}']:.4f}")
    print(f"  MRR: {pop_metrics['mrr']:.4f}")
    
class UserCFGPU:
    """
    GPU-accelerated User-based (Sequence-based) Collaborative Filtering.
    Finds similar user sequences in training data and recommends their next items.
    """
    
    def __init__(self, k_neighbors: int = 50, device='cuda'):
        self.k_neighbors = k_neighbors
        self.device = device
        # We store training sequences as sparse or dense matrix?
        # Num sequences = 10k (subset) to 11M (full).
        # Items = 400k.
        # User-Item matrix is (Num_Sequences, Num_Items).
        # We want to find cosine sim between Test_Sequence and Train_Sequences.
        self.train_matrix = None # Sparse (Num_Train, Num_Items)
        self.train_targets = None # (Num_Train,)
        self.dim = 0
        
    def fit(self, data: list, num_items: int):
        print(f"Fitting User-CF on {self.device}...")
        self.dim = num_items
        
        # Build Sparse User-Item Matrix
        indices = []
        values = []
        targets = []
        
        for idx, sample in enumerate(tqdm(data, desc="Building User Matrix")):
            # User vector includes sequence items
            items = list(set(sample['sequence']))
            for item in items:
                if 0 < item < num_items:
                    indices.append([idx, item])
                    values.append(1.0)
            
            targets.append(sample['target'])
            
        indices = torch.tensor(indices, dtype=torch.long).t().to(self.device)
        values = torch.tensor(values, dtype=torch.float32).to(self.device)
        self.train_targets = torch.tensor(targets, dtype=torch.long, device=self.device)
        
        # Create Sparse Matrix (Users, Items)
        self.train_matrix = torch.sparse_coo_tensor(
            indices, values,
            (len(data), num_items),
            device=self.device
        )
        
        # Precompute norms for cosine similarity
        # Row sums of squared values. Since binary, just count of items.
        # Can compute from coalesced indices or dense sum if small enough.
        # Sparse sum is not fully supported for dim arg in older pytorch.
        # Let's count explicitly.
        print("Computing training norms...")
        # self.train_norms = torch.sparse.sum(self.train_matrix, dim=1).to_dense() # might not exist
        # Manual norm computation
        ones = torch.ones(num_items, 1, device=self.device)
        # (Users, Items) @ (Items, 1) -> (Users, 1) counts
        self.train_norms = torch.sparse.mm(self.train_matrix, ones).squeeze()
        self.train_norms = torch.sqrt(self.train_norms + 1e-8)
        
        print(f"User-CF fitted on {len(data)} sequences.")

    def predict(self, sequences: torch.Tensor, k: int = 10) -> torch.Tensor:
        """
        Predict based on similar training sequences.
        For each test sequence, find K nearest training sequences.
        Recommendation score = sum(sim * 1_target_is_item)
        Basically, vote for the targets of the nearest neighbors weighted by similarity.
        """
        batch_size = sequences.size(0)
        
        # 1. Build Dense Query Matrix (Batch, Items)
        # Since batch is small (256), dense is fine. (256 * 400k * 4 bytes = 400MB)
        # Actually 400k items is large for dense if batch is large.
        # But we can use sparse query matrix?
        # MatMul: (Train_Users, Items) @ (Items, Batch_Users) -> (Train_Users, Batch_Users)
        
        # Construct Sparse Query Matrix efficiently
        q_indices = []
        q_values = []
        
        seq_cpu = sequences.cpu()
        for i in range(batch_size):
            # Get unique items in sequence
            items = seq_cpu[i].unique()
            for item in items:
                if 0 < item < self.dim:
                    q_indices.append([item.item(), i]) # Transposed for matmul
                    q_values.append(1.0)
        
        if not q_indices:
            return torch.zeros((batch_size, k), dtype=torch.long, device=self.device)
            
        q_indices = torch.tensor(q_indices, dtype=torch.long).t().to(self.device)
        q_values = torch.tensor(q_values, dtype=torch.float32).to(self.device)
        
        Q_T = torch.sparse_coo_tensor(
            q_indices, q_values,
            (self.dim, batch_size),
            device=self.device
        )
        
        # 2. Compute Similarities
        # Sim = Train @ Q.T
        # Result: (Num_Train, Batch_Size) - Dense
        dots = torch.sparse.mm(self.train_matrix, Q_T).to_dense() # (Train, Batch)
        
        # Normalize
        # Q norms
        q_norms = torch.sparse.mm(Q_T.t(), torch.ones(self.dim, 1, device=self.device)).squeeze()
        q_norms = torch.sqrt(q_norms + 1e-8)
        
        # dots[t, b] / (train_norm[t] * q_norm[b])
        # (Train, Batch) / (Train, 1) -> (Train, Batch)
        sims = dots / (self.train_norms.unsqueeze(1) + 1e-8)
        # (Train, Batch) / (1, Batch)
        sims = sims / (q_norms.unsqueeze(0) + 1e-8)
        
        # 3. Find Top-K Neighbors for each test user
        # We want top neighbors per column (test user)
        # (K_neighbors, Batch)
        top_sims, top_indices = sims.topk(self.k_neighbors, dim=0)
        
        # 4. Aggregate Votes for Targets
        # Each neighbor votes for its 'target' item
        # We need to map neighbors -> targets
        # top_indices is (K_neigh, Batch). Values are indices into train_targets.
        
        # Get targets of neighbors: (K_neigh, Batch)
        neighbor_targets = self.train_targets[top_indices]
        
        # Ensure we work with flattened arrays for scatter/bincount
        # This is tricky vectorized.
        # Iterate over batch is easier? Batch=256 is small.
        
        predictions = torch.zeros((batch_size, k), dtype=torch.long, device=self.device)
        
        top_sims_cpu = top_sims.t().cpu() # (Batch, K_neigh)
        neighbor_targets_cpu = neighbor_targets.t().cpu() # (Batch, K_neigh)
        sequences_cpu = sequences.cpu()
        
        # Hybrid CPU/GPU voting because 'scatter_add' with variable targets is hard
        # Actually we can do it on GPU but dense scores size (Batch, Items) is big (400k items)
        # Sparse scatter add?
        # Let's try CPU voting loop, it should be fast enough for K=50
        
        for i in range(batch_size):
            target_scores = {}
            seen_items = set(sequences_cpu[i].numpy())
            
            for j in range(self.k_neighbors):
                tgt = neighbor_targets_cpu[i, j].item()
                weight = top_sims_cpu[i, j].item()
                
                if tgt not in seen_items and tgt > 0:
                    target_scores[tgt] = target_scores.get(tgt, 0.0) + weight
            
            # Sort
            if target_scores:
                best_items = sorted(target_scores.items(), key=lambda x: x[1], reverse=True)[:k]
                for idx, (item, score) in enumerate(best_items):
                    predictions[i, idx] = item
                
        return predictions


def evaluate_gpu(model, test_data: list, num_items: int, top_k_values: list, device: str, batch_size: int = 256):
    """Batch evaluation on GPU."""
    print(f"Evaluating on {device}...")
    
    max_k = max(top_k_values)
    hits = {k: 0 for k in top_k_values}
    ndcg = {k: 0.0 for k in top_k_values}
    mrr_sum = 0.0
    
    # Process in batches
    n_samples = len(test_data)
    max_seq_len = max(len(s['sequence']) for s in test_data)
    
    for start in tqdm(range(0, n_samples, batch_size), desc="Evaluating"):
        end = min(start + batch_size, n_samples)
        batch = test_data[start:end]
        
        # Pad sequences
        sequences = torch.zeros(len(batch), max_seq_len, dtype=torch.long, device=device)
        targets = torch.zeros(len(batch), dtype=torch.long, device=device)
        
        for i, sample in enumerate(batch):
            seq = sample['sequence']
            sequences[i, :len(seq)] = torch.tensor(seq, device=device)
            targets[i] = sample['target']
        
        # Get predictions
        top_k_preds = model.predict(sequences, k=max_k)
        
        # Compute metrics
        for i in range(len(batch)):
            target = targets[i].item()
            preds = top_k_preds[i].tolist()
            
            if target in preds:
                rank = preds.index(target) + 1
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


import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None, help='Limit training samples to avoid OOM')
    args = parser.parse_args()

    config = load_config()
    
    # Check GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    processed_path = Path(config['data']['processed_path'])
    
    # Load vocab
    print("Loading vocabulary...")
    with open(processed_path / 'vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    num_items = len(vocab)
    print(f"Vocabulary size: {num_items:,}")
    
    # Load data samples
    print("Loading training data...")
    with open(processed_path / 'train.pkl', 'rb') as f:
        train_full = pickle.load(f)
        
    print(f"Full training data: {len(train_full):,} samples")
    
    # Popularity uses FULL data (it's cheap)
    print("\n" + "="*50)
    print("POPULARITY BASELINE (GPU)")
    print("="*50)
    
    popularity = PopularityBaselineGPU(device=device)
    popularity.fit(train_full, num_items)
    
    # Load test data
    print("Loading test data...")
    with open(processed_path / 'test.pkl', 'rb') as f:
        test = pickle.load(f)
    print(f"Using {len(test):,} test samples")
    
    # Evaluate Popularity
    pop_metrics = evaluate_gpu(popularity, test, num_items, config['eval']['top_k'], device)
    
    # For CF models, apply limit if needed
    if args.limit and args.limit < len(train_full):
        print(f"\nSubsampling training data for CF models to {args.limit:,} samples (Randomized Stratified)...")
        # Use random sampling to preserve user/item distribution
        # This acts as stratified sampling by user activity volume
        indices = np.random.choice(len(train_full), args.limit, replace=False)
        train_cf = [train_full[i] for i in indices]
    else:
        train_cf = train_full
        
    # Free full training memory if we created a subset
    if train_cf is not train_full:
        del train_full
        import gc
        gc.collect()
        
    print(f"CF Training Data: {len(train_cf):,} samples")
    
    # Global metrics dict
    all_metrics = {'popularity': pop_metrics}
    
    # ========== Item-CF Baseline ==========
    print("\n" + "="*50)
    print("ITEM-CF BASELINE (GPU)")
    print("="*50)
    
    if device == 'cuda': torch.cuda.empty_cache()
    
    item_cf = ItemCFGPU(k_neighbors=50, device=device)
    item_cf.fit(train_cf, num_items)
    
    cf_metrics = evaluate_gpu(item_cf, test, num_items, config['eval']['top_k'], device)
    all_metrics['item_cf'] = cf_metrics
    
    # ========== User-CF Baseline ==========
    print("\n" + "="*50)
    print("USER-CF BASELINE (GPU)")
    print("="*50)
    
    if device == 'cuda': torch.cuda.empty_cache()
    
    user_cf = UserCFGPU(k_neighbors=50, device=device)
    user_cf.fit(train_cf, num_items)
    
    user_metrics = evaluate_gpu(user_cf, test, num_items, config['eval']['top_k'], device, batch_size=128)
    all_metrics['user_cf'] = user_metrics
    
    # ========== Summary ==========

    print("\n" + "="*50)
    print("BASELINE COMPARISON SUMMARY")
    print("="*50)
    print(f"{'Model':<20} {'Hit@10':<10} {'NDCG@10':<10} {'MRR':<10}")
    print("-"*50)
    print(f"{'Popularity':<20} {pop_metrics['hit@10']:<10.4f} {pop_metrics['ndcg@10']:<10.4f} {pop_metrics['mrr']:<10.4f}")
    print(f"{'Item-CF':<20} {cf_metrics['hit@10']:<10.4f} {cf_metrics['ndcg@10']:<10.4f} {cf_metrics['mrr']:<10.4f}")
    print(f"{'User-CF':<20} {user_metrics['hit@10']:<10.4f} {user_metrics['ndcg@10']:<10.4f} {user_metrics['mrr']:<10.4f}")
    
    # Save metrics
    checkpoint_dir = Path(__file__).parent.parent / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    with open(checkpoint_dir / 'baseline_metrics.pkl', 'wb') as f:
        pickle.dump(all_metrics, f)
    
    print("\nBaseline training complete!")


if __name__ == "__main__":
    main()
