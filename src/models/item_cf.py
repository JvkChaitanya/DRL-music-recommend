"""
Item-based Collaborative Filtering model.
Uses cosine similarity between item co-occurrence vectors.
This model is used as the backbone for the RL agent.
"""
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import pickle
from pathlib import Path


class ItemBasedCF:
    """
    Item-based Collaborative Filtering using item similarities.
    Uses cosine similarity between item co-occurrence vectors.
    """
    
    def __init__(self, k_neighbors: int = 50):
        self.k_neighbors = k_neighbors
        self.item_similarity = None
        self.num_items = None
    
    def fit(self, train_data: list, num_items: int):
        """
        Build item-item similarity matrix.
        
        Args:
            train_data: List of dicts with 'sequence' and 'target'
            num_items: Total vocabulary size
        """
        print("Fitting Item-based CF...")
        self.num_items = num_items
        
        # Build user-item matrix (sparse)
        # Each sequence represents a "user session"
        print("Building user-item matrix...")
        row_indices = []
        col_indices = []
        data = []
        
        for user_idx, sample in enumerate(tqdm(train_data, desc="Processing")):
            items = list(set(sample['sequence'])) + [sample['target']]
            for item in items:
                if 0 < item < num_items:  # Skip padding
                    row_indices.append(user_idx)
                    col_indices.append(item)
                    data.append(1.0)
        
        user_item_matrix = csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(len(train_data), num_items)
        )
        
        # Compute item-item similarity
        print("Computing item similarities...")
        # Normalize by item frequency
        item_norms = np.sqrt(np.array(user_item_matrix.power(2).sum(axis=0)).flatten() + 1e-8)
        
        # For efficiency, we compute similarity in batches
        batch_size = 1000
        self.item_similarity = lil_matrix((num_items, num_items))
        
        for start in tqdm(range(0, num_items, batch_size), desc="Similarity"):
            end = min(start + batch_size, num_items)
            batch = user_item_matrix[:, start:end].T.toarray()
            
            # Compute similarities for this batch against all items
            similarities = cosine_similarity(batch, user_item_matrix.T.toarray())
            
            # Keep only top-k neighbors
            for i in range(end - start):
                item_idx = start + i
                sim_scores = similarities[i]
                
                # Get top-k excluding self
                sim_scores[item_idx] = -1
                top_k_indices = np.argsort(sim_scores)[-self.k_neighbors:]
                
                for neighbor_idx in top_k_indices:
                    if sim_scores[neighbor_idx] > 0:
                        self.item_similarity[item_idx, neighbor_idx] = sim_scores[neighbor_idx]
        
        self.item_similarity = self.item_similarity.tocsr()
        print("Item-based CF fitted!")
    
    def predict(self, sequence: list, k: int = 10) -> list:
        """
        Predict top-K items based on sequence similarity.
        
        Args:
            sequence: User's recent items
            k: Number of items to return
            
        Returns:
            List of recommended items
        """
        scores = self.predict_scores(sequence, self.num_items)
        
        # Filter out items in sequence
        seen = set(sequence)
        for item in seen:
            if item < self.num_items:
                scores[item] = -np.inf
        
        top_k = np.argsort(scores)[-k:][::-1]
        return top_k.tolist()
    
    def predict_scores(self, sequence: list, num_items: int) -> np.ndarray:
        """
        Get scores for all items.
        
        Args:
            sequence: User's recent items
            num_items: Total number of items in vocabulary
            
        Returns:
            scores: (num_items,) array of item scores
        """
        scores = np.zeros(num_items)
        
        # Get actual size of similarity matrix (may be smaller due to memory optimization)
        similarity_size = self.item_similarity.shape[0]
        
        for item in sequence:
            if 0 < item < similarity_size:  # Check against similarity matrix size
                # Add similarity scores from this item
                item_sims = self.item_similarity[item].toarray().flatten()
                
                # If similarity matrix is smaller than num_items, pad with zeros
                if len(item_sims) < num_items:
                    padded_sims = np.zeros(num_items)
                    padded_sims[:len(item_sims)] = item_sims
                    scores += padded_sims
                else:
                    scores += item_sims
        
        return scores


def save_item_cf(item_cf, output_dir: str):
    """Save trained Item-CF model."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / 'item_cf.pkl', 'wb') as f:
        pickle.dump(item_cf, f)
    
    print(f"Saved Item-CF to {output_path}")


def load_item_cf(output_dir: str):
    """Load trained Item-CF model."""
    output_path = Path(output_dir)
    
    with open(output_path / 'item_cf.pkl', 'rb') as f:
        item_cf = pickle.load(f)
    
    return item_cf
