"""
Baseline recommendation models for comparison.
Implements:
1. Popularity-based baseline
2. Item-based Collaborative Filtering (Item-KNN)
3. User-based Collaborative Filtering (User-KNN)
"""
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import pickle
from pathlib import Path
from tqdm import tqdm
import yaml


def load_config():
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


class PopularityBaseline:
    """
    Simple popularity-based baseline.
    Recommends the most popular items regardless of user.
    """
    
    def __init__(self):
        self.item_counts = None
        self.popular_items = None
    
    def fit(self, train_data: list):
        """
        Fit on training data.
        
        Args:
            train_data: List of dicts with 'sequence' and 'target' keys
        """
        print("Fitting Popularity Baseline...")
        self.item_counts = defaultdict(int)
        
        for sample in train_data:
            for item in sample['sequence']:
                self.item_counts[item] += 1
            self.item_counts[sample['target']] += 1
        
        # Sort by count descending
        self.popular_items = sorted(
            self.item_counts.keys(),
            key=lambda x: self.item_counts[x],
            reverse=True
        )
        print(f"Fitted on {len(self.popular_items)} unique items")
    
    def predict(self, sequence: list, k: int = 10) -> list:
        """
        Predict top-K items (just returns most popular).
        
        Args:
            sequence: User's sequence (ignored)
            k: Number of items to return
            
        Returns:
            List of top-K popular items
        """
        # Filter out items already in sequence
        seen = set(sequence)
        recommendations = [item for item in self.popular_items if item not in seen]
        return recommendations[:k]
    
    def predict_scores(self, sequence: list, num_items: int) -> np.ndarray:
        """
        Get scores for all items.
        
        Returns:
            scores: (num_items,) array of popularity scores
        """
        scores = np.zeros(num_items)
        for item, count in self.item_counts.items():
            if item < num_items:
                scores[item] = count
        return scores


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
            
        Returns:
            scores: (num_items,) array of item scores
        """
        scores = np.zeros(num_items)
        
        for item in sequence:
            if 0 < item < num_items:
                # Add similarity scores from this item
                item_sims = self.item_similarity[item].toarray().flatten()
                scores += item_sims
        
        return scores


class UserBasedCF:
    """
    User-based Collaborative Filtering.
    Finds similar users and recommends what they listened to.
    """
    
    def __init__(self, k_neighbors: int = 50):
        self.k_neighbors = k_neighbors
        self.user_profiles = None
        self.user_items = None
        self.num_items = None
    
    def fit(self, train_data: list, num_items: int):
        """
        Build user profiles.
        
        Args:
            train_data: List of samples
            num_items: Vocabulary size
        """
        print("Fitting User-based CF...")
        self.num_items = num_items
        
        # Group by user
        user_to_items = defaultdict(set)
        
        for sample in train_data:
            user_id = sample.get('user_id', hash(tuple(sample['sequence'])))
            user_to_items[user_id].update(sample['sequence'])
            user_to_items[user_id].add(sample['target'])
        
        # Build user-item matrix
        print("Building user-item matrix...")
        user_ids = list(user_to_items.keys())
        self.user_id_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
        
        row_indices = []
        col_indices = []
        data = []
        
        for uid, items in user_to_items.items():
            user_idx = self.user_id_to_idx[uid]
            for item in items:
                if 0 < item < num_items:
                    row_indices.append(user_idx)
                    col_indices.append(item)
                    data.append(1.0)
        
        self.user_profiles = csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(len(user_ids), num_items)
        )
        
        self.user_items = user_to_items
        print(f"User-based CF fitted on {len(user_ids)} users")
    
    def predict(self, sequence: list, k: int = 10, user_id=None) -> list:
        """
        Predict items based on similar users.
        
        Args:
            sequence: User's recent items
            k: Number of items to return
            user_id: Optional user ID for known users
            
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
        Get scores based on similar users.
        
        Args:
            sequence: User's items
            
        Returns:
            scores: (num_items,) array
        """
        # Create query user vector
        query_vec = np.zeros(num_items)
        for item in sequence:
            if 0 < item < num_items:
                query_vec[item] = 1.0
        
        # Find similar users
        similarities = cosine_similarity(
            query_vec.reshape(1, -1),
            self.user_profiles.toarray()
        ).flatten()
        
        # Get top-k similar users
        top_user_indices = np.argsort(similarities)[-self.k_neighbors:]
        top_similarities = similarities[top_user_indices]
        
        # Aggregate their preferences
        scores = np.zeros(num_items)
        for user_idx, sim in zip(top_user_indices, top_similarities):
            if sim > 0:
                user_items = self.user_profiles[user_idx].toarray().flatten()
                scores += sim * user_items
        
        return scores


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
