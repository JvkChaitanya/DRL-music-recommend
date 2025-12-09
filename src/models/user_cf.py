"""
User-based Collaborative Filtering model.
Finds similar users and recommends what they listened to.
"""
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict


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
