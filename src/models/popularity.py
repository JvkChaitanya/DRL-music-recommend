"""
Popularity-based baseline recommendation model.
Recommends the most popular items regardless of user.
"""
import numpy as np
from collections import defaultdict


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
