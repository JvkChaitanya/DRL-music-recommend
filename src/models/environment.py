"""
Simulated environment for RL-based music recommendation.
Uses offline data to simulate user responses.
"""
import torch
import numpy as np
from typing import Tuple, Dict, Optional
import pickle
from pathlib import Path


class MusicRecommendationEnv:
    """
    Simulated environment for music recommendation.
    
    State: User's listening history (sequence of tracks)
    Action: Select a track to recommend
    Reward: Simulated user response based on historical data
    """
    
    def __init__(
        self,
        data_path: str,
        vocab: dict,
        max_seq_length: int = 50,
        device: str = 'cpu'
    ):
        self.max_seq_length = max_seq_length
        self.device = device
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.num_items = len(vocab)
        
        # Load data
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        
        # Build user history for reward computation
        self._build_user_histories()
        
        # Current state
        self.current_idx = 0
        self.current_sequence = None
        self.current_user = None
    
    def _build_user_histories(self):
        """Build user listening history for reward computation."""
        self.user_histories = {}
        for sample in self.data:
            user_id = sample['user_id']
            if user_id not in self.user_histories:
                self.user_histories[user_id] = set()
            self.user_histories[user_id].update(sample['sequence'])
            self.user_histories[user_id].add(sample['target'])
    
    def reset(self, idx: Optional[int] = None) -> torch.Tensor:
        """
        Reset environment to a new episode.
        
        Returns:
            state: (max_seq_length,) tensor of track indices
        """
        if idx is None:
            self.current_idx = np.random.randint(len(self.data))
        else:
            self.current_idx = idx % len(self.data)
        
        sample = self.data[self.current_idx]
        self.current_user = sample['user_id']
        self.current_sequence = list(sample['sequence'])
        self.ground_truth = sample['target']
        
        return self._get_state()
    
    def _get_state(self) -> torch.Tensor:
        """Convert current sequence to padded tensor."""
        seq = self.current_sequence[-self.max_seq_length:]
        
        if len(seq) < self.max_seq_length:
            padding = [0] * (self.max_seq_length - len(seq))
            seq = padding + seq
        
        return torch.tensor(seq, dtype=torch.long, device=self.device)
    
    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, Dict]:
        """
        Take action (recommend a track) and get reward.
        
        Args:
            action: Track index to recommend
            
        Returns:
            next_state: New sequence state
            reward: Reward signal
            done: Whether episode is done
            info: Additional information
        """
        reward = self._compute_reward(action)
        
        # Update sequence with recommended track
        self.current_sequence.append(action)
        
        # Episode ends after one recommendation (can be extended)
        done = True
        
        info = {
            'ground_truth': self.ground_truth,
            'hit': action == self.ground_truth,
            'user_id': self.current_user
        }
        
        return self._get_state(), reward, done, info
    
    def _compute_reward(self, action: int) -> float:
        """
        Compute reward for recommending a track with enhanced shaping.
        
        Reward components:
        1. Exact hit bonus (1.0)
        2. Ranking-based reward (based on similarity to ground truth)
        3. User history match (0.3)
        4. Diversity bonus (exploration)
        5. Novelty penalty (avoid repetition)
        """
        reward = 0.0
        
        # 1. Exact hit bonus (predicted exactly what user listened to)
        if action == self.ground_truth:
            reward += 1.0
            return reward  # Perfect prediction, return immediately
        
        # 2. Ranking-based reward: Partial credit for "close" recommendations
        # This helps the agent learn even when not hitting exactly
        user_history_items = list(self.user_histories.get(self.current_user, set()))
        if len(user_history_items) > 0:
            # Check if action is similar to ground truth based on co-occurrence
            # Items the user likes together tend to be similar
            if action in user_history_items:
                # Give partial credit based on how popular this item is in user's history
                history_frequency = user_history_items.count(action)
                reward += 0.3 + min(0.2, history_frequency * 0.05)  # Up to 0.5 total
        
        # 3. Diversity bonus: Reward exploring different items
        # But penalize if too dissimilar from user's taste
        recent_items = set(self.current_sequence[-10:])
        if action not in recent_items:
            # Novel item - small exploration bonus
            reward += 0.1
        
        # 4. Novelty penalty for recommending very recently played
        if action in self.current_sequence[-3:]:
            reward -= 0.3  # Strong penalty for immediate repetition
        elif action in self.current_sequence[-5:]:
            reward -= 0.1  # Mild penalty for recent repetition
        
        # 5. Popularity baseline: Small penalty for recommending items
        # user has never interacted with (but not too harsh to allow exploration)
        if action not in user_history_items and len(user_history_items) > 0:
            reward -= 0.05
        
        return reward
    
    def get_candidates(self, top_k: int = 100) -> torch.Tensor:
        """
        Get candidate items for action selection.
        In practice, use SASRec to generate top-K candidates.
        
        Returns:
            candidates: (top_k,) tensor of candidate item indices
        """
        # For now, return random candidates (will be replaced by SASRec output)
        candidates = np.random.choice(
            range(2, self.num_items),  # Skip PAD and UNK
            size=min(top_k, self.num_items - 2),
            replace=False
        )
        return torch.tensor(candidates, dtype=torch.long, device=self.device)


class ReplayBuffer:
    """Experience replay buffer for RL training."""
    
    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool
    ):
        """Add experience to buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.position] = (
            state.cpu(),
            action,
            reward,
            next_state.cpu(),
            done
        )
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple:
        """Sample a batch of experiences."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.stack(states),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float),
            torch.stack(next_states),
            torch.tensor(dones, dtype=torch.bool)
        )
    
    def __len__(self):
        return len(self.buffer)
