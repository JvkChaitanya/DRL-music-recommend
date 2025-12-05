"""
Actor-Critic RL Agent for music recommendation.
Uses SASRec embeddings as state representation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Tuple


class Actor(nn.Module):
    """
    Policy network that selects actions from candidates.
    Takes state (user embedding) and candidate embeddings,
    outputs probability distribution over candidates.
    """
    
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, state_dim)  # Project to embedding space
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute action preference embedding.
        
        Args:
            state: (batch, state_dim) user state embedding
            
        Returns:
            action_emb: (batch, state_dim) action preference in embedding space
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
    def select_action(
        self,
        state: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        temperature: float = 1.0
    ) -> Tuple[int, torch.Tensor]:
        """
        Select action from candidates.
        
        Args:
            state: (state_dim,) user state embedding
            candidate_embeddings: (num_candidates, state_dim) candidate item embeddings
            temperature: Softmax temperature for exploration
            
        Returns:
            action_idx: Index in candidate list
            log_prob: Log probability of selected action
        """
        # Get action preference
        action_pref = self.forward(state.unsqueeze(0)).squeeze(0)
        
        # Score candidates by dot product
        scores = torch.matmul(candidate_embeddings, action_pref) / temperature
        probs = F.softmax(scores, dim=0)
        
        # Sample action
        dist = torch.distributions.Categorical(probs)
        action_idx = dist.sample()
        log_prob = dist.log_prob(action_idx)
        
        return action_idx.item(), log_prob


class Critic(nn.Module):
    """
    Value network that estimates expected return from a state.
    """
    
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Estimate value of state.
        
        Args:
            state: (batch, state_dim) user state embedding
            
        Returns:
            value: (batch, 1) estimated value
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ActorCriticAgent:
    """
    Actor-Critic agent for music recommendation.
    Uses advantage actor-critic (A2C) algorithm.
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 128,
        actor_lr: float = 0.0001,
        critic_lr: float = 0.001,
        gamma: float = 0.99,
        device: str = 'cpu'
    ):
        self.device = device
        self.gamma = gamma
        
        # Networks
        self.actor = Actor(state_dim, hidden_dim).to(device)
        self.critic = Critic(state_dim, hidden_dim).to(device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # Training history
        self.saved_log_probs = []
        self.saved_values = []
        self.rewards = []
    
    def select_action(
        self,
        state_embedding: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        temperature: float = 1.0
    ) -> int:
        """
        Select action using actor network.
        
        Args:
            state_embedding: User state from SASRec
            candidate_embeddings: Embeddings of candidate items
            temperature: Exploration temperature
            
        Returns:
            action_idx: Index in candidate list
        """
        state_embedding = state_embedding.to(self.device)
        candidate_embeddings = candidate_embeddings.to(self.device)
        
        # Get value estimate
        value = self.critic(state_embedding.unsqueeze(0))
        
        # Select action
        action_idx, log_prob = self.actor.select_action(
            state_embedding,
            candidate_embeddings,
            temperature
        )
        
        # Store for training
        self.saved_log_probs.append(log_prob)
        self.saved_values.append(value)
        
        return action_idx
    
    def store_reward(self, reward: float):
        """Store reward for current step."""
        self.rewards.append(reward)
    
    def update(self) -> Tuple[float, float]:
        """
        Update actor and critic networks using collected experience.
        
        Returns:
            actor_loss: Actor loss value
            critic_loss: Critic loss value
        """
        if len(self.rewards) == 0:
            return 0.0, 0.0
        
        # Compute returns
        returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float, device=self.device)
        
        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Compute losses
        actor_loss = 0
        critic_loss = 0
        
        for log_prob, value, R in zip(self.saved_log_probs, self.saved_values, returns):
            advantage = R - value.item()
            actor_loss -= log_prob * advantage
            critic_loss += F.mse_loss(value.squeeze(), R.unsqueeze(0))
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Clear buffers
        self.saved_log_probs = []
        self.saved_values = []
        self.rewards = []
        
        return actor_loss.item(), critic_loss.item()
    
    def save(self, path: str):
        """Save agent weights."""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
        }, path)
    
    def load(self, path: str):
        """Load agent weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
