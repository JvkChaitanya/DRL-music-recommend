"""
Training script for RL agent using Item-CF instead of SASRec.
Uses Item-CF similarity for candidate generation and state representation.
"""
import torch
import numpy as np
from pathlib import Path
import yaml
from tqdm import tqdm
import sys
import pickle

sys.path.insert(0, str(Path(__file__).parent))

from data.dataset import load_vocab
from models.environment import MusicRecommendationEnv
from models.rl_agent import ActorCriticAgent


def load_config():
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    quiet = config['data'].get('quiet_mode', True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load vocabulary
    vocab = load_vocab(config['data']['processed_path'])
    num_items = len(vocab)
    print(f"Vocabulary size: {num_items:,}")
    
    # Load trained Item-CF model
    print("Loading Item-CF model...")
    checkpoint_dir = Path(__file__).parent.parent / "checkpoints"
    item_cf_path = checkpoint_dir / "item_cf.pkl"
    
    if not item_cf_path.exists():
        print("ERROR: Item-CF model not found")
        print(f"Expected path: {item_cf_path}")
        print("Please run train_itemcf_only.py first to train Item-CF")
        return
    
    with open(item_cf_path, 'rb') as f:
        item_cf = pickle.load(f)
    
    print("✓ Item-CF loaded successfully!")
    print(f"  K-neighbors: {item_cf.k_neighbors}")
    print(f"  Similarity matrix shape: {item_cf.item_similarity.shape}")
    
    # Create environment
    print("\nCreating environment...")
    env = MusicRecommendationEnv(
        data_path=str(Path(config['data']['processed_path']) / 'train.pkl'),
        vocab=vocab,
        max_seq_length=config['sasrec']['max_seq_length'],
        device=str(device)
    )
    
    # Create RL agent
    # Use lower-dimensional state since we don't have SASRec embeddings
    state_dim = 128  # Compressed representation from Item-CF
    agent = ActorCriticAgent(
        state_dim=state_dim,
        hidden_dim=128,
        actor_lr=config['rl']['actor_lr'],
        critic_lr=config['rl']['critic_lr'],
        gamma=config['rl']['gamma'],
        device=str(device)
    )
    
    print(f"Agent state dimension: {state_dim}")
    
    # Training loop
    print("\n" + "="*60)
    print("RL TRAINING WITH ITEM-CF")
    print("="*60)
    
    num_episodes = config['rl'].get('episodes', 10000)
    eval_every = 500
    save_dir = Path(__file__).parent.parent / "checkpoints"
    
    total_rewards = []
    total_hits = 0
    
    pbar = tqdm(range(num_episodes), desc="RL Training", disable=quiet, ncols=80)
    for episode in pbar:
        # Reset environment
        state_seq = env.reset()
        
        # Convert sequence to item indices (on CPU for Item-CF)
        sequence_items = state_seq.cpu().numpy().tolist()
        sequence_items = [int(item) for item in sequence_items if item > 0]  # Filter padding
        
        # ===== IMPROVED CANDIDATE SELECTION =====
        # Hybrid approach: Item-CF + User History + Exploration + Ground Truth
        ground_truth = env.data[env.current_idx]['target']
        user_history = list(env.user_histories.get(env.current_user, set()))
        
        candidates_set = set()
        
        if len(sequence_items) > 0:
            # Get Item-CF predictions
            item_scores = item_cf.predict_scores(sequence_items, num_items)
            
            # Filter out items already in sequence
            for item in sequence_items:
                if item < num_items:
                    item_scores[item] = -np.inf
            
            # Top 50 from Item-CF
            top_itemcf = np.argsort(item_scores)[-50:][::-1]
            candidates_set.update(top_itemcf[top_itemcf > 0])
        
        # Add top 30 from user history (excluding current sequence)
        history_candidates = [item for item in user_history 
                            if item not in sequence_items and item > 0][:30]
        candidates_set.update(history_candidates)
        
        # Add 15 random exploration candidates
        available_items = set(range(1, min(num_items, 10000))) - set(sequence_items)
        if len(available_items) > 15:
            exploration = np.random.choice(list(available_items), 15, replace=False)
            candidates_set.update(exploration)
        
        # IMPORTANT: Include ground truth for learning (teacher forcing during training)
        if ground_truth > 0:
            candidates_set.add(ground_truth)
        
        # Convert to list and ensure we have exactly 100 candidates
        top_candidates_list = list(candidates_set)
        if len(top_candidates_list) < 100:
            # Pad with random items
            needed = 100 - len(top_candidates_list)
            available = list(set(range(1, min(num_items, 10000))) - candidates_set - set(sequence_items))
            if len(available) >= needed:
                padding = np.random.choice(available, needed, replace=False)
                top_candidates_list.extend(padding)
        elif len(top_candidates_list) > 100:
            # Trim to 100
            top_candidates_list = top_candidates_list[:100]
        
        top_candidates = torch.tensor(top_candidates_list, dtype=torch.long, device=device)
        
        # ===== IMPROVED STATE EMBEDDING =====
        # Deterministic aggregation using top-K similarities with recency weighting
        state_embedding = torch.zeros(state_dim, device=device)
        if len(sequence_items) > 0:
            similarity_size = item_cf.item_similarity.shape[0]
            
            # Use last 10 items with recency weighting
            recent_items = sequence_items[-10:]
            weights = np.exp(np.linspace(-1, 0, len(recent_items)))  # Exponential recency
            weights = weights / weights.sum()
            
            valid_embeddings = []
            valid_weights = []
            
            for idx, item in enumerate(recent_items):
                if 0 < item < similarity_size:
                    sim_vec = item_cf.item_similarity[item].toarray().flatten()
                    
                    # Deterministic: Take top-K most similar items
                    if len(sim_vec) >= state_dim:
                        top_k_indices = np.argpartition(sim_vec, -state_dim)[-state_dim:]
                        embedding = sim_vec[top_k_indices]
                    else:
                        # Pad if needed
                        embedding = np.zeros(state_dim)
                        embedding[:len(sim_vec)] = sim_vec
                    
                    valid_embeddings.append(embedding)
                    valid_weights.append(weights[idx])
            
            if valid_embeddings:
                # Weighted average based on recency
                valid_weights = np.array(valid_weights)
                valid_weights = valid_weights / valid_weights.sum()
                weighted_emb = np.average(valid_embeddings, axis=0, weights=valid_weights)
                state_embedding = torch.tensor(weighted_emb, dtype=torch.float32, device=device)
        
        # ===== IMPROVED CANDIDATE EMBEDDINGS =====
        similarity_size = item_cf.item_similarity.shape[0]
        candidate_embeddings = torch.zeros(len(top_candidates), state_dim, device=device)
        
        for i, cand_item in enumerate(top_candidates):
            cand_item = int(cand_item.item())
            if 0 < cand_item < similarity_size:
                sim_vec = item_cf.item_similarity[cand_item].toarray().flatten()
                
                # Deterministic: Take top-K most similar items
                if len(sim_vec) >= state_dim:
                    top_k_indices = np.argpartition(sim_vec, -state_dim)[-state_dim:]
                    embedding = sim_vec[top_k_indices]
                else:
                    embedding = np.zeros(state_dim)
                    embedding[:len(sim_vec)] = sim_vec
                
                candidate_embeddings[i] = torch.tensor(embedding, dtype=torch.float32, device=device)
        
        # Select action
        action_idx = agent.select_action(
            state_embedding,
            candidate_embeddings,
            temperature=max(0.5, 1.0 - episode / num_episodes)
        )
        action = int(top_candidates[action_idx].item())
        
        # Take action in environment
        next_state_seq, reward, done, info = env.step(action)
        
        # Store reward
        agent.store_reward(reward)
        total_rewards.append(reward)
        
        if info['hit']:
            total_hits += 1
        
        # Update agent more frequently (every 5 episodes or when done)
        if done:
            actor_loss, critic_loss = agent.update()
        elif (episode + 1) % 5 == 0:  # Batch update every 5 episodes
            if len(agent.rewards) > 0:
                actor_loss, critic_loss = agent.update()
        
        # Enhanced logging with more frequent updates
        if (episode + 1) % 100 == 0:  # Log every 100 episodes
            recent_rewards = total_rewards[-100:]
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            hit_rate = total_hits / (episode + 1)
            
            if not quiet:
                print(f"Episode {episode+1}/{num_episodes}")
                print(f"  Avg Reward (last 100): {avg_reward:.4f}")
                print(f"  Hit Rate: {hit_rate:.4f}")
                print(f"  Total Hits: {total_hits}")
        
        # Update progress bar
        if (episode + 1) % eval_every == 0:
            avg_reward = sum(total_rewards[-eval_every:]) / eval_every
            hit_rate = total_hits / (episode + 1)
            pbar.set_postfix({
                'avg_reward': f'{avg_reward:.3f}',
                'hit_rate': f'{hit_rate:.3f}'
            })
    
    # Save agent
    agent.save(str(save_dir / "rl_agent_itemcf.pt"))
    
    print("\n" + "="*60)
    print("RL TRAINING COMPLETE!")
    print("="*60)
    print(f"Final hit rate: {total_hits / num_episodes:.4f}")
    print(f"Average reward: {sum(total_rewards) / len(total_rewards):.4f}")
    print(f"\nModel saved to: {save_dir / 'rl_agent_itemcf.pt'}")


if __name__ == "__main__":
    main()
