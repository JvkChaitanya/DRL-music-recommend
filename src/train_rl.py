"""
Training script for RL agent (Phase 2).
Uses trained SASRec for state representation and candidate generation.
"""
import torch
from pathlib import Path
import yaml
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent))

from data.dataset import load_vocab
from models.sasrec import SASRec
from models.environment import MusicRecommendationEnv
from models.rl_agent import ActorCriticAgent


def load_config():
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load vocabulary
    vocab = load_vocab(config['data']['processed_path'])
    num_items = len(vocab)
    print(f"Vocabulary size: {num_items:,}")
    
    # Load trained SASRec
    print("Loading SASRec model...")
    sasrec = SASRec(
        num_items=num_items,
        embedding_dim=config['sasrec']['embedding_dim'],
        num_heads=config['sasrec']['num_heads'],
        num_layers=config['sasrec']['num_layers'],
        max_seq_length=config['sasrec']['max_seq_length'],
        dropout=config['sasrec']['dropout']
    ).to(device)
    
    checkpoint_path = Path(__file__).parent.parent / "checkpoints" / "sasrec_best.pt"
    if not checkpoint_path.exists():
        print("ERROR: SASRec model not found. Please run train_sasrec.py first.")
        return
    
    sasrec.load_state_dict(torch.load(checkpoint_path, map_location=device))
    sasrec.eval()
    print("SASRec loaded successfully!")
    
    # Create environment
    print("Creating environment...")
    env = MusicRecommendationEnv(
        data_path=str(Path(config['data']['processed_path']) / 'train.pkl'),
        vocab=vocab,
        max_seq_length=config['sasrec']['max_seq_length'],
        device=str(device)
    )
    
    # Create RL agent
    agent = ActorCriticAgent(
        state_dim=config['sasrec']['embedding_dim'],
        hidden_dim=128,
        actor_lr=config['rl']['actor_lr'],
        critic_lr=config['rl']['critic_lr'],
        gamma=config['rl']['gamma'],
        device=str(device)
    )
    
    # Training loop
    print("\nStarting RL training...")
    num_episodes = config['rl'].get('episodes', 10000)
    eval_every = 500
    save_dir = Path(__file__).parent.parent / "checkpoints"
    
    total_rewards = []
    total_hits = 0
    
    pbar = tqdm(range(num_episodes), desc="RL Training")
    for episode in pbar:
        # Reset environment
        state_seq = env.reset()
        
        with torch.no_grad():
            # Get state embedding from SASRec
            state_embedding = sasrec.get_embedding(state_seq.unsqueeze(0)).squeeze(0)
            
            # Get candidate items using SASRec predictions
            logits = sasrec(state_seq.unsqueeze(0)).squeeze(0)
            _, top_candidates = logits.topk(100)
            
            # Get candidate embeddings
            candidate_embeddings = sasrec.item_embedding(top_candidates)
        
        # Select action
        action_idx = agent.select_action(
            state_embedding,
            candidate_embeddings,
            temperature=max(0.5, 1.0 - episode / num_episodes)  # Anneal temperature
        )
        action = top_candidates[action_idx].item()
        
        # Take action in environment
        next_state_seq, reward, done, info = env.step(action)
        
        # Store reward
        agent.store_reward(reward)
        total_rewards.append(reward)
        
        if info['hit']:
            total_hits += 1
        
        # Update agent
        if done:
            actor_loss, critic_loss = agent.update()
        
        # Logging
        if (episode + 1) % eval_every == 0:
            avg_reward = sum(total_rewards[-eval_every:]) / eval_every
            hit_rate = total_hits / (episode + 1)
            pbar.set_postfix({
                'avg_reward': f'{avg_reward:.3f}',
                'hit_rate': f'{hit_rate:.3f}'
            })
    
    # Save agent
    agent.save(str(save_dir / "rl_agent.pt"))
    print(f"\nRL training complete!")
    print(f"Final hit rate: {total_hits / num_episodes:.4f}")
    print(f"Average reward: {sum(total_rewards) / len(total_rewards):.4f}")


if __name__ == "__main__":
    main()
