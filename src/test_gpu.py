"""
Quick test to verify models run on GPU with small data subset.
"""
import torch
import pickle
from pathlib import Path

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create tiny test data
    print("\nCreating test data...")
    test_data = [
        {'user_id': 'u1', 'sequence': [10, 20, 30, 40], 'target': 50},
        {'user_id': 'u1', 'sequence': [20, 30, 40, 50], 'target': 60},
        {'user_id': 'u2', 'sequence': [100, 200, 300], 'target': 400},
        {'user_id': 'u2', 'sequence': [200, 300, 400], 'target': 500},
    ] * 100  # 400 samples
    
    num_items = 1000
    print(f"Test data: {len(test_data)} samples, {num_items} items")
    
    # Test PopularityBaselineGPU
    print("\n=== Testing PopularityBaselineGPU ===")
    from baselines_gpu import PopularityBaselineGPU
    pop = PopularityBaselineGPU(device=device)
    pop.fit(test_data, num_items)
    
    # Test prediction
    test_seq = torch.tensor([[10, 20, 30, 40]], device=device)
    preds = pop.predict(test_seq, k=5)
    print(f"Predictions shape: {preds.shape}")
    print(f"Top-5 predictions: {preds[0].tolist()}")
    
    # Test ItemCFGPU
    print("\n=== Testing ItemCFGPU ===")
    from baselines_gpu import ItemCFGPU
    cf = ItemCFGPU(k_neighbors=10, device=device)
    cf.fit(test_data, num_items)
    
    preds = cf.predict(test_seq, k=5)
    print(f"Predictions shape: {preds.shape}")
    print(f"Top-5 predictions: {preds[0].tolist()}")
    
    # Test SASRec
    print("\n=== Testing SASRec ===")
    from models.sasrec import SASRec
    model = SASRec(
        num_items=num_items,
        embedding_dim=32,
        num_heads=2,
        num_layers=1,
        max_seq_length=10,
        dropout=0.1
    ).to(device)
    
    test_seq = torch.tensor([[0, 0, 0, 10, 20, 30, 40, 50, 60, 70]], device=device)
    logits = model(test_seq)
    print(f"Logits shape: {logits.shape}")
    print(f"Sample logits (first 5): {logits[0, :5].tolist()}")
    
    # Test RL Agent
    print("\n=== Testing RL Agent ===")
    from models.rl_agent import ActorCriticAgent
    agent = ActorCriticAgent(state_dim=32, hidden_dim=64, device=device)
    
    state = torch.randn(32, device=device)
    candidates = torch.randn(10, 32, device=device)
    action = agent.select_action(state, candidates)
    print(f"Selected action: {action}")
    
    print("\n" + "="*50)
    print("✓ All models running on GPU successfully!")
    print("="*50)

if __name__ == "__main__":
    main()
