"""
Training script for SASRec model (Phase 1).
"""
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import yaml
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent))

from data.dataset import create_dataloaders, load_vocab
from models.sasrec import SASRec
from evaluate import evaluate_model


def load_config():
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        sequence = batch['sequence'].to(device)
        target = batch['target'].to(device)
        
        optimizer.zero_grad()
        logits = model(sequence)
        loss = criterion(logits, target)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({'loss': total_loss / num_batches})
    
    return total_loss / num_batches


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            sequence = batch['sequence'].to(device)
            target = batch['target'].to(device)
            
            logits = model(sequence)
            loss = criterion(logits, target)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def main():
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader = create_dataloaders(
        config['data']['processed_path'],
        batch_size=config['training']['batch_size'],
        max_seq_length=config['sasrec']['max_seq_length']
    )
    
    vocab = load_vocab(config['data']['processed_path'])
    num_items = len(vocab)
    print(f"Vocabulary size: {num_items:,}")
    
    # Create model
    model = SASRec(
        num_items=num_items,
        embedding_dim=config['sasrec']['embedding_dim'],
        num_heads=config['sasrec']['num_heads'],
        num_layers=config['sasrec']['num_layers'],
        max_seq_length=config['sasrec']['max_seq_length'],
        dropout=config['sasrec']['dropout']
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate']
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    save_dir = Path(__file__).parent.parent / "checkpoints"
    save_dir.mkdir(exist_ok=True)
    
    print("\nStarting training...")
    for epoch in range(config['training']['sasrec_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['training']['sasrec_epochs']}")
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_dir / "sasrec_best.pt")
            print("Saved best model!")
        else:
            patience_counter += 1
            if patience_counter >= config['training']['early_stopping_patience']:
                print("Early stopping triggered!")
                break
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(save_dir / "sasrec_best.pt"))
    metrics = evaluate_model(model, test_loader, device, config['eval']['top_k'])
    
    print("\n=== Test Results ===")
    for k in config['eval']['top_k']:
        print(f"Hit@{k}: {metrics[f'hit@{k}']:.4f} | NDCG@{k}: {metrics[f'ndcg@{k}']:.4f}")
    print(f"MRR: {metrics['mrr']:.4f}")


if __name__ == "__main__":
    main()
