"""
Evaluation metrics for sequential recommendation.
"""
import torch
import numpy as np
from typing import Dict, List
from tqdm import tqdm


def hit_at_k(predictions: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    """
    Compute Hit@K metric.
    
    Args:
        predictions: (batch, num_items) prediction scores
        targets: (batch,) ground truth item indices
        k: Number of top items to consider
        
    Returns:
        hit_rate: Proportion of samples where target is in top-K
    """
    _, topk_indices = predictions.topk(k, dim=1)
    hits = (topk_indices == targets.unsqueeze(1)).any(dim=1)
    return hits.float().mean().item()


def ndcg_at_k(predictions: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    """
    Compute NDCG@K metric.
    
    Args:
        predictions: (batch, num_items) prediction scores
        targets: (batch,) ground truth item indices
        k: Number of top items to consider
        
    Returns:
        ndcg: Normalized Discounted Cumulative Gain
    """
    _, topk_indices = predictions.topk(k, dim=1)
    
    # Find position of target in top-K (1-indexed)
    matches = (topk_indices == targets.unsqueeze(1))
    
    # DCG = 1/log2(rank+1) if hit, else 0
    positions = torch.arange(1, k + 1, device=predictions.device).float()
    dcg = (matches.float() / torch.log2(positions + 1)).sum(dim=1)
    
    # IDCG = 1 (single relevant item at rank 1)
    idcg = 1.0
    
    return (dcg / idcg).mean().item()


def mrr(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute Mean Reciprocal Rank.
    
    Args:
        predictions: (batch, num_items) prediction scores
        targets: (batch,) ground truth item indices
        
    Returns:
        mrr: Mean Reciprocal Rank
    """
    # Get ranks of all items
    sorted_indices = predictions.argsort(dim=1, descending=True)
    
    # Find rank of target
    ranks = (sorted_indices == targets.unsqueeze(1)).nonzero(as_tuple=True)[1] + 1
    
    return (1.0 / ranks.float()).mean().item()


def evaluate_model(
    model,
    dataloader,
    device: str,
    top_k_values: List[int] = [5, 10, 20]
) -> Dict[str, float]:
    """
    Evaluate model on all metrics.
    
    Args:
        model: Trained SASRec model
        dataloader: Test dataloader
        device: Device to run on
        top_k_values: List of K values for Hit@K and NDCG@K
        
    Returns:
        metrics: Dictionary of metric names to values
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            sequence = batch['sequence'].to(device)
            target = batch['target'].to(device)
            
            logits = model(sequence)
            
            all_predictions.append(logits.cpu())
            all_targets.append(target.cpu())
    
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    metrics = {}
    
    for k in top_k_values:
        metrics[f'hit@{k}'] = hit_at_k(predictions, targets, k)
        metrics[f'ndcg@{k}'] = ndcg_at_k(predictions, targets, k)
    
    metrics['mrr'] = mrr(predictions, targets)
    
    return metrics


def compute_diversity(predictions: torch.Tensor, k: int = 10) -> float:
    """
    Compute recommendation diversity (unique items ratio in top-K).
    
    Args:
        predictions: (batch, num_items) prediction scores
        k: Number of top items
        
    Returns:
        diversity: Ratio of unique items to total recommendations
    """
    _, topk_indices = predictions.topk(k, dim=1)
    unique_items = topk_indices.unique()
    total_recs = topk_indices.numel()
    return len(unique_items) / total_recs


def compute_coverage(predictions: torch.Tensor, num_items: int, k: int = 10) -> float:
    """
    Compute catalog coverage (proportion of items ever recommended).
    
    Args:
        predictions: (batch, num_items) prediction scores
        num_items: Total number of items
        k: Number of top items
        
    Returns:
        coverage: Proportion of catalog covered
    """
    _, topk_indices = predictions.topk(k, dim=1)
    unique_items = topk_indices.unique()
    return len(unique_items) / num_items
