"""
SASRec: Self-Attentive Sequential Recommendation
Implements content-aware sequential recommendation using self-attention.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:x.size(1)]


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with causal masking."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        # Linear projections
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Causal mask (prevent attending to future positions)
        if mask is None:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            mask = mask.unsqueeze(0).unsqueeze(0)
        
        scores = scores.masked_fill(mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.W_o(context)


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(self, d_model: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class SASRecBlock(nn.Module):
    """Single SASRec transformer block."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_out = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_out)
        
        # FFN with residual
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_out)
        
        return x


class SASRec(nn.Module):
    """
    Self-Attentive Sequential Recommendation model.
    
    Args:
        num_items: Size of item vocabulary
        embedding_dim: Dimension of embeddings
        num_heads: Number of attention heads
        num_layers: Number of transformer blocks
        max_seq_length: Maximum sequence length
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 64,
        num_heads: int = 2,
        num_layers: int = 2,
        max_seq_length: int = 50,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        
        # Embeddings
        self.item_embedding = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        self.positional_encoding = PositionalEncoding(embedding_dim, max_seq_length)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            SASRecBlock(embedding_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Output projection
        self.output_layer = nn.Linear(embedding_dim, num_items)
        
        # Share weights with embedding
        self.output_layer.weight = self.item_embedding.weight
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            sequence: (batch, seq_len) tensor of item indices
            
        Returns:
            logits: (batch, num_items) prediction scores for next item
        """
        # Create padding mask
        padding_mask = (sequence == 0)
        
        # Get embeddings
        x = self.item_embedding(sequence)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # Causal mask
        seq_len = sequence.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=sequence.device), 
            diagonal=1
        ).bool()
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, causal_mask.unsqueeze(0).unsqueeze(0))
        
        x = self.norm(x)
        
        # Get last position output (for next item prediction)
        # Find the last non-padding position for each sequence
        seq_lengths = (sequence != 0).sum(dim=1) - 1
        batch_size = sequence.size(0)
        last_hidden = x[torch.arange(batch_size, device=x.device), seq_lengths]
        
        # Project to item space
        logits = self.output_layer(last_hidden)
        
        return logits
    
    def get_embedding(self, sequence: torch.Tensor) -> torch.Tensor:
        """Get the hidden state embedding for a sequence (for RL state)."""
        padding_mask = (sequence == 0)
        
        x = self.item_embedding(sequence)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        seq_len = sequence.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=sequence.device), 
            diagonal=1
        ).bool()
        
        for block in self.blocks:
            x = block(x, causal_mask.unsqueeze(0).unsqueeze(0))
        
        x = self.norm(x)
        
        # Get last position
        seq_lengths = (sequence != 0).sum(dim=1) - 1
        batch_size = sequence.size(0)
        last_hidden = x[torch.arange(batch_size, device=x.device), seq_lengths]
        
        return last_hidden
