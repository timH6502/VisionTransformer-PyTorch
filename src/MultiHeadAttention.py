import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism as described in https://arxiv.org/pdf/1706.03762
    """

    def __init__(self, num_heads: int, embedding_dim: int, dropout: float = 0.0) -> None:
        """
        Initialize the Multi-Head Attention mechanism.

        Parameters
        ----------
        num_heads : int
            Number of attention heads.
        embedding_dim : int
            Dimension of the embedding space.
        dropout : float, default=0.0
            Dropout rate for regularization.
        """
        super().__init__()
        assert embedding_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads."
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.head_dim = embedding_dim // num_heads

        # Scaling factor for the attention scores (\sqrt{d_k})
        self.sqrt_d_k = self.head_dim ** 0.5

        # Linear layers for query, key and value projection (Q, K, V)
        self.q_projection = nn.Linear(embedding_dim, embedding_dim)
        self.k_projection = nn.Linear(embedding_dim, embedding_dim)
        self.v_projection = nn.Linear(embedding_dim, embedding_dim)

        # Linear layer for output projection
        self.out_projection = nn.Linear(embedding_dim, embedding_dim)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Multi-Head Attention mechanism.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_length, embedding_dim)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, seq_length, embedding_dim)
        """
        batch_size, seq_length, embedding_dim = x.size()

        # Project the input to query, key and value.
        Q = self.q_projection(x)
        K = self.k_projection(x)
        V = self.v_projection(x)

        # Reshape and permute the results. K is permuted differently, because the result is K^T
        Q = Q.view(batch_size, seq_length, self.num_heads,
                   self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, seq_length, self.num_heads,
                   self.head_dim).permute(0, 2, 3, 1)
        V = V.view(batch_size, seq_length, self.num_heads,
                   self.head_dim).permute(0, 2, 1, 3)

        # Equation (1) of the paper (Attention(Q, K, V) = softmax(\frac{Q K^T}{\sqrt{d_k}})V)
        attention_scores = torch.matmul(
            Q, K) / self.sqrt_d_k
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        attention_out = torch.matmul(attention_weights, V)
        attention_out = attention_out.permute(0, 2, 1, 3).contiguous().view(
            batch_size, seq_length, embedding_dim)

        # Project attention output
        out = self.out_projection(attention_out)

        return out
