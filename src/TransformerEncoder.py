import torch
import torch.nn as nn

from MultiHeadAttention import MultiHeadAttention


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder block for the Vision Transformer. See also:
    https://arxiv.org/pdf/2010.11929
    https://arxiv.org/pdf/1706.03762
    """

    def __init__(self,
                 num_heads: int,
                 embedding_dim: int,
                 mlp_hidden_dim: int,
                 mhsa_dropout: float,
                 mlp_dropout: float) -> None:
        """
        Initialize the Transformer Encoder block.

        Parameters
        ----------
        num_heads : int
            Number of attention heads.
        embedding_dim : int
            Dimension of the embedding space.
        mlp_hidden_dim : int
            Dimension of the hidden layer in the multi layer perceptron.
        mhsa_dropout : float
            Dropout rate for the multi-head self-attention.
        mlp_dropout : float
            Dropout rate for the multi layer perceptron.
        """
        super().__init__()
        self.embedding_dim = embedding_dim

        # Layer normalization before the multi head attention
        self.layer_normalization_1 = nn.LayerNorm(embedding_dim)

        # Multi head attention
        self.multi_head_self_attention = MultiHeadAttention(
            num_heads=num_heads, embedding_dim=embedding_dim, dropout=mhsa_dropout)

        # Layer norm after the first residual connection
        self.layer_normalization_2 = nn.LayerNorm(embedding_dim)

        # MLP as described in https://arxiv.org/pdf/2010.11929
        self.mlp_in = nn.Linear(embedding_dim, mlp_hidden_dim)
        self.mlp_dropout = nn.Dropout(mlp_dropout)
        self.gelu = nn.GELU()
        self.mlp_out = nn.Linear(mlp_hidden_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the transformer encoder block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_length, embedding_dim)

        Returns
        -------
        torch.Tensor
            Output tensor after forward pass. Tensor is of the same shape as the input tensor.
        """
        # Multi head self attention with residual connection and layer normalization
        x = x + self.multi_head_self_attention(self.layer_normalization_1(x))

        # Multi layer perceptron with residual connection and layer normalization
        x = x + self.mlp_out(self.mlp_dropout(
            self.gelu(self.mlp_in(self.layer_normalization_2(x)))))

        return x
