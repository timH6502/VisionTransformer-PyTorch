import torch
import torch.nn as nn

from TransformerEncoder import TransformerEncoder


class ViT(nn.Module):
    """
    Vision Transformer (ViT) for image classification as described in https://arxiv.org/abs/2010.11929
    """

    def __init__(self,
                 image_dims: tuple[int, int, int],
                 num_patches: int,
                 embedding_dim: int,
                 num_heads: int,
                 output_dim: int,
                 mlp_hidden_dim: int = 1024,
                 num_blocks: int = 16,
                 mhsa_dropout: float = 0.0,
                 mlp_dropout: float = 0.0) -> None:
        """
        Initialize the Vision Transformer.

        Parameters
        ----------
        image_dims : tuple[int, int, int]
            Image dimensions (channels, height, width).
        num_patches : int
            Number of patches in each direction.
        embedding_dim : int
            Dimension of the embedding space.
        num_heads : int
            Number of attention heads.
        output_dim : int
            Output dimension (typically number of classes).
        mlp_hidden_dim : int, default=1024
            Dimension of the hidden layer in the MLP.
        num_blocks : int, default=16
            Number of TransformerEncoder blocks.
        mhsa_dropout : float, default=0.0
            Dropout rate for the multi-head self-attention.
        mlp_dropout : float, defualt=0.0
            Dropout rate for the MLP in the TransformerEncoder.
        """
        super().__init__()
        self.num_patches = num_patches
        self.embedding_dim = embedding_dim

        # Ensure that image dimensions are divisible by the number of patches
        assert image_dims[1] % num_patches == 0 and image_dims[2] % num_patches == 0, \
            "Image height and width must be divisible by the number of patches."

        # Calculate patch size
        self.patch_size = (
            image_dims[1] // num_patches, image_dims[2] // num_patches)

        # Calculate input dimension for linear embedding layer
        self.input_dim = int(
            image_dims[0] * self.patch_size[0] * self.patch_size[1])

        # Linear layer for embedding the patches
        self.linear_embedding = nn.Linear(self.input_dim, self.embedding_dim)

        # Extra learnable [class] embedding
        self.extra_class_embedding = nn.Parameter(
            torch.rand(1, self.embedding_dim))

        # Learnable positional embedding
        self.positional_embedding = nn.Parameter(
            torch.rand(1, num_patches ** 2 + 1, embedding_dim))

        # List of TransformerEncoder blocks
        self.blocks = nn.ModuleList(
            [TransformerEncoder(num_heads, embedding_dim, mlp_hidden_dim,
                                mhsa_dropout=mhsa_dropout, mlp_dropout=mlp_dropout) for _ in range(num_blocks)])

        # Classification head
        self.classification_head = nn.Linear(embedding_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Vision Transformer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, height, width).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, output_dim)
        """
        # Create patches
        patches = self.create_patches(x, self.num_patches)
        # Linear Projection of Flattened Patches
        tokens = self.linear_embedding(patches)
        # Extra learnable [class] embedding
        batch_size = tokens.size(0)
        class_tokens = self.extra_class_embedding.expand(batch_size, -1, -1)
        tokens = torch.cat((class_tokens, tokens), dim=1)
        # Positional embedding
        positional_embedding = self.positional_embedding.expand(
            batch_size, -1, -1)
        out = tokens + positional_embedding
        # Transformer Blocks
        for block in self.blocks:
            out = block(out)
        # Classification Head
        out = self.classification_head(out[:, 0])

        return out

    def create_patches(self, images: torch.Tensor, n_patches: int) -> torch.Tensor:
        """
        Create patches from the input images.

        Parameters
        ----------
        images : torch.Tensor
            Images to be patchified.
        n_patches : int
            Number of patches in each direction.

        Returns
        -------
        torch.Tensor
            Tensor of patches. The tensor has shape
            (batch_size, num_patches ** 2, patch_size_height * patch_size_width * channels).
        """
        patch_height, patch_width = self.patch_size
        unfold = nn.Unfold(kernel_size=(patch_height, patch_width),
                           stride=(patch_height, patch_width))
        patches = unfold(images).transpose(1, 2)
        return patches
