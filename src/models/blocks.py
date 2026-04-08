"""Neural network building blocks for audio models."""
import torch
import torch.nn as nn
from typing import Optional, Tuple


class ConvBlock(nn.Module):
    """Convolutional block: Conv1d -> BatchNorm -> Activation."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 9,
        stride: int = 1,
        padding: str = "same",
        activation: str = "leaky_relu"
    ):
        """Initialize conv block.
        
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Kernel size of conv layer.
            stride: Stride of conv layer.
            padding: Padding type ("same" or int).
            activation: Activation function ("relu", "leaky_relu", "gelu", etc).
        """
        super().__init__()
        
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.bn = nn.BatchNorm1d(out_channels)
        
        # Activation
        if activation == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation == "leaky_relu":
            self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif activation == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class DownBlock(nn.Module):
    """Downsampling block: Conv -> Pooling."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 9,
        pool_size: int = 2,
        activation: str = "leaky_relu"
    ):
        """Initialize downsampling block.
        
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Kernel size of conv layer.
            pool_size: Pooling kernel size.
            activation: Activation function.
        """
        super().__init__()
        self.conv = ConvBlock(
            in_channels, out_channels,
            kernel_size=kernel_size,
            activation=activation
        )
        self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Returns:
            (pooled_output, skip_connection)
        """
        x = self.conv(x)
        skip = x
        x = self.pool(x)
        return x, skip


class UpBlock(nn.Module):
    """Upsampling block: Upsample -> Conv."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 9,
        scale_factor: int = 2,
        activation: str = "leaky_relu"
    ):
        """Initialize upsampling block.
        
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Kernel size of conv layer.
            scale_factor: Upsampling scale factor.
            activation: Activation function.
        """
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode="linear", align_corners=True)
        self.conv = ConvBlock(
            in_channels, out_channels,
            kernel_size=kernel_size,
            activation=activation
        )
    
    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Main input.
            skip: Optional skip connection to concatenate.
            
        Returns:
            Output tensor.
        """
        x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class SelfAttention1d(nn.Module):
    """Transformer-style self-attention block over temporal dimension."""

    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        mlp_ratio: float = 2.0,
    ):
        """Initialize 1D self-attention block.

        Args:
            channels: Embedding dimension (channel count).
            num_heads: Number of attention heads.
            dropout: Dropout probability.
            mlp_ratio: Expansion ratio for feed-forward hidden size.
        """
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads for SelfAttention1d")
        if mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be > 0")

        hidden_dim = int(channels * mlp_ratio)

        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, channels),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply MHSA + FFN residual block.

        Args:
            x: Tensor of shape [batch, channels, time].

        Returns:
            Tensor of shape [batch, channels, time].
        """
        x_t = x.transpose(1, 2)  # [B, T, C]

        x_norm = self.norm1(x_t)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x_t = x_t + self.drop1(attn_out)

        x_ffn = self.ffn(self.norm2(x_t))
        x_t = x_t + self.drop2(x_ffn)
        return x_t.transpose(1, 2)

