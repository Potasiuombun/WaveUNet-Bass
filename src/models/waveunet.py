"""Residual Wave-U-Net baseline model."""
import torch
import torch.nn as nn
import torch.nn.init as init
from typing import Optional
import math

from .blocks import ConvBlock, DownBlock, UpBlock, SelfAttention1d


class ResidualWaveUNet(nn.Module):
    """Residual Wave-U-Net for waveform-domain loudness enhancement.
    
    Architecture:
    - Encoder: downsampling path with skip connections
    - Bottleneck: convolutional block at lowest resolution
    - Decoder: upsampling path with concatenated skip connections
    - Output: residual prediction (input + delta)
    
    The model predicts a residual delta that is added to the input to produce output.
    """
    
    def __init__(
        self,
        depth: int = 8,
        base_channels: int = 16,
        kernel_size: int = 9,
        activation: str = "leaky_relu",
        output_activation: Optional[str] = None,
        max_scale: Optional[float] = 0.5,
        use_bottleneck_attention: bool = False,
        attention_heads: int = 8,
        attention_dropout: float = 0.0,
        attention_layers: int = 1,
        attention_mlp_ratio: float = 2.0,
    ):
        """Initialize Wave-U-Net.
        
        Args:
            depth: Number of encoder/decoder layers.
            base_channels: Number of channels in first conv layer.
            kernel_size: Kernel size for all conv layers.
            activation: Activation function ("relu", "leaky_relu", "gelu").
            output_activation: Output activation ("relu", "tanh", None).
            max_scale: If using tanh, scale output to [-max_scale, max_scale].
                      If None, no scaling.
        """
        super().__init__()
        
        self.depth = depth
        self.base_channels = base_channels
        self.kernel_size = kernel_size
        self.activation = activation
        self.output_activation = output_activation
        self.max_scale = max_scale
        self.use_bottleneck_attention = bool(use_bottleneck_attention)
        self.attention_heads = int(attention_heads)
        self.attention_dropout = float(attention_dropout)
        self.attention_layers = int(attention_layers)
        self.attention_mlp_ratio = float(attention_mlp_ratio)
        
        # Build encoder
        self.encoder = nn.ModuleList()
        in_ch = 1
        for i in range(depth):
            out_ch = base_channels * (2 ** i)
            self.encoder.append(
                DownBlock(
                    in_ch, out_ch,
                    kernel_size=kernel_size,
                    activation=activation
                )
            )
            in_ch = out_ch
        
        # Bottleneck
        bottleneck_ch = base_channels * (2 ** depth)
        self.bottleneck = ConvBlock(
            in_ch, bottleneck_ch,
            kernel_size=3,
            activation=activation
        )

        self.bottleneck_attention = None
        if self.use_bottleneck_attention:
            if self.attention_layers < 1:
                raise ValueError("attention_layers must be >= 1 when attention is enabled")
            self.bottleneck_attention = nn.ModuleList(
                [
                    SelfAttention1d(
                        channels=bottleneck_ch,
                        num_heads=self.attention_heads,
                        dropout=self.attention_dropout,
                        mlp_ratio=self.attention_mlp_ratio,
                    )
                    for _ in range(self.attention_layers)
                ]
            )
        
        # Build decoder
        self.decoder = nn.ModuleList()
        for i in range(depth):
            in_ch = bottleneck_ch if i == 0 else base_channels * (2 ** (depth - i))
            out_ch = base_channels * (2 ** (depth - i - 1))
            skip_ch = out_ch  # Skip has same channels as output of that encoder layer
            
            self.decoder.append(
                UpBlock(
                    in_ch + skip_ch,  # Input channels after concatenation
                    out_ch,
                    kernel_size=kernel_size,
                    activation=activation
                )
            )
        
        # Output layer: predict residual (1 channel)
        self.output_conv = nn.Conv1d(base_channels + 1, 1, kernel_size=1)
        
        # Optional output activation
        if output_activation == "tanh":
            self.output_act = nn.Tanh()
        elif output_activation == "relu":
            self.output_act = nn.ReLU()
        else:
            self.output_act = None
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                init.ones_(m.weight)
                init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input waveform, shape [batch, 1, T].
            
        Returns:
            Output waveform, shape [batch, 1, T].
        """
        # Encoder with skip connections
        skips = []
        enc = x
        for i, down_block in enumerate(self.encoder):
            enc, skip = down_block(enc)
            skips.append(skip)
        
        # Bottleneck
        enc = self.bottleneck(enc)
        if self.bottleneck_attention is not None:
            for attn_block in self.bottleneck_attention:
                enc = attn_block(enc)
        
        # Decoder with skip connections
        for i, up_block in enumerate(self.decoder):
            skip = skips[self.depth - i - 1]
            enc = up_block(enc, skip)
        
        # Concatenate first layer output with input for final convolution
        enc = torch.cat([enc, x], dim=1)
        
        # Output: residual prediction
        delta = self.output_conv(enc)
        
        # Apply optional output activation
        if self.output_act is not None:
            delta = self.output_act(delta)
            if self.max_scale is not None and isinstance(self.output_act, nn.Tanh):
                delta = delta * self.max_scale
        
        # Residual connection: output = input + delta
        output = x + delta
        
        return output


def create_waveunet(
    depth: int = 8,
    base_channels: int = 16,
    kernel_size: int = 9,
    activation: str = "leaky_relu",
    output_activation: Optional[str] = None,
    max_scale: Optional[float] = None,
    use_bottleneck_attention: bool = False,
    attention_heads: int = 8,
    attention_dropout: float = 0.0,
    attention_layers: int = 1,
    attention_mlp_ratio: float = 2.0,
    device: str = "cpu"
) -> ResidualWaveUNet:
    """Create and return a Wave-U-Net model.
    
    Args:
        depth: Number of encoder/decoder layers.
        base_channels: Number of channels in first layer.
        kernel_size: Convolution kernel size.
        activation: Activation function.
        output_activation: Output activation.
        max_scale: Max scale for tanh output.
        device: Device to place model on.
        
    Returns:
        Model on specified device.
    """
    model = ResidualWaveUNet(
        depth=depth,
        base_channels=base_channels,
        kernel_size=kernel_size,
        activation=activation,
        output_activation=output_activation,
        max_scale=max_scale,
        use_bottleneck_attention=use_bottleneck_attention,
        attention_heads=attention_heads,
        attention_dropout=attention_dropout,
        attention_layers=attention_layers,
        attention_mlp_ratio=attention_mlp_ratio,
    )
    model = model.to(device)
    return model
