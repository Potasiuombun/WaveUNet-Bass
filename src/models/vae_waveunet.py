"""Variational Wave-U-Net model for Stage-3 training."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.init as init

from .blocks import ConvBlock, DownBlock, UpBlock, SelfAttention1d


class VariationalWaveUNet(nn.Module):
    """Wave-U-Net with a variational bottleneck.

    The model encodes a waveform into a latent bottleneck, samples via the
    reparameterization trick, and decodes back to waveform space.
    """

    def __init__(
        self,
        depth: int = 5,
        base_channels: int = 32,
        kernel_size: int = 9,
        activation: str = "leaky_relu",
        output_activation: Optional[str] = None,
        max_scale: Optional[float] = 0.5,
        latent_channels: int = 64,
        use_bottleneck_attention: bool = False,
        attention_heads: int = 8,
        attention_dropout: float = 0.0,
        attention_layers: int = 1,
        attention_mlp_ratio: float = 2.0,
    ) -> None:
        """Initialize variational Wave-U-Net.

        Args:
            depth: Number of encoder and decoder scales.
            base_channels: Channel count at first encoder block.
            kernel_size: Kernel size for most conv layers.
            activation: Activation name for blocks.
            output_activation: Optional output activation.
            max_scale: Optional scaling for tanh output.
            latent_channels: Channel count in latent bottleneck.
        """
        super().__init__()
        self.depth = int(depth)
        self.base_channels = int(base_channels)
        self.kernel_size = int(kernel_size)
        self.activation = str(activation)
        self.output_activation = output_activation
        self.max_scale = max_scale
        self.latent_channels = int(latent_channels)
        self.use_bottleneck_attention = bool(use_bottleneck_attention)
        self.attention_heads = int(attention_heads)
        self.attention_dropout = float(attention_dropout)
        self.attention_layers = int(attention_layers)
        self.attention_mlp_ratio = float(attention_mlp_ratio)

        self.encoder = nn.ModuleList()
        in_ch = 1
        for idx in range(self.depth):
            out_ch = self.base_channels * (2**idx)
            self.encoder.append(
                DownBlock(
                    in_ch,
                    out_ch,
                    kernel_size=self.kernel_size,
                    activation=self.activation,
                )
            )
            in_ch = out_ch

        bottleneck_ch = self.base_channels * (2**self.depth)
        self.bottleneck = ConvBlock(
            in_ch,
            bottleneck_ch,
            kernel_size=3,
            activation=self.activation,
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

        self.mu_head = nn.Conv1d(bottleneck_ch, self.latent_channels, kernel_size=1)
        self.logvar_head = nn.Conv1d(bottleneck_ch, self.latent_channels, kernel_size=1)
        self.latent_to_decoder = nn.Conv1d(self.latent_channels, bottleneck_ch, kernel_size=1)

        self.decoder = nn.ModuleList()
        for i in range(self.depth):
            dec_in_ch = bottleneck_ch if i == 0 else self.base_channels * (2 ** (self.depth - i))
            dec_out_ch = self.base_channels * (2 ** (self.depth - i - 1))
            skip_ch = dec_out_ch
            self.decoder.append(
                UpBlock(
                    dec_in_ch + skip_ch,
                    dec_out_ch,
                    kernel_size=self.kernel_size,
                    activation=self.activation,
                )
            )

        self.output_conv = nn.Conv1d(self.base_channels + 1, 1, kernel_size=1)

        if output_activation == "tanh":
            self.output_act: Optional[nn.Module] = nn.Tanh()
        elif output_activation == "relu":
            self.output_act = nn.ReLU()
        else:
            self.output_act = None

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize conv weights with Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                init.ones_(module.weight)
                init.zeros_(module.bias)

    @staticmethod
    def _reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample latent tensor from Gaussian parameters."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """Encode waveform into latent Gaussian parameters and skips."""
        skips = []
        enc = x
        for down_block in self.encoder:
            enc, skip = down_block(enc)
            skips.append(skip)

        enc = self.bottleneck(enc)
        if self.bottleneck_attention is not None:
            for attn_block in self.bottleneck_attention:
                enc = attn_block(enc)
        mu = self.mu_head(enc)
        logvar = torch.clamp(self.logvar_head(enc), min=-10.0, max=10.0)
        z = self._reparameterize(mu, logvar)
        return z, mu, logvar, skips

    def decode(self, z: torch.Tensor, skips: list[torch.Tensor], x_input: torch.Tensor) -> torch.Tensor:
        """Decode latent tensor to waveform prediction."""
        dec = self.latent_to_decoder(z)

        for i, up_block in enumerate(self.decoder):
            skip = skips[self.depth - i - 1]
            dec = up_block(dec, skip)

        dec = torch.cat([dec, x_input], dim=1)
        delta = self.output_conv(dec)

        if self.output_act is not None:
            delta = self.output_act(delta)
            if self.max_scale is not None and isinstance(self.output_act, nn.Tanh):
                delta = delta * self.max_scale

        return x_input + delta

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run forward pass and return reconstruction with latent stats."""
        z, mu, logvar, skips = self.encode(x)
        recon = self.decode(z, skips, x)
        return {
            "recon": recon,
            "mu": mu,
            "logvar": logvar,
            "z": z,
        }


def create_vae_waveunet(
    depth: int = 5,
    base_channels: int = 32,
    kernel_size: int = 9,
    activation: str = "leaky_relu",
    output_activation: Optional[str] = None,
    max_scale: Optional[float] = 0.5,
    latent_channels: int = 64,
    use_bottleneck_attention: bool = False,
    attention_heads: int = 8,
    attention_dropout: float = 0.0,
    attention_layers: int = 1,
    attention_mlp_ratio: float = 2.0,
    device: str = "cpu",
) -> VariationalWaveUNet:
    """Create and place VAE Wave-U-Net model on target device."""
    model = VariationalWaveUNet(
        depth=depth,
        base_channels=base_channels,
        kernel_size=kernel_size,
        activation=activation,
        output_activation=output_activation,
        max_scale=max_scale,
        latent_channels=latent_channels,
        use_bottleneck_attention=use_bottleneck_attention,
        attention_heads=attention_heads,
        attention_dropout=attention_dropout,
        attention_layers=attention_layers,
        attention_mlp_ratio=attention_mlp_ratio,
    )
    return model.to(device)
