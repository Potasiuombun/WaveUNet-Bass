"""Model factory utilities.

This centralizes model creation so new model families can be registered
incrementally without changing every training script.
"""

from typing import Any, Dict

import torch.nn as nn

from .waveunet import create_waveunet
from .vae_waveunet import create_vae_waveunet


def create_model(model_config: Dict[str, Any], device: str) -> nn.Module:
    """Create a model from config.

    Supported model names:
    - waveunet
    - vae_waveunet

    Future planned names:
    - vae_waveunet
    - attention_vae_waveunet

    Args:
        model_config: Model section from YAML config.
        device: Runtime device string ("cpu" or "cuda").

    Returns:
        Instantiated model on requested device.

    Raises:
        ValueError: If model name is unknown or not implemented yet.
    """
    model_name = str(model_config.get("type", model_config.get("name", "waveunet"))).lower()

    if model_name == "waveunet":
        output_activation = model_config.get("output_activation")
        if output_activation is None and "output_head" in model_config:
            head = model_config.get("output_head")
            output_activation = None if head == "identity" else head

        return create_waveunet(
            depth=int(model_config["depth"]),
            base_channels=int(model_config["base_channels"]),
            kernel_size=int(model_config["kernel_size"]),
            activation=str(model_config.get("activation", "leaky_relu")),
            output_activation=output_activation,
            max_scale=model_config.get("max_scale"),
            use_bottleneck_attention=bool(model_config.get("use_bottleneck_attention", False)),
            attention_heads=int(model_config.get("attention_heads", 8)),
            attention_dropout=float(model_config.get("attention_dropout", 0.0)),
            attention_layers=int(model_config.get("attention_layers", 1)),
            attention_mlp_ratio=float(model_config.get("attention_mlp_ratio", 2.0)),
            device=device,
        )

    if model_name == "vae_waveunet":
        output_activation = model_config.get("output_activation")
        if output_activation is None and "output_head" in model_config:
            head = model_config.get("output_head")
            output_activation = None if head == "identity" else head

        return create_vae_waveunet(
            depth=int(model_config["depth"]),
            base_channels=int(model_config["base_channels"]),
            kernel_size=int(model_config["kernel_size"]),
            activation=str(model_config.get("activation", "leaky_relu")),
            output_activation=output_activation,
            max_scale=model_config.get("max_scale"),
            latent_channels=int(model_config.get("latent_channels", 64)),
            use_bottleneck_attention=bool(model_config.get("use_bottleneck_attention", False)),
            attention_heads=int(model_config.get("attention_heads", 8)),
            attention_dropout=float(model_config.get("attention_dropout", 0.0)),
            attention_layers=int(model_config.get("attention_layers", 1)),
            attention_mlp_ratio=float(model_config.get("attention_mlp_ratio", 2.0)),
            device=device,
        )

    if model_name == "attention_vae_waveunet":
        output_activation = model_config.get("output_activation")
        if output_activation is None and "output_head" in model_config:
            head = model_config.get("output_head")
            output_activation = None if head == "identity" else head

        return create_vae_waveunet(
            depth=int(model_config["depth"]),
            base_channels=int(model_config["base_channels"]),
            kernel_size=int(model_config["kernel_size"]),
            activation=str(model_config.get("activation", "leaky_relu")),
            output_activation=output_activation,
            max_scale=model_config.get("max_scale"),
            latent_channels=int(model_config.get("latent_channels", 64)),
            use_bottleneck_attention=True,
            attention_heads=int(model_config.get("attention_heads", 8)),
            attention_dropout=float(model_config.get("attention_dropout", 0.0)),
            attention_layers=int(model_config.get("attention_layers", 1)),
            attention_mlp_ratio=float(model_config.get("attention_mlp_ratio", 2.0)),
            device=device,
        )

    raise ValueError(f"Unknown model name '{model_name}'.")
