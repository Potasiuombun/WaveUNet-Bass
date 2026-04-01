"""Evaluation utilities."""
import torch
from typing import Dict
from pathlib import Path

from ..training.metrics import nmse, mae, peak_reduction, crest_factor, loudness_proxy


def evaluate_batch(
    outputs: torch.Tensor,
    targets: torch.Tensor
) -> Dict[str, float]:
    """Evaluate a batch of predictions.
    
    Args:
        outputs: Predictions [batch, 1, T].
        targets: Targets [batch, 1, T].
        
    Returns:
        Dictionary of metrics (averaged over batch).
    """
    metrics = {}
    
    # Batch metrics
    metrics["batch_nmse"] = nmse(outputs, targets)
    metrics["batch_mae"] = mae(outputs, targets)
    metrics["batch_peak_reduction"] = peak_reduction(outputs, targets)
    
    # Per-sample metrics (averaged)
    crest_factors_in = []
    crest_factors_out = []
    loudness_in = []
    loudness_out = []
    
    for i in range(len(outputs)):
        output = outputs[i].squeeze(0)
        target = targets[i].squeeze(0)
        
        crest_factors_in.append(crest_factor(target))
        crest_factors_out.append(crest_factor(output))
        loudness_in.append(loudness_proxy(target))
        loudness_out.append(loudness_proxy(output))
    
    metrics["avg_cf_target"] = sum(crest_factors_in) / len(crest_factors_in)
    metrics["avg_cf_output"] = sum(crest_factors_out) / len(crest_factors_out)
    metrics["avg_loudness_target"] = sum(loudness_in) / len(loudness_in)
    metrics["avg_loudness_output"] = sum(loudness_out) / len(loudness_out)
    
    return metrics


def evaluate_dataset(
    model: torch.nn.Module,
    dataloader,
    device: str = "cpu"
) -> Dict[str, float]:
    """Evaluate model on entire dataset.
    
    Args:
        model: Inference model.
        dataloader: Data loader.
        device: Device to run on.
        
    Returns:
        Dictionary of aggregated metrics.
    """
    model.eval()
    
    aggregate_metrics = {}
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            
            outputs = model(inputs)
            batch_metrics = evaluate_batch(outputs, targets)
            
            for key, value in batch_metrics.items():
                if key not in aggregate_metrics:
                    aggregate_metrics[key] = 0.0
                aggregate_metrics[key] += value
            
            num_batches += 1
    
    # Average
    for key in aggregate_metrics:
        aggregate_metrics[key] /= num_batches
    
    return aggregate_metrics
