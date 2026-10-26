from typing import Dict, Optional, Tuple

import torch
from boiling_data.flashx.batch import FlashXBatch
from boiling_data.flashx.normalizer import NormalizerWrapper
from boiling_data.tasks.forecast.flashx_batch import FlashXForecastBatch

PredictionTargetPairs = Dict[str, Tuple[torch.Tensor, torch.Tensor]]


def normalize_forecast_batch(normalizer: NormalizerWrapper, batch: FlashXForecastBatch) -> FlashXForecastBatch:
    return FlashXForecastBatch(normalizer.normalize(batch.input), normalizer.normalize(batch.target))


def pooled_l1_loss(pairs: PredictionTargetPairs, cell_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """The mean absolute error over the cells of every predicted grid together, so
    fields on grids of different shapes, such as cell centers and faces, weigh by
    their number of samples. With (B, T, H, W) cell_weights, the weighted mean, each
    face weighted as the heavier of the cells it borders."""
    if cell_weights is None:
        total_abs_error = sum((prediction - goal).abs().sum() for prediction, goal in pairs.values())
        total_elements = sum(prediction.numel() for prediction, _ in pairs.values())
        return total_abs_error / total_elements
    total_abs_error = total_weight = 0.0
    for prediction, goal in pairs.values():
        weights = grid_weights(cell_weights, prediction.shape[-2:])
        total_abs_error = total_abs_error + (weights * (prediction - goal).abs()).sum()
        total_weight = total_weight + weights.sum()
    return total_abs_error / total_weight


def weighted_l1_loss(prediction: torch.Tensor, goal: torch.Tensor, cell_weights: torch.Tensor) -> torch.Tensor:
    """The mean absolute error of (B, T, H, W, C) cell fields, weighted by (B, T, H, W) cell_weights."""
    weights = cell_weights[..., None]
    return (weights * (prediction - goal).abs()).sum() / (weights.sum() * prediction.shape[-1])


def interface_weights(
    normalizer: NormalizerWrapper, target: FlashXBatch, interface_weight: float, band_cells: float
) -> Optional[torch.Tensor]:
    """(B, T, H, W) loss weights on the normalized target's cells: interface_weight
    within band_cells cells of the target's interface, 1 elsewhere, so the loss favors
    the interface over the far field that most cells hold; None for a weight of 1."""
    if interface_weight == 1.0:
        return None
    with torch.no_grad():
        sdf = normalizer.unnormalize_field("dfun", target.fields["dfun"])
        band = band_cells * target.dx("dfun").to(sdf.dtype)[:, None, None, None]
        return torch.where(sdf.abs() < band, interface_weight, 1.0).to(sdf.dtype)


def grid_weights(cell_weights: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """cell_weights (..., H, W) on a grid of (H, W) cells, (H, W + 1) x-faces or
    (H + 1, W) y-faces, a face taking the larger weight of the cells it borders."""
    height, width = cell_weights.shape[-2:]
    if tuple(shape) == (height, width):
        return cell_weights
    for dim, face_shape in ((-1, (height, width + 1)), (-2, (height + 1, width))):
        if tuple(shape) == face_shape:
            cells = cell_weights.shape[dim]
            interior = torch.maximum(cell_weights.narrow(dim, 0, cells - 1), cell_weights.narrow(dim, 1, cells - 1))
            # the boundary faces border only one cell
            first, last = cell_weights.narrow(dim, 0, 1), cell_weights.narrow(dim, cells - 1, 1)
            return torch.cat([first, interior, last], dim=dim)
    raise ValueError(f"a grid of shape {tuple(shape)} is neither the ({height}, {width}) cells nor their faces")


def per_field_mae(pairs: PredictionTargetPairs, prefix: str) -> Dict[str, torch.Tensor]:
    return {f"{prefix}/mae_{name}": (prediction - goal).abs().mean().detach() for name, (prediction, goal) in pairs.items()}
