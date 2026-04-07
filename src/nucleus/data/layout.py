import torch
import einops

_LAYOUTS = [
    "t h w c", #  Ideal for neighborhood attention, MLPs
    "h w t c",
    "t c h w" # used by bubbleformer and convs
]

def convert_layout(data: torch.Tensor, target_layout: str, source_layout: str = "t h w c") -> torch.Tensor:
    assert target_layout in _LAYOUTS, f"Invalid target layout: {target_layout}"
    assert source_layout in _LAYOUTS, f"Invalid source layout: {source_layout}"
    assert data.dim() >= 4, f"Data must have at least 4 dimensions, got {data.dim()}"
    return einops.rearrange(data, f"... {source_layout} -> ... {target_layout}")