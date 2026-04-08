import torch
import torch.nn as nn

from nucleus.layers.attention import (
    Nucleus1TemporalAttention,
    Nucleus1SpatialNeighborhoodAttention, 
    Nucleus1SpatialAttention, 
    Nucleus1SpatialAxialAttention,
)

class Nucleus1SpaceTimeAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int
    ):
        super().__init__()
        self.temporal = Nucleus1TemporalAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
        )
        self.spatial = Nucleus1SpatialAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal(x)
        x = self.spatial(x)
        return x

class Nucleus1SpaceTimeNeighborAttention(Nucleus1SpaceTimeAttention):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__(embed_dim, num_heads)
        
        self.spatial = Nucleus1SpatialNeighborhoodAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
        )
    
class Nucleus1SpaceTimeAxialAttention(Nucleus1SpaceTimeAttention):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__(embed_dim, num_heads)
        
        self.spatial = Nucleus1SpatialAxialAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
        )