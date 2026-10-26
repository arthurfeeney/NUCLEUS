import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseMLP(nn.Module):
    """A two-layer GELU MLP shaped like one expert of TopkMoE, applied to every token,
    so a model can swap it in for the MoE."""

    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        # float32, so small updates are not rounded away; cast to the input's dtype in forward
        self.w1 = nn.Parameter(torch.empty(intermediate_dim, hidden_dim))
        self.w2 = nn.Parameter(torch.empty(hidden_dim, intermediate_dim))
        self.reset_parameters()

    def reset_parameters(self):
        # the initialization of a TopkMoE expert, so the two start alike
        with torch.no_grad():
            gain = math.sqrt(2)
            self.w1.normal_(0, gain / math.sqrt(self.hidden_dim))
            self.w2.normal_(0, gain / math.sqrt(self.intermediate_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: ``(..., hidden_dim)``; returns ``(..., hidden_dim)`` in x's dtype."""
        return F.linear(F.gelu(F.linear(x, self.w1.to(x.dtype))), self.w2.to(x.dtype))
