import torch
import torch.nn.functional as F
import math

class LogUniformNoise:
    def __init__(self, min, max, skip_prob):
        self.min = min
        self.max = max
        self.skip_prob = skip_prob # probability to NOT add noise.

    def __call__(self, tensor: torch.Tensor):
        assert tensor.dim() == 5
        log_scale = torch.rand(tensor.shape[0], 1, 1, 1, 1, device=tensor.device) * (math.log(self.max) - math.log(self.min)) + math.log(self.min)
        scale = log_scale.exp()
        noise = torch.randn_like(tensor) * scale
        skip = (torch.rand(tensor.shape[0], 1, 1, 1, 1, device=tensor.device) < self.skip_prob)
        return tensor + noise * (~skip).to(noise.dtype)


class FrameDropout:
    def __init__(self, p: float = 0.1, active_prob: float = 0.4):
        self.p = p
        self.active_prob = active_prob
        
    def __call__(self, x):
        b, t, _, _, c = x.shape
        active = (torch.rand(b, 1, 1, 1, c, device=x.device) < self.active_prob)
        mask = (torch.rand(b, t, 1, 1, c, device=x.device) > self.p).to(x.dtype)
        return x * torch.where(active, mask, torch.ones_like(mask))
    

class FieldDropout:
    def __init__(self, p: float = 0.1, active_prob: float = 0.4):
        self.p = p
        self.active_prob = active_prob
        
    def __call__(self, x):
        b, _, _, _, c = x.shape
        active = (torch.rand(b, 1, 1, 1, c, device=x.device) < self.active_prob)
        mask = (torch.rand(b, 1, 1, 1, c, device=x.device) > self.p).to(x.dtype)
        return x * torch.where(active, mask, torch.ones_like(mask))
