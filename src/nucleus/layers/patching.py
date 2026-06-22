import math
import torch
import torch.nn as nn
from typing import Tuple
import einops

class HMLPEmbed(nn.Module):
    """
    Image to Patch Embedding using hierarchical Conv2d.
    It preserves the spatial ordering of the patches
    Args:
        patch_size (int): Size of the square patch
        in_channels (int): Number of input channels
        embed_dim (int): Dimension of the embedding
    """
    def __init__(
        self,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768
    ):
        super().__init__()
        self.patch_size = patch_size
        num_layers = int(math.log2(patch_size))
        assert (num_layers - math.log2(patch_size)) == 0, "Patch size must be a power of 2"

        self.in_channels = in_channels
        self.embed_dim = embed_dim
        layers = []
        conv_in = in_channels
        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            if num_layers == 1:
                conv_out = embed_dim
            else:
                conv_out = embed_dim if is_last else embed_dim // 4
            layers.append(
                nn.Conv2d(
                    in_channels=conv_in,
                    out_channels=conv_out,
                    kernel_size=2,
                    stride=2,
                    bias=False,
                    dtype=torch.bfloat16
                )
            )
            layers.append(nn.InstanceNorm2d(conv_out, affine=True, dtype=torch.bfloat16))
            if not is_last:
                layers.append(nn.GELU())
            conv_in = conv_out
        self.in_proj = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
        Returns:
            torch.Tensor: Output tensor of shape (B, Emb, H_patches, W_patches)
        """
        x = self.in_proj(x.to(torch.bfloat16))
        return x.to(torch.float32)

class HMLPDebed(nn.Module):
    """
    Patch to Image De-bedding using hierarchical ConvTranspose2d.
    It takes a spatially ordered tensor of embedded patches and reconstructs the image
    Args:
        patch_size (int): Size of the square patch
        out_channels (int): Number of output channels
        embed_dim (int): Dimension of the embedding
    """
    def __init__(
        self,
        patch_size: int = 16,
        out_channels: int = 3,
        embed_dim: int = 768
    ):
        super().__init__()
        self.patch_size = patch_size
        num_layers = int(math.log2(patch_size))
        assert (num_layers - math.log2(patch_size)) == 0, "Patch size must be a power of 2"

        self.out_channels = out_channels
        self.embed_dim = embed_dim
        layers = []
        conv_in = embed_dim
        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            if num_layers == 1:
                conv_out = out_channels
            else:
                conv_out = out_channels if is_last else embed_dim // 4
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=conv_in,
                    out_channels=conv_out,
                    kernel_size=2,
                    stride=2,
                    bias=False,
                    dtype=torch.bfloat16
                )
            )
            if not is_last:
                layers.append(nn.InstanceNorm2d(conv_out, affine=True, dtype=torch.bfloat16))
                layers.append(nn.GELU())
            conv_in = conv_out

        self.out_proj = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, Emb, H_patches, W_patches)
        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W)
        """
        x = self.out_proj(x.to(torch.bfloat16))
        return x.to(torch.float32)

class LinearEmbed(nn.Module):
    def __init__(self, patch_size: int, in_channels: int, embed_dim: int, dtype: torch.dtype):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.linear = nn.Linear(in_channels * patch_size ** 2, embed_dim, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = einops.rearrange(x, "b t (h p1) (w p2) c -> b t h w (c p1 p2)", p1=self.patch_size, p2=self.patch_size)
        x = self.linear(x)
        return x

class LinearDebed(nn.Module):
    def __init__(self, patch_size: int, out_channels: int, embed_dim: int, dtype: torch.dtype):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.embed_dim = embed_dim
        self.linear = nn.Linear(embed_dim, out_channels * patch_size ** 2, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = einops.rearrange(x, "b t h w (c p1 p2) -> b t (h p1) (w p2) c", p1=self.patch_size, p2=self.patch_size)
        return x
    
class AdaptiveDebed(nn.Module):
    """
    Inverts AdaptiveEmbed: upsamples from patch space back to an arbitrary
    target resolution using bilinear interpolation, then projects channels.

    Input:  [..., out_H, out_W, in_channels]
    Output: [..., H, W, out_channels]

    The target spatial size (H, W) is passed at forward time since it is
    determined by the original input resolution, which varies at runtime.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x: torch.Tensor, target_shape: Tuple[int, int]) -> torch.Tensor:
        leading = x.shape[:-3]
        h, w, c = x.shape[-3], x.shape[-2], x.shape[-1]

        flat = x.reshape(-1, h, w, c).permute(0, 3, 1, 2)
        flat = torch.nn.functional.interpolate(flat, size=target_shape, mode="bilinear", align_corners=False)
        flat = flat.permute(0, 2, 3, 1)

        out = self.linear(flat.reshape(*leading, *target_shape, c))
        return out


class AdaptiveEmbed(nn.Module):
    """
    Breaks an arbitrary-resolution input into patches by average pooling to a
    fixed output shape. Patch size is implicitly (H / out_H, W / out_W) and
    adapts to whatever resolution is passed at runtime.

    Input:  [..., H, W, C]
    Output: [..., out_H, out_W, C]
    """
    def __init__(self, in_channels, out_channels, out_shape: Tuple[int, int]):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.out_shape = out_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        leading = x.shape[:-3]
        
        x = self.linear(x)

        # AdaptiveAvgPool2d expects (N, C, H, W)
        h, w, c = x.shape[-3], x.shape[-2], x.shape[-1]
        flat = x.reshape(-1, h, w, c).permute(0, 3, 1, 2)
        flat = torch.nn.functional.adaptive_avg_pool2d(flat, self.out_shape)
        flat = flat.permute(0, 2, 3, 1)

        return flat.reshape(*leading, *self.out_shape, c)