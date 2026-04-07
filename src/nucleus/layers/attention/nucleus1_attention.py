r"""
Includes all the attention modules for ablating Nucleus1's space-time attention blocks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from einops import rearrange
import einops
from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb
import math
import time
import natten

class Nucleus1SpatialAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.input_head = nn.Linear(embed_dim, 3 * embed_dim, dtype=torch.bfloat16)
        self.output_head = nn.Linear(embed_dim, embed_dim, dtype=torch.bfloat16)
        self.qnorm = nn.LayerNorm(self.head_dim, dtype=torch.bfloat16)
        self.knorm = nn.LayerNorm(self.head_dim, dtype=torch.bfloat16)
        
        # TODO: should each attention block use the same rotary embedding?
        self.rotary_emb = RotaryEmbedding(
            # NOTE: This must be smaller than the head dim. 
            dim=self.head_dim // 3,
            freqs_for="pixel",
            max_freq=256
        )
        
        self.work_dtype = torch.bfloat16

    def forward(self, x):
        b, t, h, w, c = x.shape
        
        # rotary embedding expects seq-last [batch, heads, seq1, seq2, dim] layout
        heads = einops.rearrange(self.input_head(x.to(self.work_dtype)), 
                                 "b t h w (heads head_dim) -> (b t) heads h w head_dim", 
                                 heads=self.num_heads).contiguous()
        q, k, v = heads.tensor_split(3, dim=-1)
        q = self.qnorm(q)
        k = self.knorm(k)
        freqs = self.rotary_emb.get_axial_freqs(h, w)
        q = apply_rotary_emb(freqs, q)
        k = apply_rotary_emb(freqs, k)
        
        # SPDA expects sequence to be flattened [batch, heads, seq1 * seq2, dim] layout
        q, k, v = map(
            lambda qkv: rearrange(qkv, "bt heads h w head_dim -> bt heads (h w) head_dim").contiguous(), [q, k, v]
        )
        
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            output = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v
            )
             
        output = einops.rearrange(output,
                                  "(b t) heads (h w) head_dim -> b t h w (heads head_dim)", 
                                  b=b, t=t, h=h, w=w, heads=self.num_heads).contiguous()
        output = self.output_head(output).to(torch.float32)
        return output
    
class Nucleus1SpatialAxialAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
    ):
        """
        Args:
            embed_dim (int):Embedding dimension
            num_heads (int): Number of attention heads
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.input_head = nn.Linear(embed_dim, 3 * embed_dim, dtype=torch.bfloat16)
        self.output_head = nn.Linear(embed_dim, embed_dim, dtype=torch.bfloat16)
        self.qnorm = nn.LayerNorm(self.head_dim, dtype=torch.bfloat16)
        self.knorm = nn.LayerNorm(self.head_dim, dtype=torch.bfloat16)
        
        # TODO: should each attention block use the same rotary embedding?
        self.rotary_emb = RotaryEmbedding(
            # NOTE: This must be smaller than the head dim. 
            dim=self.head_dim // 3,
            freqs_for="pixel",
            max_freq=256
        )
        
        self.work_dtype = torch.bfloat16
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, H, W, C)
        Returns:
            torch.Tensor: Output tensor of shape (B, T, H, W, C)
        """
        batch_size, t, h, w, c = x.shape
        
        x = self.input_head(x.to(self.work_dtype))
        q, k, v = x.tensor_split(3, dim=-1)
        q = q.view(batch_size, t, h, w, self.num_heads, self.head_dim)
        k = k.view(batch_size, t, h, w, self.num_heads, self.head_dim)
        v = v.view(batch_size, t, h, w, self.num_heads, self.head_dim)
        # rearrange for rotary embedding
        q, k, v = map(
            lambda x: rearrange(x, "b t h w num_heads head_dim -> b t num_heads h w head_dim"), [q, k, v]
        )
        q = self.qnorm(q)
        k = self.knorm(k)
        
        freqs = self.rotary_emb.get_axial_freqs(h, w)
        q = apply_rotary_emb(freqs, q)
        k = apply_rotary_emb(freqs, k)
       
        # X direction attention
        qx, kx, vx = map(
            lambda x: rearrange(x, "b t num_heads h w head_dim -> (b t h) num_heads w head_dim"), [q, k, v]
        )
        xx = F.scaled_dot_product_attention(
            query=qx.contiguous(),
            key=kx.contiguous(),
            value=vx.contiguous(),
        )
        xx = rearrange(xx, "(b t h) num_heads w head_dim -> b t h w (num_heads head_dim)", t=t, h=h).contiguous()

        # Y direction attention
        qy, ky, vy = map(
            lambda x: rearrange(x, "b t num_heads h w head_dim ->  (b t w) num_heads h head_dim"), [q, k, v]
        )
        xy = F.scaled_dot_product_attention(
            query=qy.contiguous(),
            key=ky.contiguous(),
            value=vy.contiguous(),
        )
        xy = rearrange(xy, "(b t w) num_heads h head_dim -> b t h w (num_heads head_dim)", t=t, w=w).contiguous()
        
        x = (xx + xy) / 2
        x = self.output_head(x)
        return x.to(torch.float32)
    
class Nucleus1SpatialNeighborhoodAttention(nn.Module):
    r"""
    This is similar to natten's NaighborhoodAttention2D,
    but includes additional query and key normalization.
    """
    def __init__(
        self,
        embed_dim,
        num_heads,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.kernel_size = kernel_size

        self.input_head = nn.Linear(embed_dim, 3 * embed_dim, dtype=torch.bfloat16)
        self.output_head = nn.Linear(embed_dim, embed_dim, dtype=torch.bfloat16)
        self.qnorm = nn.LayerNorm(self.head_dim, dtype=torch.bfloat16)
        self.knorm = nn.LayerNorm(self.head_dim, dtype=torch.bfloat16)
        
        # TODO: should each attention block use the same rotary embedding?
        self.rotary_emb = RotaryEmbedding(
            # NOTE: This must be smaller than the head dim. 
            dim=self.head_dim // 3,
            freqs_for="pixel",
            max_freq=256
        )
        
        self.work_dtype = torch.bfloat16

    def forward(self, x):
        b, t, h, w, c = x.shape
        
        # rotary embedding expects seq-last [batch, heads, seq1, seq2, dim] layout
        heads = einops.rearrange(self.input_head(x.to(self.work_dtype)), 
                                 "b t h w (heads head_dim) -> (b t) heads h w head_dim", 
                                 heads=self.num_heads).contiguous()
        q, k, v = heads.tensor_split(3, dim=-1)
        q = self.qnorm(q)
        k = self.knorm(k)
        freqs = self.rotary_emb.get_axial_freqs(h, w)
        q = apply_rotary_emb(freqs, q)
        k = apply_rotary_emb(freqs, k)
        
        # natten expects head-last [batch, seq1, seq2, heads, dim] layout
        q, k, v = map(
            lambda qkv: rearrange(qkv, "bt heads h w head_dim -> bt h w heads head_dim").contiguous(), [q, k, v]
        )
        
        output = natten.na2d(
            q,
            k,
            v,
            kernel_size=self.kernel_size,
            stride=1,
            dilation=1,   
        )
        
        output = einops.rearrange(output,
                                  "(b t) h w heads head_dim -> b t h w (heads head_dim)", 
                                  b=b, t=t).contiguous()
        output = self.output_head(output).to(torch.float32)
        return output
    
class Nucleus1TemporalAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
    ):
        """
        Args:
            embed_dim (int): Number of features in the input tensor
            num_heads (int): Number of attention heads
            drop_path (float): Drop path rate
            bias_type (str): Type of bias to use in the attention mechanism
            attn_scale (bool): Whether to apply attention scaling
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.input_head = nn.Linear(embed_dim, 3 * embed_dim, dtype=torch.bfloat16)
        self.output_head = nn.Linear(embed_dim, embed_dim, dtype=torch.bfloat16)
        self.qnorm = nn.LayerNorm(self.head_dim, dtype=torch.bfloat16)
        self.knorm = nn.LayerNorm(self.head_dim, dtype=torch.bfloat16)
        
        self.rotary_emb = RotaryEmbedding(dim=32)
        self.work_dtype = torch.bfloat16

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, H, W, C)
        Returns:
            torch.Tensor: Output tensor of shape (B, T, H, W, C)
        """
        batch_size, t, h, w, c = x.shape
        inp = x.clone()
        x = self.input_head(x.to(self.work_dtype))
        x = x.view(batch_size, t, h, w, self.num_heads, 3 * self.head_dim)
        x = rearrange(x, "b t h w heads head_dim -> (b h w) heads t head_dim").contiguous()
        q, k, v = x.tensor_split(3, dim=-1)
        q = self.qnorm(q)
        k = self.knorm(k)
        
        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)

        # Sequence length is really small (like 5)... So, just compute the attention manually.
        x = F.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.head_dim), dim=-1) @ v
        x = rearrange(x, "(b h w) num_heads t head_dim -> b t h w (num_heads head_dim)", 
                      t=t, h=h, w=w).contiguous()
        
        x = self.output_head(x).to(torch.float32)
        return x