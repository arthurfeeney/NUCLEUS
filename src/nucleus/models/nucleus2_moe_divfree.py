import dataclasses
from dataclasses import dataclass, field, replace
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import record_function
from rotary_embedding_torch import RotaryEmbedding
from typing import Dict, Literal, Optional
import wandb
from lightning.pytorch.loggers.wandb import WandbLogger

from nucleus.layers.adaptive_layernorm import AdaptiveLayerNorm
from nucleus.layers.attention import NeighborhoodAttention
from nucleus.layers.moe.topk_moe import TopkMoE, TopkMoEOutput, TopkRouterWithBias
from nucleus.layers.droppath import DropPath
from nucleus.layers import (
    AdaptiveEmbed,
    AdaptiveDebed,
    LinearEmbed,
    LinearDebed,
    OverlappingPatchDebed
)
from nucleus.data.in_mem_divfree_forecast_dataset import DivFreeBatch, DivFreeData
from nucleus.trajectory import Trajectory
from nucleus.models.modules import MoEConditionedForecastModule
from nucleus.noise import LogUniformNoise
from nucleus.utils.sdf_reinit import sdf_reinit_sussman
from nucleus.utils.inf_stabilizer import clip_temp_by_phase
from nucleus.physics.poisson import (
    solve_poisson_neumann_dirichlet,
    reconstruct_velocity_from_helmholtz,
    helmholtz_from_faces,
    divergence_centers_from_faces,
    grad_faces_from_centers,
    curl_faces_from_nodes,
    GRID_SPACING,
)
from nucleus.physics.sdf import band_mask
from nucleus.physics.mass_transfer import continuity

from ._api import register_model, load_model_state_dict

__all__ = [
    "Nucleus2MoEDivFree",
    "Nucleus2MoEDivFreeConfig",
    "Nucleus2MoEDivFreeInput",
    "Nucleus2MoEDivFreeOutput",
]


_DTYPE_TO_STR: dict[torch.dtype, str] = {
    torch.float32: "float32",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
}
_STR_TO_DTYPE: dict[str, torch.dtype] = {v: k for k, v in _DTYPE_TO_STR.items()}


NUCLEUS_DTYPE = Literal["float32", "float16", "bfloat16"]


@dataclass
class Nucleus2MoEDivFreeConfig:
    patch_size: int
    embed_dim: int
    num_heads: int
    processor_blocks: int
    num_experts: int
    topk: int
    moe_intermediate_dim: int
    patching: Literal["Linear", "Adaptive"]
    embed_dtype: NUCLEUS_DTYPE = "float32"
    debed_dtype: NUCLEUS_DTYPE = "float32"
    activation_dtype: NUCLEUS_DTYPE = "float32"
    attention_dtype: NUCLEUS_DTYPE = "bfloat16"
    moe_dtype: NUCLEUS_DTYPE = "bfloat16"
    # Backward-only rescale applied to the potentials on the velocity-reconstruction
    # branch. The reconstruction curls/grads the potentials and divides by dx, so the
    # velocity loss's gradient reaches the psi/phi heads amplified by 1/dx = 32.
    # Scaling that gradient by dx (the default) cancels the amplification exactly;
    # 1.0 disables it, and intermediate values (e.g. dx**0.5) damp it partially.
    # Only affects the backward pass -- the forward velocity is unchanged.
    potential_grad_scale: float = GRID_SPACING


def _config_to_dict(config: Nucleus2MoEDivFreeConfig) -> dict:
    d = dataclasses.asdict(config)
    dtype_fields = {"embed_dtype", "debed_dtype", "activation_dtype", "attention_dtype", "moe_dtype"}
    for key in dtype_fields:
        if not isinstance(d[key], str):
            d[key] = _DTYPE_TO_STR[d[key]]
    return d


def _config_from_dict(d: dict) -> Nucleus2MoEDivFreeConfig:
    d = dict(d)
    dtype_fields = {"embed_dtype", "debed_dtype", "activation_dtype", "attention_dtype", "moe_dtype"}
    for key in dtype_fields:
        if not isinstance(d[key], torch.dtype):
            d[key] = _STR_TO_DTYPE[d[key]]
    return Nucleus2MoEDivFreeConfig(**d)

def get_dtype(dtype):
    if isinstance(dtype, str):
        return _STR_TO_DTYPE[dtype]
    return dtype


class TransformerMoEBlock(nn.Module):
    def __init__(self, config: Nucleus2MoEDivFreeConfig, num_sim_params: int, drop_path_prob: float):
        super().__init__()

        self.activation_dtype = get_dtype(config.activation_dtype)
        self.attention_dtype = get_dtype(config.attention_dtype)
        self.moe_dtype = get_dtype(config.moe_dtype)

        self.drop_path = DropPath(drop_path_prob)

        self.attention_norm = AdaptiveLayerNorm(config.embed_dim, num_sim_params, dtype=self.attention_dtype)
        self.mlp_norm = AdaptiveLayerNorm(config.embed_dim, num_sim_params, self.moe_dtype)

        self.router = TopkRouterWithBias(
            config.num_experts,
            config.embed_dim,
            config.topk,
            bias_update_rate=0.001,
            softmax_first=False
        )

        self.attention = NeighborhoodAttention(embed_dim=config.embed_dim, num_heads=config.num_heads)

        self.mlp = TopkMoE(
            num_experts=config.num_experts,
            hidden_dim=config.embed_dim,
            intermediate_dim=config.moe_intermediate_dim,
            topk=config.topk,
            router=self.router
        )

    def _attention(self, x: torch.Tensor, freqs: torch.Tensor, sim_params: torch.Tensor) -> torch.Tensor:
        with record_function("attention"):
            h = x.to(self.attention_dtype)
            h = self.attention_norm(h, sim_params)
            h = self.attention(h, freqs)
            h = self.drop_path(h)
            x = x + h.to(self.activation_dtype)
        return x

    def _mlp(self, x: torch.Tensor, sim_params: torch.Tensor):
        with record_function("moe"):
            h = x.to(self.moe_dtype)
            h = self.mlp_norm(h, sim_params)
            moe_output: TopkMoEOutput = self.mlp(h)
            h = self.drop_path(moe_output.out)
            x = x + h.to(self.activation_dtype)
        return x, moe_output

    def forward(self, x: torch.Tensor, freqs: torch.Tensor, sim_params: torch.Tensor):
        x = self._attention(x, freqs, sim_params)
        x, moe_output = self._mlp(x, sim_params)
        return x, moe_output

class ChannelsLastConv2d(nn.Module):
    """Conv2d over the trailing spatial dims of a channels-last (B, T, H, W, C)
    tensor. Time is folded into the batch so each frame is convolved
    independently (no temporal mixing), then the layout is restored.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, **conv_kwargs):
        super().__init__()
        conv_kwargs.setdefault("padding", kernel_size // 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, **conv_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, height, width, channels = x.shape
        x = x.permute(0, 1, 4, 2, 3).reshape(batch * time, channels, height, width)
        x = self.conv(x)
        _, out_channels, out_height, out_width = x.shape
        x = x.reshape(batch, time, out_channels, out_height, out_width)
        return x.permute(0, 1, 3, 4, 2).contiguous()

class PhaseModulate(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.phase_table = nn.Embedding(2, 2 * embed_dim)
        # Start as identity: gamma = beta = 0 so (1 + gamma) * x + beta == x.
        nn.init.zeros_(self.phase_table.weight)

    def forward(self, x: torch.Tensor, phase_mask: torch.Tensor) -> torch.Tensor:
        assert x.shape[:-1] == phase_mask.shape

        gamma_beta = self.phase_table(phase_mask)  # (B, T, H, W, 2 * C)
        gamma, beta = gamma_beta.chunk(2, dim=-1)

        assert gamma.shape == x.shape
        assert beta.shape == x.shape

        return (1 + gamma) * x + beta

# NOTE: These settings are hard-coded for pool boiling.
# NOTE: They should ideally be read from the data config files.
# Physical extent of the simulation domain on a cell-centered grid (endpoints
# excluded): height (y) in [0, 16], width (x) in [-8, 8].
DOMAIN_Y_MIN, DOMAIN_Y_MAX = 0.0, 16.0
DOMAIN_X_MIN, DOMAIN_X_MAX = -8.0, 8.0

# Grid spacing passed to the velocity gradients. TODO: fixed-resolution value;
# switch to (extent / num_cells) per axis for a resolution-consistent velocity.
GRADIENT_SPACING = GRID_SPACING

# Free-slip window length scale, in grid cells: psi is windowed to zero at the
# closed walls over ~this many cells (tanh) so each wall is a streamline
# (v . n = 0, no-penetration) with the tangential velocity left free.
WALL_SLIP_CELLS = 2.0

# Dilatational wall band, in grid cells so it stays a fixed physical-cell
# thickness at any resolution: grad(phi) is held at exactly zero within
# WALL_ZERODIV_CELLS of every closed wall (so it adds no divergence there) and
# ramps back to full over the next WALL_ZERODIV_RAMP_CELLS.
WALL_ZERODIV_CELLS = 2.0
WALL_ZERODIV_RAMP_CELLS = 2.0


def nodal_psi_from_corners(psi_bottom_left, psi_bottom_right, psi_top_left, psi_top_right):
    """Average the four per-cell corner predictions of the streamfunction back onto
    the shared nodal grid.
    Each ``psi_*`` has shape ``(..., H, W)``; returns nodal psi ``(..., H+1, W+1)``.
    """
    # F.pad on the last two dims takes (W_before, W_after, H_before, H_after). Each
    # corner occupies a different (H+1, W+1) sub-block: bottom-left fills the low
    # indices, top-right the high indices, etc.
    psi_sum = (
        F.pad(psi_bottom_left, (0, 1, 0, 1))
        + F.pad(psi_bottom_right, (1, 0, 0, 1))
        + F.pad(psi_top_left, (0, 1, 1, 0))
        + F.pad(psi_top_right, (1, 0, 1, 0))
    )
    ones = torch.ones_like(psi_bottom_left)
    counts = (
        F.pad(ones, (0, 1, 0, 1))
        + F.pad(ones, (1, 0, 0, 1))
        + F.pad(ones, (0, 1, 1, 0))
        + F.pad(ones, (1, 0, 1, 0))
    )
    return psi_sum / counts


def velocity_from_potentials(psi, phi):
    height, width = psi.shape[-2], psi.shape[-1]
    velx_sol = torch.gradient(psi, dim=-2, spacing=GRADIENT_SPACING)[0]     #  ∂ψ/∂y
    vely_sol = -torch.gradient(psi, dim=-1, spacing=GRADIENT_SPACING)[0]    # -∂ψ/∂x
    velx_dil = torch.gradient(phi, dim=-1, spacing=GRADIENT_SPACING)[0]     #  ∂φ/∂x
    vely_dil = torch.gradient(phi, dim=-2, spacing=GRADIENT_SPACING)[0]     #  ∂φ/∂y

    velx = velx_sol + velx_dil
    vely = vely_sol + vely_dil
    return velx, vely


def scale_gradient(tensor: torch.Tensor, scale: float) -> torch.Tensor:
    """Identity in the forward pass; multiplies the gradient by ``scale`` in the
    backward pass (a straight-through rescale, written without autograd.Function so it
    traces cleanly under torch.compile). Placed on the psi/phi reconstruction branch to
    cancel the 1/dx curl/grad amplification of the velocity loss's gradient into the
    potential heads, without changing the reconstructed velocity or the potentials'
    value scale (so the direct psi/phi supervision is left untouched)."""
    # tensor*scale is the only path that carries gradient; the detached remainder makes
    # the forward value exactly `tensor` again.
    return tensor * scale + tensor.detach() * (1.0 - scale)


def clean_phi(phi, div_gate):
    """
    `phi` produced by a model may be noisy / incorrect. This means it can have undesired
    divergence. This function uses div_gate to remove unwanted divergence from phi, while preserving the
    fact that grad(phi) = vel.
    """
    grad_x, grad_y = grad_faces_from_centers(phi, GRADIENT_SPACING, GRADIENT_SPACING)
    div = divergence_centers_from_faces(grad_x, grad_y, GRADIENT_SPACING, GRADIENT_SPACING)
    gated_div = div_gate * div
    return solve_poisson_neumann_dirichlet(gated_div, GRADIENT_SPACING, GRADIENT_SPACING)


def divfree_input_to_cells(
    sdf: torch.Tensor,
    temperature: torch.Tensor,
    velx: torch.Tensor,
    vely: torch.Tensor,
) -> torch.Tensor:
    """Cell-centered input channels the patch embedding consumes: sdf, temperature,
    and the face velocities split onto their two bordering cells. psi/phi are model
    *outputs* only -- they are reconstructed from the predicted potentials and are not
    fed back into the embedding.

    Args (leading batch/time dims written ``...``):
        sdf, temperature: ``(..., H, W)`` cell-centered.
        velx: ``(..., H, W+1)`` x-face velocity.
        vely: ``(..., H+1, W)`` y-face velocity.

    Returns:
        ``(..., H, W, 6)`` ordered ``[sdf, temperature, vel_left, vel_right,
        vel_bottom, vel_top]``.
    """
    vel_left, vel_right = velx[..., :, :-1], velx[..., :, 1:]
    vel_bottom, vel_top = vely[..., :-1, :], vely[..., 1:, :]
    return torch.stack(
        (sdf, temperature, vel_left, vel_right, vel_bottom, vel_top), dim=-1
    )


def divfree_fields_to_cells(
    sdf: torch.Tensor,
    temperature: torch.Tensor,
    velx: torch.Tensor,
    vely: torch.Tensor,
    psi: torch.Tensor,
    phi: torch.Tensor,
) -> torch.Tensor:
    """Split natural-grid divfree fields onto the dataset's 11 cell-centered
    channels: each x-face velocity onto the left/right face of its two cells, each
    y-face velocity onto their bottom/top face, and ``psi`` onto its four cell
    corners. The inverse of the dataset's per-cell split.

    Args (leading batch/time dims written ``...``):
        sdf, temperature, phi: ``(..., H, W)`` cell-centered.
        velx: ``(..., H, W+1)`` x-face velocity.
        vely: ``(..., H+1, W)`` y-face velocity.
        psi: ``(..., H+1, W+1)`` nodal streamfunction.

    Returns:
        ``(..., H, W, 11)`` in the channel order the dataset produces
        ``[sdf, temperature, vel_left, vel_right, vel_bottom, vel_top,
        psi_bl, psi_br, psi_tl, psi_tr, phi]``.
    """
    vel_left, vel_right = velx[..., :, :-1], velx[..., :, 1:]
    vel_bottom, vel_top = vely[..., :-1, :], vely[..., 1:, :]
    return torch.stack(
        (
            sdf, temperature,
            vel_left, vel_right, vel_bottom, vel_top,
            psi[..., :-1, :-1], psi[..., :-1, 1:], psi[..., 1:, :-1], psi[..., 1:, 1:],
            phi,
        ),
        dim=-1,
    )


@dataclass
class Nucleus2MoEDivFreeInput:
    """ Shapes (leading batch/time dims written ``...``):
        ``sdf``, ``temperature``, ``phi``: ``(..., H, W)`` cell-centered.
        ``velx``: ``(..., H, W+1)`` x-face velocity.
        ``vely``: ``(..., H+1, W)`` y-face velocity.
        ``psi``: ``(..., H+1, W+1)`` nodal streamfunction.
    """
    sdf: torch.Tensor
    temperature: torch.Tensor
    velx: torch.Tensor
    vely: torch.Tensor
    psi: torch.Tensor
    phi: torch.Tensor

    def to_cell_tensor(self) -> torch.Tensor:
        """Split the fields onto the dataset's 11 cell-centered channels, shape
        ``(..., H, W, 11)`` (see :func:`divfree_fields_to_cells`)."""
        return divfree_fields_to_cells(
            self.sdf, self.temperature, self.velx, self.vely, self.psi, self.phi
        )

    @classmethod
    def from_cell_tensor(cls, cells: torch.Tensor) -> "Nucleus2MoEDivFreeInput":
        """Recombine the dataset's 11 cell channels back onto the natural grids,
        averaging the values that overlap: a shared face is the mean of the two
        cells that border it, a shared node the mean of the (up to four) cells that
        corner it, and boundary faces/nodes come from their single owning cell. The
        inverse of :meth:`to_cell_tensor`. ``cells`` has shape ``(..., H, W, 11)``.
        """
        sdf, temperature, phi = cells[..., 0], cells[..., 1], cells[..., 10]
        vel_left, vel_right = cells[..., 2], cells[..., 3]
        vel_bottom, vel_top = cells[..., 4], cells[..., 5]
        velx = torch.cat(
            [vel_left[..., :1], 0.5 * (vel_left[..., 1:] + vel_right[..., :-1]), vel_right[..., -1:]],
            dim=-1,
        )
        vely = torch.cat(
            [vel_bottom[..., :1, :], 0.5 * (vel_bottom[..., 1:, :] + vel_top[..., :-1, :]), vel_top[..., -1:, :]],
            dim=-2,
        )
        psi = nodal_psi_from_corners(cells[..., 6], cells[..., 7], cells[..., 8], cells[..., 9])
        return cls(sdf, temperature, velx, vely, psi, phi)


@dataclass
class Nucleus2MoEDivFreeOutput:
    """Prediction of :class:`Nucleus2MoEDivFree`, each field on its natural grid.

    Shapes match :class:`Nucleus2MoEDivFreeInput`; all fields are in normalized
    units. ``moe_outputs`` is the per-block MoE routing output used for the
    auxiliary / router losses.
    """
    sdf: torch.Tensor
    temperature: torch.Tensor
    velx: torch.Tensor
    vely: torch.Tensor
    psi: torch.Tensor
    phi: torch.Tensor
    # The predicted divergence source (cell-centered, before gating): the RHS of the
    # Poisson solve for phi. Supervised against the divergence of the target velocity.
    div_source: torch.Tensor
    moe_outputs: list

    def to_cell_tensor(self) -> torch.Tensor:
        """Split the fields onto the dataset's 11 cell-centered channels, shape
        ``(..., H, W, 11)`` (see :func:`divfree_fields_to_cells`)."""
        return divfree_fields_to_cells(
            self.sdf, self.temperature, self.velx, self.vely, self.psi, self.phi
        )


@register_model("nucleus2_moe_divfree")
@torch.compile(fullgraph=True)
class Nucleus2MoEDivFree(nn.Module):    
    config_class = Nucleus2MoEDivFreeConfig
    config_from_dict = staticmethod(_config_from_dict)
    expected_fluid_params = [
        "inv_reynolds", "cpgas", "mugas", "rhogas", "thcogas",
        "stefan", "prandtl", "bulk_temp", "sat_temp"
    ]
    expected_heater_params = ["wallTemp", "xMin", "xMax"]
    expected_global_params = ["gravy"]
    # Cell-centered channels the patch embedding consumes: sdf, temperature, and the
    # x/y-face velocities split onto their two bordering cells. psi/phi are NOT fed in
    # -- they are predicted (via the potential heads) and reconstructed into velocity,
    # but the input is only sdf, temperature, and velocity. See divfree_input_to_cells.
    expected_fields = [
        "dfun", "temperature",
        "vel_left", "vel_right", "vel_bottom", "vel_top",
    ]
    num_sim_params = len(expected_fluid_params) + len(expected_heater_params) + len(expected_global_params)
    layout = "t h w c"

    def __init__(self, config: Nucleus2MoEDivFreeConfig):
        super().__init__()
        self.config = config
        n_fields = len(self.expected_fields)

        self.embed_dtype = get_dtype(config.embed_dtype)
        self.debed_dtype = get_dtype(config.debed_dtype)

        self.rotary_emb = RotaryEmbedding(
            dim=(config.embed_dim // config.num_heads) // 3,
            freqs_for="pixel",
            max_freq=256,
            seq_before_head_dim=True
        )

        drop_path_probs = torch.linspace(0.0, 0.1, config.processor_blocks)
        self.blocks = nn.ModuleList([
            TransformerMoEBlock(
                config=config,
                num_sim_params=self.num_sim_params,
                drop_path_prob=drop_path_probs[idx].item(),
            )
            for idx in range(config.processor_blocks)
        ])

        self.out_norm = nn.RMSNorm(config.embed_dim, dtype=self.debed_dtype)

        assert config.patching in ("Linear", "Adaptive")
        if config.patching == "Linear":
            self.embed = LinearEmbed(
                patch_size=config.patch_size,
                in_channels=n_fields,
                embed_dim=config.embed_dim,
                dtype=self.embed_dtype
            )
            self.debed = LinearDebed(
                patch_size=config.patch_size,
                embed_dim=config.embed_dim,
                out_channels=2,
                dtype=self.debed_dtype
            )
        else:
            self.embed = AdaptiveEmbed(
                in_channels=n_fields,
                out_channels=config.embed_dim,
                out_shape=(16, 16),
                dtype=config.embed_dtype
            )
            self.debed = AdaptiveDebed(
                in_channels=config.embed_dim,
                out_channels=2,
                patch_shape=(16, 16),
                dtype=config.debed_dtype
            )

        # Zero-init the output head so the predicted residuals start at zero, rather
        # than being extremely noisy
        #nn.init.zeros_(self.debed.linear.weight)

        # psi is debedded onto the nodal (H+1, W+1) grid; its curl is the divergence-free
        # velocity part. The divergent part is predicted as a cell-centered *divergence
        # source*: it is gated to the interface band and used as the right-hand side of a
        # Poisson solve for a potential phi, whose gradient is added to curl(psi). Gating
        # the source (not the velocity) makes div(velocity) = gate * source, so the
        # divergence is exactly zero outside the band while the potential flow still
        # extends smoothly into the bulk. The source -> velocity map integrates (Poisson)
        # rather than differentiates, so it carries no 1/dx amplification.
        self.debed_psi = OverlappingPatchDebed(
            patch_size=config.patch_size,
            embed_dim=config.embed_dim,
            out_channels=1,
            dtype=self.debed_dtype,
            output_padding=1,
        )
        self.debed_div = OverlappingPatchDebed(
            patch_size=config.patch_size,
            embed_dim=config.embed_dim,
            out_channels=1,
            dtype=self.debed_dtype,
        )

        # Zero-init only the psi head so curl(psi) starts at ~0 (no amplified grid-scale
        # noise from curling a random streamfunction). The divergence-source head keeps
        # its default init: it maps to velocity through the smoothing Poisson solve (no
        # 1/dx amplification), so a nonzero start is well-behaved and avoids the slow ramp.
        nn.init.zeros_(self.debed_psi.conv_transpose.weight)
        nn.init.zeros_(self.debed_div.conv_transpose.weight)

    def get_extra_state(self):
        return {"model_name": getattr(self, "_model_name", None), "config": _config_to_dict(self.config)}

    def set_extra_state(self, state):
        self._model_name = state.get("model_name")
        self.config = _config_from_dict(state["config"])

    def forward(self, batch: DivFreeBatch, normalizer) -> "Nucleus2MoEDivFreeOutput":
        input = Nucleus2MoEDivFreeInput(
            batch.sdf,
            batch.temperature,
            batch.velx,
            batch.vely,
            batch.psi,
            batch.phi
        )
        return self.step(input, batch.sim_params_tensor, normalizer)

    def step(
        self,
        input,
        sim_params: torch.Tensor = None,
        normalizer=None,
        use_sdf_reinit: bool = False,
        use_div_gate: bool = False,
        use_mass_transfer: bool = False,
        sim_params_dict: Optional[dict] = None,
    ):
        # Accept either a DivFreeBatch (which carries the input fields and the
        # sim-parameter tensor) or a bare field container -- Nucleus2MoEDivFreeInput
        # or DivFreeData -- with sim_params passed separately. Both field containers
        # expose the same six natural-grid fields.
        if isinstance(input, DivFreeBatch):
            sim_params = input.sim_params_tensor
            # The physical sim-parameter dict continuity needs, if not passed explicitly.
            if sim_params_dict is None and input.sim_params:
                sim_params_dict = input.sim_params[0]
            input = input.input
        assert sim_params.dtype == torch.float32, f"expected float32, got {sim_params.dtype}"

        # Split sdf, temperature, and the face velocities onto the 6 cell channels the
        # patch embedding consumes. psi/phi are not part of the input -- they are
        # predicted and reconstructed into velocity downstream.
        cells = divfree_input_to_cells(
            input.sdf, input.temperature, input.velx, input.vely
        )
        assert cells.dtype == torch.float32, f"expected float32, got {cells.dtype}"

        _, _, h, w, _ = cells.shape

        with record_function("encode"):
            x = embed = self.embed(cells.to(self.embed_dtype))

        x = x.to(get_dtype(self.config.activation_dtype))

        with record_function("get_axial_freqs"):
            with torch.no_grad():
                _, embed_t, embed_h, embed_w, _ = embed.shape
                rotary_freqs = self.rotary_emb.get_axial_freqs(embed_t, embed_h, embed_w)[None, :, :, :, None, :]

        moe_outputs = []
        for idx, blk in enumerate(self.blocks):
            with record_function(f"block_{idx}"):
                x, moe_output = blk(x, rotary_freqs, sim_params)
                moe_outputs.append(moe_output)

        with record_function("debed"):
            x = self.out_norm(x.to(self.debed_dtype) + embed.to(self.debed_dtype))
            
            sdf, temp = self.debed(x, target_shape=(h, w)).unbind(-1)

            sdf_physical = normalizer.unnormalize_sdf(sdf)
            if use_sdf_reinit:
                sdf_physical = sdf_reinit_sussman(sdf_physical, GRADIENT_SPACING, n_iter=5, near_threshold=0.1)

            psi_nodal = self.debed_psi(x, target_shape=(h + 1, w + 1))[..., 0]
            psi_for_recon = scale_gradient(psi_nodal, self.config.potential_grad_scale)
            psi_nodal_physical = normalizer.unnormalize_psi(psi_for_recon)
            velfacex_sol, velfacey_sol = curl_faces_from_nodes(
                psi_nodal_physical, GRADIENT_SPACING, GRADIENT_SPACING
            )
            velfacex_sol = normalizer.normalize_velx(velfacex_sol)
            velfacey_sol = normalizer.normalize_vely(velfacey_sol)

            if use_mass_transfer:
                # Replace the learned divergence source with the physics source from the
                # Stefan condition (mdot * n.grad(rho)), derived from the predicted temp/sdf.
                assert sim_params_dict is not None, "use_mass_transfer requires sim_params_dict"
                div_source = self._continuity_div_source(
                    temp.to(torch.float64), sdf_physical.to(torch.float64), sim_params_dict, normalizer)
            else:
                div_source = self.debed_div(x, target_shape=(h, w))[..., 0]
            gated_source = div_source
            if use_div_gate:
                div_gate = band_mask(sdf_physical, 3.0 * GRADIENT_SPACING).to(self.debed_dtype)
                gated_source = div_gate * div_source
            # div_source is used as a training target, so this downweights the contribution of
            # phi to its gradient.
            gated_source_for_recon = scale_gradient(gated_source, self.config.potential_grad_scale)
            # NOTE: using float64 improves divergence-free results drastically.
            phi = solve_poisson_neumann_dirichlet(gated_source.to(torch.float64), GRADIENT_SPACING, GRADIENT_SPACING)
            velfacex_dil, velfacey_dil = grad_faces_from_centers(phi, GRADIENT_SPACING, GRADIENT_SPACING)

            phi = normalizer.normalize_phi(phi)
            velfacex_dil = normalizer.normalize_velx(velfacex_dil).to(torch.float32)
            velfacey_dil = normalizer.normalize_vely(velfacey_dil).to(torch.float32)

            velfacex = velfacex_sol + velfacex_dil
            velfacey = velfacey_sol + velfacey_dil

        return Nucleus2MoEDivFreeOutput(
            sdf=sdf.to(torch.float32),
            temperature=temp.to(torch.float32),
            velx=velfacex.to(torch.float32),
            vely=velfacey.to(torch.float32),
            psi=psi_nodal.to(torch.float32),
            phi=phi.to(torch.float32),
            div_source=div_source.to(torch.float32),
            moe_outputs=moe_outputs,
        )

    def _continuity_div_source(self, temp, sdf_physical, sim_params_dict: dict, normalizer):
        bulk_temp = sim_params_dict["bulk_temp"]
        heater_temp = sim_params_dict["heater"]["wallTemp"]
        scale = heater_temp - bulk_temp
        # non-dimensionalize temperatures
        temp_nd = (normalizer.unnormalize_temp(temp, bulk_temp) - bulk_temp) / scale
        sat_temp_nd = (sim_params_dict["sat_temp"] - bulk_temp) / scale
        return continuity(
            temp_nd, sdf_physical,
            sat_temp=sat_temp_nd,
            dx=GRADIENT_SPACING, dy=GRADIENT_SPACING,
            stefan=sim_params_dict["stefan"],
            reynolds=1.0 / sim_params_dict["inv_reynolds"],
            prandtl=sim_params_dict["prandtl"],
            thermal_conductivity=sim_params_dict["thcogas"],
            rhogas=sim_params_dict["rhogas"],
            wall_temp=1.0,  # heater temperature on the non-dimensional scale
        )

    def _normalized_sim_params(self, sim_params_dict: dict, normalizer, device, batch_size: int) -> torch.Tensor:
        # Assemble the normalized conditioning tensor the network expects from the
        # physical sim-parameter dict, ordered to match the expected_* fields and
        # broadcast to the rollout batch size.
        normalized = normalizer.normalize_params([sim_params_dict])[0]
        values = (
            [normalized[param] for param in self.expected_fluid_params]
            + [normalized["heater"][param] for param in self.expected_heater_params]
            + [normalized[param] for param in self.expected_global_params]
        )
        sim_params = torch.tensor(values, device=device, dtype=torch.float32)
        return sim_params[None, :].expand(batch_size, -1).contiguous()

    def _trajectory_window_to_input(self, window: Trajectory, normalizer, bulk_temp) -> "Nucleus2MoEDivFreeInput":
        # Every field -- including the potentials psi/phi -- is tracked on the
        # trajectory and fed back autoregressively, so the window is just normalized
        # into the model's six-field input (no Helmholtz recomputation here).
        return Nucleus2MoEDivFreeInput(
            sdf=normalizer.normalize_sdf(window.sdf),
            temperature=normalizer.normalize_temp(window.temp, bulk_temp),
            velx=normalizer.normalize_velx(window.velx),
            vely=normalizer.normalize_vely(window.vely),
            psi=normalizer.normalize_psi(window.psi),
            phi=normalizer.normalize_phi(window.phi),
        )

    def _predicted_fields(self, output: "Nucleus2MoEDivFreeOutput", normalizer, bulk_temp, output_time_window_size):
        # Unnormalize the model output to physical units on its natural grids and
        # keep the last output_time_window_size frames of every tracked field. psi/phi
        # are tracked too so they can be fed back as the next input.
        keep = slice(-output_time_window_size, None)
        return (
            normalizer.unnormalize_sdf(output.sdf)[:, keep],
            normalizer.unnormalize_temp(output.temperature, bulk_temp)[:, keep],
            normalizer.unnormalize_velx(output.velx)[:, keep],
            normalizer.unnormalize_vely(output.vely)[:, keep],
            normalizer.unnormalize_psi(output.psi)[:, keep],
            normalizer.unnormalize_phi(output.phi)[:, keep],
        )

    def forward_trajectory(
        self,
        trajectory: Trajectory,
        normalizer,
        dx: float,
        input_time_window_size: int,
        output_time_window_size: int,
        trajectory_steps: int,
        use_sdf_reinit: bool = False,
        use_div_gate: bool = True,
        use_mass_transfer: bool = True,
        return_moe_outputs: bool = False,
        clip_temp: bool = False,
    ) -> Trajectory:
        # trajectory carries all six fields on their natural grids. The potentials
        # psi/phi are part of the model's autoregressive state: they are fed back
        # from the previous step's output, not recomputed from the velocity. Returns
        # the rolled Trajectory.
        assert input_time_window_size <= trajectory.num_steps

        # Bootstrap psi/phi for the initial window from its velocities; every later
        # window uses the model's own predicted potentials.
        if trajectory.psi is None or trajectory.phi is None:
            psi, phi = helmholtz_from_faces(
                trajectory.velx, trajectory.vely, GRADIENT_SPACING, GRADIENT_SPACING
            )
            trajectory = replace(trajectory, psi=psi, phi=phi)

        sim_params_dict = trajectory.sim_params[0]
        bulk_temp = sim_params_dict["bulk_temp"]
        sim_params = self._normalized_sim_params(
            sim_params_dict, normalizer, trajectory.sdf.device, trajectory.sdf.shape[0]
        )

        trajectory_moe_outputs = [] if return_moe_outputs else None

        for _ in range(input_time_window_size, trajectory_steps, output_time_window_size):
            model_input = self._trajectory_window_to_input(
                trajectory.last(input_time_window_size), normalizer, bulk_temp
            )
            output = self.step(
                model_input, sim_params, normalizer,
                use_sdf_reinit=use_sdf_reinit, use_div_gate=use_div_gate,
                use_mass_transfer=use_mass_transfer, sim_params_dict=sim_params_dict,
            )
            sdf, temp, velx, vely, psi, phi = self._predicted_fields(
                output, normalizer, bulk_temp, output_time_window_size
            )

            if use_sdf_reinit:
                sdf = sdf_reinit_sussman(sdf, dx=dx, n_iter=5, near_threshold=0.1)

            if clip_temp:
                temp = clip_temp_by_phase(
                    temp, sdf,
                    sim_params_dict["sat_temp"],
                    sim_params_dict["heater"]["wallTemp"],
                )

            trajectory = trajectory.extend(sdf, temp, velx, vely, psi, phi)
            if return_moe_outputs:
                trajectory_moe_outputs.append(output.moe_outputs)

        if return_moe_outputs:
            return trajectory, trajectory_moe_outputs
        return trajectory

class DivFreeForecastModule(MoEConditionedForecastModule):
    """Training module for the divergence-free model.

    Consumes a :class:`DivFreeBatch`: the input and target are ``DivFreeData`` with
    every field already on its natural grid (velocity on faces, psi on nodes), so
    the model runs on the input directly and the loss compares the six fields with
    no cell-splitting.
    """

    # Natural-grid fields compared in the loss / logged per-field. velx/vely are
    # the reconstructed face velocities; the rest are predicted directly.
    LOSS_FIELD_NAMES = ("sdf", "temperature", "velx", "vely", "psi", "phi")

    # Curriculum: supervise only the directly-predicted potentials (sdf, temperature,
    # psi, phi) first, then add the reconstructed velocities once those are decent.
    # The reconstructed-velocity loss backprops through the curl/grad operators
    # (a ~1/dx amplification), so gating it out early lets the potentials settle
    # before that stiff gradient path is switched on. velx/vely are still logged
    # per-field throughout so their MAE is visible before they enter the loss.
    VELOCITY_LOSS_START_STEP = 10000

    def __init__(self, checkpoint_path=None, *args, **kwargs):
        # ModuleBase reads checkpoint_path as "rebuild the exact model saved in the
        # checkpoint and load its full state". Here we instead warm-start from an
        # *unconstrained* base model (nucleus2_moe): it shares the transformer trunk
        # but has a different embed/debed (4 vs 11 input channels, 4 vs 2 output
        # channels) and lacks the psi/phi heads. Pass None down so the base class
        # builds a fresh divfree model, then transfer the trunk weights ourselves.
        super().__init__(None, *args, **kwargs)
        if checkpoint_path is not None:
            self._warm_start_from_base(checkpoint_path)
            # Keep the base checkpoint recorded for reproducibility -- super() saved
            # checkpoint_path=None into the hparams to skip its own load path.
            self.checkpoint_path = checkpoint_path
            self.hparams["checkpoint_path"] = checkpoint_path

        # Override ModuleBase's augmentations: each divfree field is augmented on its
        # own natural grid as a single channel (see _augment_field), so only additive
        # noise applies for now -- FieldDropout, which zeroes whole input channels, is
        # left out until the per-field policy is worked out.
        self.augmentations = [LogUniformNoise(0.001, 5e-2, skip_prob=0.1)]

    def _warm_start_from_base(self, base_checkpoint_path: str):
        """Copy every parameter/buffer whose name and shape match the base checkpoint
        into the divfree model, leaving the resized embed/debed and the zero-init
        psi/phi heads untouched. This transfers the shared transformer trunk
        (blocks + out_norm) so training starts from the unconstrained model's learned
        dynamics rather than from scratch."""
        base_state = load_model_state_dict(base_checkpoint_path, map_location="cpu")
        model_state = self.model.state_dict()
        # Skip the base's _extra_state: it carries the base model's name/config, which
        # set_extra_state would use to overwrite the divfree config.
        transferable = {
            key: tensor
            for key, tensor in base_state.items()
            if key != "_extra_state"
            and key in model_state
            and model_state[key].shape == tensor.shape
        }
        self.model.load_state_dict(transferable, strict=False)

        skipped = sorted(
            {key.split(".")[0] for key in model_state if key not in transferable and key != "_extra_state"}
        )
        print(
            f"Warm-started divfree model from {base_checkpoint_path}: "
            f"transferred {len(transferable)}/{len(model_state) - 1} tensors "
            f"(kept init for: {', '.join(skipped) or 'none'})."
        )

    # How often to log the per-head gradient/weight diagnostics. Scalars are cheap;
    # histograms are heavier, so they go at a coarser cadence.
    DEBED_SCALAR_LOG_INTERVAL = 100
    DEBED_HISTOGRAM_LOG_INTERVAL = 500

    def _debed_heads(self) -> Dict[str, nn.Module]:
        """The output heads whose gradients/weights are logged separately: the
        sdf/temperature head, the streamfunction head (curled into velocity, so it
        carries the 1/dx amplification), and the divergence-source head."""
        return {
            "debed": self.model.debed,
            "debed_psi": self.model.debed_psi,
            "debed_div": self.model.debed_div,
        }

    def _wandb_run(self):
        """The active wandb run if a WandbLogger is attached, else None. Histograms
        go through the run directly since Lightning's self.log only takes scalars."""
        logger = self.logger
        if isinstance(logger, WandbLogger):
            return logger.experiment
        return None

    def _debed_diagnostics(self, include_scalars: bool, include_histograms: bool):
        """Per-head gradient and weight diagnostics as (scalar_log, histogram_log).
        Grad stats span every parameter of the head; the weight distribution is the
        >=2D weight tensors only (excludes 1D biases)."""
        scalar_log: dict = {}
        histogram_log: dict = {}
        for name, head in self._debed_heads().items():
            grads = [p.grad.detach().flatten() for p in head.parameters() if p.grad is not None]
            weights = [p.detach().flatten() for p in head.parameters() if p.ndim >= 2]

            if grads:
                grad_vector = torch.cat(grads)
                if include_scalars:
                    scalar_log[f"train/{name}/grad_norm"] = grad_vector.norm()
                    scalar_log[f"train/{name}/grad_absmax"] = grad_vector.abs().max()
                if include_histograms:
                    histogram_log[f"train/{name}/grad_hist"] = wandb.Histogram(
                        grad_vector.float().cpu().numpy()
                    )
            if weights:
                weight_vector = torch.cat(weights)
                if include_scalars:
                    scalar_log[f"train/{name}/weight_std"] = weight_vector.std()
                    scalar_log[f"train/{name}/weight_absmax"] = weight_vector.abs().max()
                if include_histograms:
                    histogram_log[f"train/{name}/weight_hist"] = wandb.Histogram(
                        weight_vector.float().cpu().numpy()
                    )
        return scalar_log, histogram_log

    def on_before_optimizer_step(self, optimizer):
        # Gradients are populated here (after backward, before the optimizer step).
        super().on_before_optimizer_step(optimizer)
        if not self.trainer.is_global_zero:
            return

        log_scalars = self.global_step % self.DEBED_SCALAR_LOG_INTERVAL == 0
        run = self._wandb_run()
        log_histograms = run is not None and self.global_step % self.DEBED_HISTOGRAM_LOG_INTERVAL == 0
        if not (log_scalars or log_histograms):
            return

        scalar_log, histogram_log = self._debed_diagnostics(log_scalars, log_histograms)
        if scalar_log:
            # Match the base class's grad-norm log: per-step, no epoch reduction. Without
            # these, self.log applies its non-training-step defaults (on_epoch=True) and
            # the scalars get bucketed to epoch boundaries instead of logged per step.
            self.default_log_dict(scalar_log, on_step=True, on_epoch=False)
        if histogram_log:
            # commit=False (and no explicit step) appends the histograms to wandb's
            # current step without advancing its counter. Passing step=global_step here
            # instead races Lightning's own step-committed logging: it jumps wandb ahead
            # and drops the metrics Lightning had buffered for the preceding steps.
            run.log(histogram_log, commit=False)

    @staticmethod
    def _model_input(data: DivFreeData) -> Nucleus2MoEDivFreeInput:
        return Nucleus2MoEDivFreeInput(
            sdf=data.sdf, temperature=data.temperature,
            velx=data.velx, vely=data.vely, psi=data.psi, phi=data.phi,
        )

    def _per_field_mae(self, output: Nucleus2MoEDivFreeOutput, target: DivFreeData, prefix: str) -> dict:
        """Mean absolute error of each field (normalized units), so the psi/phi
        contribution can be separated from sdf/temperature/velocity. Logged for
        every field regardless of which ones currently enter the loss."""
        metrics = {
            f"{prefix}/mae_{name}": (getattr(output, name) - getattr(target, name)).abs().mean()
            for name in self.LOSS_FIELD_NAMES
        }
        metrics[f"{prefix}/mae_div_source"] = (
            output.div_source - self._target_div_source(target)
        ).abs().mean()
        return metrics

    def _target_div_source(self, target: DivFreeData) -> torch.Tensor:
        # The true divergence source is the divergence of the target velocity. The head
        # predicts it as the RHS of a *physical*-spacing Poisson solve, so it is a
        # physical divergence; the target velocities are normalized, so scale by vel_std
        # (div(unnormalized velocity) = vel_std * div(normalized velocity)).
        return self.normalizer.vel_std * divergence_centers_from_faces(
            target.velx, target.vely, GRADIENT_SPACING, GRADIENT_SPACING
        )

    def _loss_fields(self, output: Nucleus2MoEDivFreeOutput, target: DivFreeData):
        names = ["sdf", "temperature", "velx", "vely"]
        #if self.global_step >= self.VELOCITY_LOSS_START_STEP:
        #    names += ["velx", "vely"]
        pred_grids = [getattr(output, name) for name in names]
        target_grids = [getattr(target, name) for name in names]
        # The divergence source has no field on the DivFreeData target -- its target is
        # the divergence of the target velocity.
        pred_grids.append(output.div_source)
        target_grids.append(self._target_div_source(target))
        return pred_grids, target_grids

    def _field_loss(self, output: Nucleus2MoEDivFreeOutput, target: DivFreeData) -> torch.Tensor:
        pred_grids, target_grids = self._loss_fields(output, target)
        total_abs_error = sum(
            (pred_grid - target_grid).abs().sum()
            for pred_grid, target_grid in zip(pred_grids, target_grids)
        )
        total_elements = sum(pred_grid.numel() for pred_grid in pred_grids)
        return total_abs_error / total_elements
    
    def _sdf_sign_loss(self, output: Nucleus2MoEDivFreeOutput, target: DivFreeData):
        r""" An extra penalty that the sign of the SDF matches the target. This
        is important near the interface, since an incorrect sign may still be "close"
        to the ground-truth in terms of MAE.
        """
        alpha = 0.01
        loss_interface_weight = torch.exp(-abs(output.sdf))
        elem_loss = torch.nn.functional.softplus(-alpha * sdf_true * sdf_pred) * loss_interface_weight
        return elem_loss.mean(dim=(-3, -2, -1)).mean()

    def _augment_field(self, field: torch.Tensor) -> torch.Tensor:
        # self.augmentations (LogUniformNoise, FieldDropout) expect a channels-last
        # (B, T, H, W, C) tensor. The divfree fields live on different natural grids,
        # so each is augmented on its own as a single channel -- noise and dropout are
        # applied to every field individually.
        augmented = field.unsqueeze(-1)
        for augmentation in self.augmentations:
            augmented = augmentation(augmented)
        return augmented.squeeze(-1)

    def _augment(self, data: DivFreeData) -> DivFreeData:
        # Apply the augmentations to every input field on its natural grid.
        return data._apply(self._augment_field)

    def training_step(self, batch: DivFreeBatch, batch_idx: int) -> torch.Tensor:
        torch.compiler.cudagraph_mark_step_begin()
        model_input = self._model_input(self._augment(batch.input))
        output = self.model.step(model_input, batch.sim_params_tensor, self.normalizer)

        data_loss = self._field_loss(output, batch.target)

        aux_loss, router_has_loss = self._router_loss(output.moe_outputs)
        loss = data_loss + aux_loss

        self._update_router_bias(output.moe_outputs)

        log_dict = {
            "train/loss": loss,
            "train/data_loss": data_loss,
            "train/step": self.global_step,
            "train/learning_rate": self.get_current_lr(),
        }
        log_dict |= self._per_field_mae(output, batch.target, "train")
        log_dict = self._moe_metrics(output.moe_outputs, log_dict, "train")
        self.default_log_dict(log_dict)
        return loss

    def validation_step(self, batch: DivFreeBatch, batch_idx: int) -> torch.Tensor:
        output = self.model.step(self._model_input(batch.input), batch.sim_params_tensor, self.normalizer)
        loss = self._field_loss(output, batch.target)

        # Re-stack onto the dataset's cell channels for plotting / tensor metrics.
        pred_cells = output.to_cell_tensor()
        target_cells = self._model_input(batch.target).to_cell_tensor()
        if batch_idx == 0:
            input_cells = self._model_input(batch.input).to_cell_tensor()
            self.validation_sample = (input_cells.detach(), target_cells.detach(), pred_cells.detach())

        log_dict = {"val/loss": loss}
        log_dict |= self._per_field_mae(output, batch.target, "val")
        log_dict = self.log_step_metrics(
            log_dict, pred_cells, target_cells, batch.dx[0].item(), batch.dy[0].item(), "val"
        )
        log_dict = self._moe_metrics(output.moe_outputs, log_dict, "val")
        self.default_log_dict(log_dict)
        return loss
