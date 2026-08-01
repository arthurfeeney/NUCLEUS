import dataclasses
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import record_function
from rotary_embedding_torch import RotaryEmbedding
from typing import Literal, Optional

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
from nucleus.models.modules import MoEConditionedForecastModule
from nucleus.utils.sdf_reinit import sdf_reinit_sussman
from nucleus.utils.inf_stabilizer import clip_temp_by_phase
from nucleus.physics.poisson import (
    solve_poisson_neumann_dirichlet,
    reconstruct_velocity_from_helmholtz,
    divergence_centers_from_faces,
    grad_faces_from_centers,
)
from nucleus.physics.sdf import band_mask

from ._api import register_model

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
GRADIENT_SPACING = 1 / 32

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


def _cell_centers(num_cells, low, high, device, dtype):
    step = (high - low) / num_cells
    return low + (torch.arange(num_cells, device=device, dtype=dtype) + 0.5) * step


def _smoothstep(t):
    # Smooth (C1) 0 -> 1 ramp on [0, 1], clamped outside.
    t = torch.clamp(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


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
    # Cell-centered channels emitted by InMemDivFreeForecastDataset: the x-face
    # velocity split into left/right, the y-face velocity into bottom/top, and the
    # nodal streamfunction psi into its four cell corners (see the dataset's
    # _split_to_cells). The model both consumes and predicts all eleven.
    expected_fields = [
        "dfun", "temperature",
        "vel_left", "vel_right", "vel_bottom", "vel_top",
        "psi_bottom_left", "psi_bottom_right", "psi_top_left", "psi_top_right",
        "phi",
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
        nn.init.zeros_(self.debed.linear.weight)

        # Overlapping debeds for the differentiated fields (curled/grad'd into the
        # velocity, so they need a smoother reconstruction than the linear
        # sdf/temperature head). psi is debedded directly onto the nodal
        # (H+1, W+1) grid via output_padding=1 -- the earlier four-corner-average
        # scheme left 3/4 of the psi output channels in an untrained null-space
        # (only the corner mean received a gradient). phi stays cell-centered (H, W).
        self.debed_psi = OverlappingPatchDebed(
            patch_size=config.patch_size,
            embed_dim=config.embed_dim,
            out_channels=1,
            dtype=self.debed_dtype,
            output_padding=1,
        )
        self.debed_phi = OverlappingPatchDebed(
            patch_size=config.patch_size,
            embed_dim=config.embed_dim,
            out_channels=1,
            dtype=self.debed_dtype,
        )

        # Zero-init the psi/phi heads so both potentials start at the normalized
        # mean. The reconstructed velocity then begins at ~0 (a small, smooth field)
        # rather than the amplified grid-scale noise that curling a random psi would
        # produce -- keeping the early-training gradients through the curl/grad
        # operators well-conditioned.
        nn.init.zeros_(self.debed_psi.conv_transpose.weight)
        nn.init.zeros_(self.debed_phi.conv_transpose.weight)

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
    ):
        # Accept either a DivFreeBatch (which carries the input fields and the
        # sim-parameter tensor) or a bare field container -- Nucleus2MoEDivFreeInput
        # or DivFreeData -- with sim_params passed separately. Both field containers
        # expose the same six natural-grid fields.
        if isinstance(input, DivFreeBatch):
            sim_params = input.sim_params_tensor
            input = input.input
        assert sim_params.dtype == torch.float32, f"expected float32, got {sim_params.dtype}"

        # Split the natural-grid input fields onto the dataset's 11 cell channels
        # that the patch embedding consumes.
        cells = divfree_fields_to_cells(
            input.sdf, input.temperature, input.velx, input.vely, input.psi, input.phi
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
            sdf_delta, temp_delta = self.debed(x, target_shape=(h, w)).unbind(-1)

            sdf = input.sdf.to(self.debed_dtype) + sdf_delta
            temp = input.temperature.to(self.debed_dtype) + temp_delta

            # Allow the divergence source only within 5 cells of the interface and
            # zero elsewhere. 
            # Normalization shifts the sdf zero level-set off zero, so denormalize first.
            div_gate = band_mask(
                normalizer.unnormalize_sdf(sdf), 5.0 * GRADIENT_SPACING
            ).to(self.debed_dtype)

            # psi directly on the nodal (H+1, W+1) grid; phi cell-centered (H, W).
            # NOTE: we unnormalize psi and phi to physical units for reconstructing velocity.
            psi_nodal = self.debed_psi(x, target_shape=(h + 1, w + 1))[..., 0]
            phi = self.debed_phi(x, target_shape=(h, w))[..., 0]
            phi_physical = clean_phi(normalizer.unnormalize_phi(phi), div_gate)
            psi_nodal_physical = normalizer.unnormalize_psi(psi_nodal)
            
            velfacex, velfacey = reconstruct_velocity_from_helmholtz(
                psi_nodal_physical, phi_physical,
                GRADIENT_SPACING, GRADIENT_SPACING,
            )
            velfacex = normalizer.normalize_velx(velfacex)
            velfacey = normalizer.normalize_vely(velfacey)

        return Nucleus2MoEDivFreeOutput(
            sdf=sdf.to(torch.float32),
            temperature=temp.to(torch.float32),
            velx=velfacex.to(torch.float32),
            vely=velfacey.to(torch.float32),
            psi=psi_nodal.to(torch.float32),
            phi=phi.to(torch.float32),
            moe_outputs=moe_outputs,
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

    def _normalize_cells(self, cells: torch.Tensor, normalizer, bulk_temp) -> "Nucleus2MoEDivFreeInput":
        # Normalize a physical 11-channel cell tensor per field on its natural grid.
        # The DivFree normalizer is per-field (sdf, temp, velx, vely, psi, phi each
        # on its own scale), so it is applied to the reconstructed grids rather than
        # blindly to the raw cell channels. Returns the normalized fields as the
        # Nucleus2MoEDivFreeInput the step consumes.
        fields = Nucleus2MoEDivFreeInput.from_cell_tensor(cells)
        return Nucleus2MoEDivFreeInput(
            sdf=normalizer.normalize_sdf(fields.sdf),
            temperature=normalizer.normalize_temp(fields.temperature, bulk_temp),
            velx=normalizer.normalize_velx(fields.velx),
            vely=normalizer.normalize_vely(fields.vely),
            psi=normalizer.normalize_psi(fields.psi),
            phi=normalizer.normalize_phi(fields.phi),
        )

    def _unnormalize_output(self, output: "Nucleus2MoEDivFreeOutput", normalizer, bulk_temp) -> torch.Tensor:
        # Unnormalize the model output per field (natural grids) and split back to a
        # physical 11-channel cell tensor for the rolled trajectory.
        return divfree_fields_to_cells(
            normalizer.unnormalize_sdf(output.sdf),
            normalizer.unnormalize_temp(output.temperature, bulk_temp),
            normalizer.unnormalize_velx(output.velx),
            normalizer.unnormalize_vely(output.vely),
            normalizer.unnormalize_psi(output.psi),
            normalizer.unnormalize_phi(output.phi),
        )
        
    def forward_trajectory(
        self,
        initial_state: torch.Tensor,
        sim_params_dict: dict,
        normalizer,
        dx: float,
        input_time_window_size: int,
        output_time_window_size: int,
        trajectory_steps: int,
        use_sdf_reinit: bool = False,
        return_moe_outputs: bool = False,
        clip_temp: bool = False,
    ) -> torch.Tensor:
        # initial_state carries the physical fields on their natural grids; the
        # rolled trajectory is kept as the dataset's 11-channel cell tensor.
        trajectory = initial_state.to_cell_tensor()
        assert trajectory.dim() == 5, "initial state fields must have shape [B, T, H, W]"
        assert input_time_window_size <= trajectory.shape[1]

        bulk_temp = sim_params_dict["bulk_temp"]
        sim_params = self._normalized_sim_params(
            sim_params_dict, normalizer, trajectory.device, trajectory.shape[0]
        )

        trajectory_moe_outputs = [] if return_moe_outputs else None

        for _ in range(input_time_window_size, trajectory_steps, output_time_window_size):
            normalized_window = self._normalize_cells(
                trajectory[:, -input_time_window_size:], normalizer, bulk_temp
            )
            output = self.step(normalized_window, sim_params, normalizer)
            pred = self._unnormalize_output(output, normalizer, bulk_temp)
            output_time_window = pred[:, -output_time_window_size:]

            if use_sdf_reinit:
                output_time_window[..., 0] = sdf_reinit_sussman(output_time_window[..., 0], dx=dx, n_iter=5)

            if clip_temp:
                output_time_window[..., 1] = clip_temp_by_phase(
                    output_time_window[..., 1],
                    output_time_window[..., 0],
                    sim_params_dict["sat_temp"],
                    sim_params_dict["heater"]["wallTemp"],
                )

            trajectory = torch.cat((trajectory, output_time_window), dim=1)
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

    @staticmethod
    def _model_input(data: DivFreeData) -> Nucleus2MoEDivFreeInput:
        """Wrap a batch's DivFreeData fields as the model's input dataclass (the
        same six natural-grid fields)."""
        return Nucleus2MoEDivFreeInput(
            sdf=data.sdf, temperature=data.temperature,
            velx=data.velx, vely=data.vely, psi=data.psi, phi=data.phi,
        )

    def _per_field_mae(self, output: Nucleus2MoEDivFreeOutput, target: DivFreeData, prefix: str) -> dict:
        """Mean absolute error of each field (normalized units), so the psi/phi
        contribution can be separated from sdf/temperature/velocity. Logged for
        every field regardless of which ones currently enter the loss."""
        return {
            f"{prefix}/mae_{name}": (getattr(output, name) - getattr(target, name)).abs().mean()
            for name in self.LOSS_FIELD_NAMES
        }

    def _loss_fields(self, output: Nucleus2MoEDivFreeOutput, target: DivFreeData):
        pred_grids = [output.sdf, output.temperature, output.velx, output.vely]
        target_grids = [target.sdf, target.temperature, target.velx, target.vely]
        #velocity_loss_start_step = 10000
        #if self.global_step >= velocity_loss_start_step:
        #    pred_grids += [output.velx, output.vely]
        #    target_grids += [target.velx, target.vely]
        return pred_grids, target_grids

    def _field_loss(self, output: Nucleus2MoEDivFreeOutput, target: DivFreeData) -> torch.Tensor:
        # L1 error over the active natural-grid fields, summed over every element of
        # every field and divided by the total element count. The fields have
        # different (staggered/nodal) shapes, so they are accumulated separately;
        # summing then dividing weights each element equally.
        pred_grids, target_grids = self._loss_fields(output, target)
        total_abs_error = sum(
            (pred_grid - target_grid).abs().sum()
            for pred_grid, target_grid in zip(pred_grids, target_grids)
        )
        total_elements = sum(pred_grid.numel() for pred_grid in pred_grids)
        return total_abs_error / total_elements

    def training_step(self, batch: DivFreeBatch, batch_idx: int) -> torch.Tensor:
        # NOTE: Data augmentations not applied.
        
        torch.compiler.cudagraph_mark_step_begin()
        output = self.model.step(self._model_input(batch.input), batch.sim_params_tensor, self.normalizer)

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
