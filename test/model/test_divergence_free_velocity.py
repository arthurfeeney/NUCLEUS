import pytest
import torch

from nucleus.models.nucleus2_moe_divfree import (
    dilatational_wall_mask,
    vapor_gate_from_sdf,
    velocity_from_potentials,
)


def pointwise_divergence(velx: torch.Tensor, vely: torch.Tensor) -> torch.Tensor:
    """Discrete divergence with the same central-difference stencil the velocity
    is constructed from (x = last dim, y = second-to-last), so curl(psi) cancels
    to numerical precision. Matches physical_metrics.divergence.
    """
    velx_grad_x = torch.gradient(velx, dim=-1)[0]
    vely_grad_y = torch.gradient(vely, dim=-2)[0]
    return velx_grad_x + vely_grad_y


def interior(field: torch.Tensor, margin: int = 5) -> torch.Tensor:
    return field[..., margin:-margin, margin:-margin]


@pytest.mark.parametrize("batch_size", [1, 3])
def test_bulk_liquid_velocity_is_divergence_free(batch_size):
    torch.manual_seed(0)
    B, T, H, W = batch_size, 2, 48, 48

    psi = torch.randn(B, T, H, W)
    phi = torch.randn(B, T, H, W)
    # gate == 0 everywhere: pure bulk liquid, velocity is exactly curl(psi).
    bulk_gate = torch.zeros(B, T, H, W)

    velx, vely = velocity_from_potentials(psi, phi, bulk_gate)
    divergence = pointwise_divergence(velx, vely)

    assert interior(divergence).abs().max() < 1e-4


def test_potential_region_stays_divergence_free_in_deep_liquid():
    """Deep liquid on the bottom, vapor on top: divergence free in the deep-liquid
    interior even though grad(phi) makes it divergent in the vapor region."""
    torch.manual_seed(1)
    B, T, H, W = 2, 2, 48, 48
    band = 2.0

    psi = torch.randn(B, T, H, W)
    phi = torch.randn(B, T, H, W)

    sdf = torch.full((B, T, H, W), -10.0)
    sdf[..., H // 2:, :] = 5.0
    gate = vapor_gate_from_sdf(sdf, band=band)

    velx, vely = velocity_from_potentials(psi, phi, gate)
    divergence = pointwise_divergence(velx, vely)

    deep_liquid = divergence[..., 5:H // 2 - 2, 5:-5]
    assert deep_liquid.abs().max() < 1e-4

    # Sanity: grad(phi) injects divergence where the gate is on (and away from the
    # walls, where the dilatational mask zeroes it out).
    vapor = divergence[..., H // 2 + 2:-5, 5:-5]
    assert vapor.abs().max() > 1e-2


def test_velocity_is_divergence_free_near_the_walls():
    """Even with vapor everywhere (gate == 1, grad(phi) fully active), the velocity
    is divergence free in the band next to each closed wall, because the
    dilatational part is masked to exactly zero there and only curl(psi) remains.
    """
    torch.manual_seed(2)
    B, T, H, W = 2, 2, 64, 64

    psi = torch.randn(B, T, H, W)
    phi = torch.randn(B, T, H, W)
    gate = torch.ones(B, T, H, W)          # vapor everywhere, incl. the walls

    velx, vely = velocity_from_potentials(psi, phi, gate)
    divergence = pointwise_divergence(velx, vely)

    # First cell against each closed wall (interior span to avoid corners).
    assert divergence[..., 0, W // 4:3 * W // 4].abs().max() < 1e-4   # bottom
    assert divergence[..., H // 4:3 * H // 4, 0].abs().max() < 1e-4    # left
    assert divergence[..., H // 4:3 * H // 4, -1].abs().max() < 1e-4   # right

    # Sanity: grad(phi) is genuinely active in the interior, so the check above is
    # not passing because divergence is zero everywhere.
    assert interior(divergence).abs().max() > 1e-2


def test_free_slip_at_walls():
    """Free-slip: windowing psi drives the wall-normal velocity toward zero at the
    closed walls while leaving the wall-tangential velocity free (comparable to the
    interior), for any potentials."""
    torch.manual_seed(3)
    B, T, H, W = 1, 1, 64, 64

    psi = torch.randn(B, T, H, W)
    phi = torch.randn(B, T, H, W)
    gate = torch.ones(B, T, H, W)          # vapor everywhere, grad(phi) fully active

    velx, vely = velocity_from_potentials(psi, phi, gate)
    velx_interior = velx[..., H // 4:3 * H // 4, W // 4:3 * W // 4].abs().mean()
    vely_interior = vely[..., H // 4:3 * H // 4, W // 4:3 * W // 4].abs().mean()

    bottom_normal = vely[..., 0, W // 4:3 * W // 4].abs().mean()       # bottom: normal = vely
    bottom_tangential = velx[..., 0, W // 4:3 * W // 4].abs().mean()   # bottom: tangential = velx
    left_normal = velx[..., H // 4:3 * H // 4, 0].abs().mean()          # left: normal = velx
    right_normal = velx[..., H // 4:3 * H // 4, -1].abs().mean()        # right: normal = velx

    # Wall-normal velocity is strongly damped at each closed wall...
    assert bottom_normal < 0.4 * vely_interior
    assert left_normal < 0.4 * velx_interior
    assert right_normal < 0.4 * velx_interior
    # ...but the tangential velocity stays free (free-slip, not no-slip).
    assert bottom_tangential > 0.3 * velx_interior
    assert bottom_tangential > 2.0 * bottom_normal


def test_dilatational_wall_mask_vanishes_at_walls_and_open_top():
    mask = dilatational_wall_mask(64, 64, device="cpu", dtype=torch.float32)
    assert mask.shape == (64, 64)

    # Exactly 0 in the band next to each closed wall.
    assert torch.equal(mask[0, 32], torch.tensor(0.0))    # bottom
    assert torch.equal(mask[32, 0], torch.tensor(0.0))    # left
    assert torch.equal(mask[32, -1], torch.tensor(0.0))   # right
    # 1 in the interior and open at the top.
    assert mask[32, 32] > 0.99
    assert mask[-1, 32] > 0.99


def test_vapor_gate_is_exactly_zero_in_deep_liquid():
    band = 2.0
    sdf = torch.tensor([-5.0, -band, -band - 1e-3, -1.0, 0.0, 3.0])
    gate = vapor_gate_from_sdf(sdf, band=band)

    assert torch.equal(gate[sdf <= -band], torch.zeros_like(gate[sdf <= -band]))
    assert torch.equal(gate[sdf >= 0.0], torch.ones_like(gate[sdf >= 0.0]))
    assert torch.all((gate >= 0.0) & (gate <= 1.0))
