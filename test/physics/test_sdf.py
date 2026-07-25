import torch

from nucleus.physics.sdf import (
    band_mask,
    constant_normal_extrapolation,
    interface_normals,
    smoothed_delta,
    vapor_mask,
)


def _vertical_sdf(height, width, dx, dy):
    """Signed distance for a vertical interface: liquid (sdf < 0) left, vapor right.
    The normal is +x everywhere."""
    x = (torch.arange(width, dtype=torch.float64) + 0.5) * dx
    y = (torch.arange(height, dtype=torch.float64) + 0.5) * dy
    _, grid_x = torch.meshgrid(y, x, indexing="ij")
    x0 = x[width // 2] + 0.5 * dx
    return grid_x - x0


def test_band_mask_widens_with_width():
    dx = dy = 1.0 / 32
    sdf = _vertical_sdf(48, 48, dx, dy)
    narrow = int(band_mask(sdf, 1 * dx).sum())
    wide = int(band_mask(sdf, 3 * dx).sum())
    assert wide > narrow


def test_extrapolation_fills_constant_across_band():
    # A field that is constant in the liquid must stay that constant when
    # constant-extrapolated into the vapor -- n.grad = 0 is satisfied exactly by a
    # constant, so the band fills to the same value.
    dx = dy = 1.0 / 32
    H = W = 64
    sdf = _vertical_sdf(H, W, dx, dy)
    normal_x, normal_y = interface_normals(sdf, dx, dy)
    field = torch.where(sdf < 0, torch.full_like(sdf, 2.0), torch.zeros_like(sdf))

    vapor_band = band_mask(sdf, 3 * dx) & vapor_mask(sdf)
    # iterate to steady state on the vapor band
    filled = constant_normal_extrapolation(field, vapor_band, normal_x, normal_y, dx, dy)

    assert (filled[vapor_band] - 2.0).abs().max() < 1e-3
    # source (liquid) values are untouched
    assert torch.all(filled[sdf < 0] == field[sdf < 0])


def test_smoothed_delta_integrates_to_one_and_is_banded():
    # The regularized delta must integrate to 1 across the normal (so it spreads a
    # surface source without changing its strength) and be supported on the band.
    dx = dy = 1.0 / 32
    H = W = 96
    sdf = _vertical_sdf(H, W, dx, dy)
    half_width = 3 * dx

    delta = smoothed_delta(sdf, half_width)
    # integrate along x (the normal) on any row: sum * dx ~= 1
    row_integral = delta[H // 2, :].sum() * dx
    assert abs(float(row_integral) - 1.0) < 1e-3
    # supported only within the band
    assert torch.all(delta[sdf.abs() >= half_width] == 0)
    assert torch.all(delta[sdf.abs() < half_width] > 0)


def test_extrapolation_stays_within_the_band():
    # Restricting the fill to a band keeps the extrapolation local: cells outside
    # the band are never filled, and the source (liquid) is never overwritten.
    dx = dy = 1.0 / 32
    H = W = 48
    band_width = 5 * dx
    sdf = _vertical_sdf(H, W, dx, dy)
    normal_x, normal_y = interface_normals(sdf, dx, dy)
    field = torch.where(sdf < 0, torch.full_like(sdf, 5.0), torch.zeros_like(sdf))

    vapor_band = band_mask(sdf, band_width) & vapor_mask(sdf)
    filled = constant_normal_extrapolation(field, vapor_band, normal_x, normal_y, dx, dy)

    # vapor beyond the band is never in the fill region -> stays exactly zero
    assert torch.all(filled[sdf > band_width] == 0)
    # the band converged to the source constant
    assert (filled[vapor_band] - 5.0).abs().max() < 1e-3
