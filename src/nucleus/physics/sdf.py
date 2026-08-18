from typing import Tuple

import torch


def _central_difference(field: torch.Tensor, spacing: float, dim: int) -> torch.Tensor:
    """Explicit finite-difference derivative of a cell-centered field along ``dim``:
    a second-order central difference ``(f[i+1] - f[i-1]) / (2h)`` in the interior and
    a first-order one-sided difference at the two boundary cells. Built with
    ``narrow`` + ``cat`` (no in-place writes) so it differentiates and compiles cleanly.
    """
    n = field.size(dim)
    interior = (field.narrow(dim, 2, n - 2) - field.narrow(dim, 0, n - 2)) / (2.0 * spacing)
    first = (field.narrow(dim, 1, 1) - field.narrow(dim, 0, 1)) / spacing
    last = (field.narrow(dim, n - 1, 1) - field.narrow(dim, n - 2, 1)) / spacing
    return torch.cat([first, interior, last], dim=dim)


def interface_normals(
    sdf: torch.Tensor, dx: float, dy: float, eps: float = 1e-12
) -> Tuple[torch.Tensor, torch.Tensor]:
    """normal direction of the SDF, which can be used to get the normal of the
       interface (the zero level set of the SDF).
    Args:
        sdf: cell-centered signed distance, shape ``(..., H, W)``. x is the last
            axis (width) and y the second-to-last (height).
        dx: cell spacing in x.
        dy: cell spacing in y.
        eps: floor on ``|grad(sdf)|``. Where the gradient vanishes -- flat regions
            far from the interface, or local extrema -- the normal is undefined;
            the floor makes those cells return a near-zero vector rather than
            dividing by zero.

    Returns:
        ``(normal_x, normal_y)``, each shape ``(..., H, W)``.
    """
    grad_x = _central_difference(sdf, dx, dim=-1)
    grad_y = _central_difference(sdf, dy, dim=-2)
    magnitude = torch.sqrt(grad_x**2 + grad_y**2).clamp_min(eps)
    return grad_x / magnitude, grad_y / magnitude


def interface_mask(sdf):
    r"""
    Cells adjacent to the zero level set: a cell is marked when any of its four
    neighbors lies in the other phase (the sign of the SDF differs). Works for any
    shape (..., H, W) and stays on the input's device.
    """
    assert sdf.dim() >= 2, "SDF must be of shape (..., H, W)"
    signs = torch.sign(sdf)
    interface = torch.zeros_like(sdf, dtype=torch.bool)

    # Inequality is symmetric, so one comparison per axis marks both neighbors.
    rows_differ = signs[..., :-1, :] != signs[..., 1:, :]
    interface[..., :-1, :] |= rows_differ
    interface[..., 1:, :] |= rows_differ

    cols_differ = signs[..., :, :-1] != signs[..., :, 1:]
    interface[..., :, :-1] |= cols_differ
    interface[..., :, 1:] |= cols_differ
    return interface


def vapor_mask(sdf):
    return sdf >= 0


def liquid_mask(sdf):
    return sdf < 0


def band_mask(sdf: torch.Tensor, band_width: float) -> torch.Tensor:
    """Cells whose center lies within ``band_width`` (a distance, in the SDF's
    units) of the interface, i.e. ``|sdf| <= band_width``. Shape ``(..., H, W)``.
    """
    return sdf.abs() <= band_width


def _upwind_normal_gradient(
    field: torch.Tensor, normal_x: torch.Tensor, normal_y: torch.Tensor, dx: float, dy: float
) -> torch.Tensor:
    """``n . grad(field)`` with first-order upwinding against the normal, so the
    stencil leans toward where the extrapolated information comes from (the
    interface). Shape ``(..., H, W)``."""
    forward_x = torch.zeros_like(field)
    forward_x[..., :, :-1] = (field[..., :, 1:] - field[..., :, :-1]) / dx
    backward_x = torch.zeros_like(field)
    backward_x[..., :, 1:] = (field[..., :, 1:] - field[..., :, :-1]) / dx

    forward_y = torch.zeros_like(field)
    forward_y[..., :-1, :] = (field[..., 1:, :] - field[..., :-1, :]) / dy
    backward_y = torch.zeros_like(field)
    backward_y[..., 1:, :] = (field[..., 1:, :] - field[..., :-1, :]) / dy

    grad_x = torch.where(normal_x > 0, backward_x, forward_x)
    grad_y = torch.where(normal_y > 0, backward_y, forward_y)
    return normal_x * grad_x + normal_y * grad_y


def constant_normal_extrapolation(
    field: torch.Tensor,
    fill_mask: torch.Tensor,
    normal_x: torch.Tensor,
    normal_y: torch.Tensor,
    dx: float,
    dy: float,
    tolerance: float = 1e-6,
    max_iterations: int = 4,
) -> torch.Tensor:
    """Constant extrapolation of ``field`` along the interface normals into
    ``fill_mask``, iterated to steady state.

    Marches ``d field / d tau + (n . grad field) = 0`` in pseudo-time on the fill
    region with first-order upwinding (Aslam extrapolation). At steady state
    ``n . grad field = 0``, so ``field`` becomes constant along each normal -- the
    value at the interface is carried straight out into the fill region. ``field``
    is held fixed outside ``fill_mask`` (the source phase), and those values feed
    the ``+n`` march into the fill region. This is how a one-sided interface flux
    is spread over a multi-cell band, matching the source band Flash-X's
    extrapolation produces.

    The pseudo-time march is run until the largest per-step change on the fill
    region falls below ``tolerance`` (relative to the field's peak magnitude), i.e.
    until steady state, rather than a fixed step count. Restrict ``fill_mask`` to a
    band around the interface so the iteration converges over that band alone
    instead of flooding the whole phase. The first-order upwind stencil reads
    toward the interface, so band cells never depend on the frozen cells beyond the
    band.

    To march in the ``-n`` direction (extrapolating the vapor flux into the liquid)
    pass the negated normals.

    Args:
        field: quantity to extrapolate, shape ``(..., H, W)``. Meaningful on the
            source phase; overwritten on ``fill_mask``.
        fill_mask: boolean mask of cells to fill (the opposite phase, usually
            intersected with an interface band), shape ``(..., H, W)``.
        normal_x: interface normal x-component, shape ``(..., H, W)``.
        normal_y: interface normal y-component, shape ``(..., H, W)``.
        dx: cell spacing in x.
        dy: cell spacing in y.
        tolerance: steady-state threshold on the per-step change, relative to the
            field's peak magnitude.
        max_iterations: cap on pseudo-time steps if steady state is not reached.

    Returns:
        Extrapolated field, shape ``(..., H, W)``.
    """
    time_step = 0.5 * min(dx, dy)
    fill = fill_mask.to(field.dtype)
    extrapolated = field.clone()
    for _ in range(max_iterations):
        normal_gradient = _upwind_normal_gradient(extrapolated, normal_x, normal_y, dx, dy)
        step = time_step * fill * normal_gradient
        extrapolated = extrapolated - step
    return extrapolated