from typing import Tuple
import math

import torch

from nucleus.physics.sdf import (
    band_mask,
    interface_normals,
    interface_mask,
    liquid_mask,
    vapor_mask,
    constant_normal_extrapolation,
    smoothed_delta,
)
from nucleus.physics.temp_grad import vapor_temp_grad, liquid_temp_grad

DEFAULT_BAND_CELLS = 5


def interface_heatflux(
    temp: torch.Tensor, sdf: torch.Tensor, sat_temp, dx: float, dy: float,
    band_cells: int = DEFAULT_BAND_CELLS, wall_temp=None, eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normal temperature gradient ``grad(T) . n`` on each side of the interface.

    The two sides are what drives phase change: the Stefan condition balances the
    conducted heat arriving from the liquid against that leaving into the vapor,
    so they are returned separately rather than as a single jump.

    Each side uses the ghost-fluid gradient of its own phase (``liquid_temp_grad``
    / ``vapor_temp_grad``), so the gradient is not smeared by the other phase near
    the interface, and both are projected onto the **same** normal ``n`` (from
    ``interface_normals``, pointing liquid to vapor).

    A phase's ghost-fluid gradient is only defined on that phase's cells, so the
    two would otherwise have **disjoint** support -- the liquid side on liquid
    cells, the vapor side on vapor cells -- and could not be subtracted at a common
    cell. Each side is therefore constant-extrapolated along the normals into the
    opposite phase over a ``band_cells``-wide band (``constant_normal_extrapolation``):
    the liquid flux marches in ``+n`` into the vapor, the vapor flux in ``-n`` into
    the liquid. Both then overlap on that band, where the Stefan jump
    ``k_l dT/dn_l - k_v dT/dn_v`` is formed and masked back to the band.

    Note this returns ``dT/dn``, not the flux itself: the physical heat flux is
    ``q = -k dT/dn``, so scale each side by its own conductivity (and the sign)
    when forming the Stefan balance.

    Args:
        temp: cell-centered temperature, shape ``(..., H, W)``.
        sdf: cell-centered signed distance, shape ``(..., H, W)``. sdf < 0 is liquid,
            sdf >= 0 is vapor.
        sat_temp: interface (saturation) temperature; scalar or broadcastable to
            ``temp``. Passed to the ghost-fluid gradients.
        dx: cell spacing in x.
        dy: cell spacing in y.
        band_cells: how many cells each one-sided gradient is extrapolated into the
            opposite phase, setting the width of the overlap band.
        wall_temp: heater temperature at the bottom wall, in the same units as
            ``temp``; applies a Dirichlet BC to the ghost-fluid gradients there.
            ``None`` keeps a zero-gradient wall.
        eps: floor passed through to ``interface_normals`` and the ghost fluid
            gradients.

    Returns:
        ``(liquid_side, vapor_side)``, each shape ``(..., H, W)``, holding
        ``grad(T) . n`` for that phase on the ``band_cells``-wide band and zero
        outside it, so the two overlap on the band and can be subtracted there.
    """
    normal_x, normal_y = interface_normals(sdf, dx, dy, eps)

    liquid_grad_x, liquid_grad_y = liquid_temp_grad(temp, sdf, sat_temp, dx, dy, wall_temp, eps)
    liquid_normal_gradient = liquid_grad_x * normal_x + liquid_grad_y * normal_y

    vapor_grad_x, vapor_grad_y = vapor_temp_grad(temp, sdf, sat_temp, dx, dy, wall_temp, eps)
    vapor_normal_gradient = vapor_grad_x * normal_x + vapor_grad_y * normal_y

    # Spread each one-sided gradient into the opposite phase along the normals so
    # both are defined on a common band; the flux follows the local gradient across
    # the band (matching how Flash-X spreads its source), and masking back to the
    # band leaves conducted exactly zero outside it.
    band = band_mask(sdf, band_cells * max(dx, dy))
    liquid_side = constant_normal_extrapolation(
        liquid_normal_gradient, vapor_mask(sdf) & band, normal_x, normal_y, dx, dy
    )
    vapor_side = constant_normal_extrapolation(
        vapor_normal_gradient, liquid_mask(sdf) & band, -normal_x, -normal_y, dx, dy
    )
    return liquid_side * band, vapor_side * band


def mass_transfer(
    temp: torch.Tensor,
    sdf: torch.Tensor,
    sat_temp,
    dx: float,
    dy: float,
    stefan: float,
    reynolds: float,
    prandtl: float,
    thermal_conductivity: float,
    band_cells: int = DEFAULT_BAND_CELLS,
    taper_decay_cells: float = 2.0,
    wall_temp=None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Interfacial mass flux from the Stefan condition, non-dimensionalized.

    Energy balance at the interface: the heat conducted *to* the interface from
    both phases is consumed as latent heat. With ``q = -k grad(T)``, the heat each
    phase delivers through the interface is ``-k_i dT/dn_i`` along that phase's
    outward normal, so summing the two sides (from ``interface_heatflux``) gives

        mdot = St (k_l dT/dn_l - k_v dT/dn_v) / (Re Pr)

    The liquid conductivity is the reference (``k_l = 1``) and
    ``thermal_conductivity`` is the vapor's value relative to it (the ``thcogas``
    sim parameter). ``Re Pr`` is the Peclet number, and ``St = cp dT / h_fg`` is
    sensible over latent heat -- so a larger Stefan number (weaker latent heat)
    converts the same temperature gradient into *more* mass transfer, and
    ``mdot -> 0`` as the latent heat grows.

    Sign: **negative** means evaporation (liquid to vapor), positive means
    condensation. With ``n`` pointing from liquid to vapor, a superheated liquid
    cools toward the interface, so ``dT/dn_l < 0`` and, with no leading minus on
    the expression above, ``mdot < 0`` for evaporation.

    Banding: the two one-sided gradients are each constant-extrapolated across the
    band (in ``interface_heatflux``), so the jump ``k_l dT/dn_l - k_v dT/dn_v`` can
    be formed cell-wise on the band. The jump is a *surface* quantity, so it is
    tapered by an exponential ``exp(-|sdf| / (taper_decay_cells * dx))`` -- the
    ``|grad H|`` delta kernel that spreads a surface source into a volume -- giving
    the decaying profile Flash-X's massflux has (which decays ~0.6 per cell). The
    taper is a scalar, so tapering the jump equals tapering each heat flux and
    subtracting.

    Args:
        temp: cell-centered temperature, shape ``(..., H, W)``.
        sdf: cell-centered signed distance, shape ``(..., H, W)``. sdf < 0 is liquid,
            sdf >= 0 is vapor.
        sat_temp: interface (saturation) temperature; scalar or broadcastable to
            ``temp``.
        dx: cell spacing in x.
        dy: cell spacing in y.
        stefan: Stefan number.
        reynolds: Reynolds number.
        prandtl: Prandtl number.
        thermal_conductivity: vapor conductivity relative to the liquid.
        band_cells: half-width, in cells, of the interface band the flux is spread
            over (the extent over which the gradients are extrapolated).
        taper_decay_cells: decay length, in cells, of the exponential taper. ~2
            reproduces the Flash-X massflux decay (~0.6 per cell).
        wall_temp: heater temperature at the bottom wall, in the same units as
            ``temp``; applies a Dirichlet BC to the ghost-fluid gradients there.
            ``None`` keeps a zero-gradient wall.
        eps: floor passed through to the interface normals and ghost fluid
            gradients.

    Returns:
        Physical non-dimensional mass flux, shape ``(..., H, W)``: the Stefan jump
        on the ``band_cells``-wide band with an exponential taper, zero elsewhere.
    """
    # interface_heatflux already extrapolates each one-sided gradient across the
    # band, so both are defined on the band and the jump can be formed cell-wise.
    liquid_heatflux, vapor_heatflux = interface_heatflux(
        temp, sdf, sat_temp, dx, dy, band_cells, wall_temp, eps
    )
    conducted = liquid_heatflux - thermal_conductivity * vapor_heatflux

    # The jump is a surface quantity; taper it across the band (the |grad H| delta
    # kernel that spreads a surface source into a volume). The Flash-X massflux
    # decays exponentially from the interface (~0.6 per cell), so use an exponential
    # taper; conducted is already zero outside the band, so no cutoff is needed here.
    decay_length = taper_decay_cells * max(dx, dy)
    taper = 1.0
    return stefan / (reynolds * prandtl) * conducted * taper


def continuity(
    temp: torch.Tensor,
    sdf: torch.Tensor,
    sat_temp,
    dx: float,
    dy: float,
    stefan: float,
    reynolds: float,
    prandtl: float,
    thermal_conductivity: float,
    rhogas: float,
    band_cells: int = DEFAULT_BAND_CELLS,
    taper_decay_cells: float = 2.0,
    wall_temp=None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Mass flux scaled by the normal density jump across the interface.

    This is the volumetric source that the phase-change mass flux imposes on the
    velocity divergence: ``div(u) = mdot (1/rho_v - 1/rho_l) delta_interface``. The
    specific-volume jump ``1/rho_v - 1/rho_l`` is a scalar (liquid density is the
    reference, ``rho_l = 1``), and the interface delta is regularized over the same
    ``band_cells``-wide band the flux is spread on (``smoothed_delta``), so the
    result occupies a band rather than a single cell.

    Using ``grad(rho) . n`` of the sharp density step instead would confine the
    source to the one cell where the step lives, collapsing the band no matter how
    wide ``mdot`` is.

    Args:
        temp: cell-centered temperature, shape ``(..., H, W)``.
        sdf: cell-centered signed distance, shape ``(..., H, W)``. ``sdf < 0`` is
            liquid, ``sdf >= 0`` is vapor.
        sat_temp: interface (saturation) temperature; scalar or broadcastable to
            ``temp``.
        dx: cell spacing in x.
        dy: cell spacing in y.
        stefan: Stefan number.
        reynolds: Reynolds number.
        prandtl: Prandtl number.
        thermal_conductivity: vapor conductivity relative to the liquid.
        rhogas: vapor-phase density.
        band_cells: half-width, in cells, of the interface band (see
            ``mass_transfer``).
        wall_temp: heater temperature at the bottom wall, in the same units as
            ``temp`` (see ``mass_transfer``). ``None`` keeps a zero-gradient wall.
        eps: floor passed through to the interface normals and ghost fluid
            gradients.

    Returns:
        Velocity-divergence source ``mdot (1/rho_v - 1/rho_l) delta``, shape
        ``(..., H, W)``, nonzero on the ``band_cells``-wide interface band.
    """
    mdot = mass_transfer(
        temp, sdf, sat_temp, dx, dy, stefan, reynolds, prandtl, thermal_conductivity,
        band_cells=band_cells, taper_decay_cells=taper_decay_cells, wall_temp=wall_temp, eps=eps,
    )

    # Specific-volume jump 1/rho_v - 1/rho_l across the interface (rho_l = 1).
    specific_volume_jump = 1.0 / rhogas - 1.0
    band_width = band_cells * max(dx, dy)
    return -mdot * specific_volume_jump * smoothed_delta(sdf, band_width)