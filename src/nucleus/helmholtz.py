from functools import lru_cache

import numpy as np
from scipy.fft import dct, idct

# ---------------------------------------------------------------------------
# MAC (staggered) grid Helmholtz decomposition.
#
# The velocity is face-valued: `velfacex` on vertical faces has shape
# (..., height, width + 1) and `velfacey` on horizontal faces has shape
# (..., height + 1, width). On this grid the discrete divergence and curl are
# *exact* adjacent differences (no interpolation), so the decomposition is
# clean where the collocated one suffered a ~1e-2 divergence artifact.
#
# The two parts live on their natural grids: the potential `phi` at cell
# centers (shape (..., height, width)) and the streamfunction `psi` at cell
# corners / nodes (shape (..., height + 1, width + 1)). Reconstruction happens
# back on the faces:
#   velfacex ~ d psi/dy + d phi/dx   (both on x-faces)
#   velfacey ~ -d psi/dx + d phi/dy  (both on y-faces)
# ---------------------------------------------------------------------------


def _grid_shape_from_faces(velfacex, velfacey):
    height = velfacey.shape[-2] - 1
    width = velfacex.shape[-1] - 1
    if velfacex.shape[-2] != height or velfacey.shape[-1] != width:
        raise ValueError(
            "expected staggered faces velfacex (..., H, W+1) and velfacey (..., H+1, W); "
            f"got {velfacex.shape} and {velfacey.shape}"
        )
    return height, width


def mac_divergence(velfacex, velfacey, dx, dy):
    """Exact cell-centered divergence of a face-valued velocity field.

    ``div[j, i] = (velfacex[j, i+1] - velfacex[j, i])/dx
                + (velfacey[j+1, i] - velfacey[j, i])/dy``, shape (..., H, W).
    """
    return np.diff(velfacex, axis=-1) / dx + np.diff(velfacey, axis=-2) / dy


def _grad_phi_faces(phi, dx, dy):
    """Face-valued gradient of a cell-centered potential, matching the BCs of
    ``potential_from_velocity``: Neumann (zero) on the closed left/right/bottom
    walls and Dirichlet (phi = 0) at the top outflow (a half cell above the top
    row of centers)."""
    grad_x = np.zeros(phi.shape[:-1] + (phi.shape[-1] + 1,), dtype=phi.dtype)
    grad_x[..., 1:-1] = np.diff(phi, axis=-1) / dx            # interior x-faces

    grad_y = np.zeros(phi.shape[:-2] + (phi.shape[-2] + 1, phi.shape[-1]), dtype=phi.dtype)
    grad_y[..., 1:-1, :] = np.diff(phi, axis=-2) / dy         # interior y-faces
    grad_y[..., -1, :] = -phi[..., -1, :] / (dy / 2)          # Dirichlet top outflow
    return grad_x, grad_y


def _curl_psi_faces(psi, dx, dy):
    """Face-valued curl of a nodal streamfunction: ``velfacex = d psi/dy`` on
    x-faces, ``velfacey = -d psi/dx`` on y-faces."""
    facex = np.diff(psi, axis=-2) / dy       # (..., H, W+1)
    facey = -np.diff(psi, axis=-1) / dx      # (..., H+1, W)
    return facex, facey


def _solve_potential(source, dx, dy):
    """Solve ``laplacian(phi) = source`` on the cell-centered grid with Neumann
    (``d phi/dn = 0``) on the closed left/right/bottom walls and Dirichlet
    (``phi = 0``) at the top outflow. The MAC ``div(grad(.))`` is exactly the
    5-point Laplacian whose eigenbasis is DCT-II in x (Neumann both walls) and
    DCT-IV in y (Neumann bottom, Dirichlet top), so the solve is exact. eig_y is
    strictly negative, so the x DC mode (eig_x[0] = 0) still has a nonzero total
    eigenvalue -- the top Dirichlet BC removes the nullspace.
    """
    height, width = source.shape[-2], source.shape[-1]
    eig_x = -4.0 / dx**2 * np.sin(np.arange(width) * np.pi / (2 * width)) ** 2
    eig_y = -4.0 / dy**2 * np.sin((2 * np.arange(height) + 1) * np.pi / (4 * height)) ** 2
    eigenvalues = eig_y[:, None] + eig_x[None, :]

    source_hat = dct(dct(source, type=2, axis=-1, norm="ortho"), type=4, axis=-2, norm="ortho")
    phi_hat = source_hat / eigenvalues
    return idct(idct(phi_hat, type=4, axis=-2, norm="ortho"), type=2, axis=-1, norm="ortho")


def potential_from_velocity(velfacex, velfacey, dx, dy):
    """Recover the dilatational potential phi (cell-centered) of a face-valued
    velocity field: ``grad(phi)`` reproduces the irrotational (curl-free) part.
    phi solves ``laplacian(phi) = div(u)`` with the MAC divergence as the source
    (Neumann walls, Dirichlet top outflow); see ``_solve_potential``.
    """
    _grid_shape_from_faces(velfacex, velfacey)
    return _solve_potential(mac_divergence(velfacex, velfacey, dx, dy), dx, dy)


@lru_cache(maxsize=None)
def _nodal_laplacian_eigsystem(height, width, dx, dy):
    """Eigen-decomposition of the separable nodal Laplacian used for the
    streamfunction solve. Cached per (height, width, dx, dy).

    x acts on interior nodes i=1..width-1 with Dirichlet walls (symmetric
    tridiagonal). y acts on nodes j=1..height with Dirichlet at the bottom wall
    and Neumann at the top outflow -- the ghost ``psi[H+1] = psi[H-1]`` makes the
    top row non-symmetric, so it is diagonalized directly rather than by an FFT.
    """
    main_x = -2.0 * np.ones(width - 1)
    off_x = np.ones(width - 2)
    laplacian_x = (np.diag(main_x) + np.diag(off_x, 1) + np.diag(off_x, -1)) / dx**2

    main_y = -2.0 * np.ones(height)
    off_y = np.ones(height - 1)
    laplacian_y = np.diag(main_y) + np.diag(off_y, 1) + np.diag(off_y, -1)
    laplacian_y[-1, -2] = 2.0                                # Neumann top ghost
    laplacian_y = laplacian_y / dy**2

    eig_x, vec_x = np.linalg.eigh(laplacian_x)               # symmetric
    eig_y, vec_y = np.linalg.eig(laplacian_y)                # non-symmetric top row
    eig_y = eig_y.real
    vec_y = vec_y.real
    vec_x_inv = vec_x.T                                      # orthonormal
    vec_y_inv = np.linalg.inv(vec_y)
    return eig_x, vec_x, vec_x_inv, eig_y, vec_y, vec_y_inv


def _mac_vorticity_nodes(velfacex, velfacey, dx, dy):
    """Exact vorticity ``d vely/dx - d velx/dy`` at the nodes solved for
    (j=1..H, i=1..W-1). The top row uses the Neumann-outflow ghost
    ``velfacex[H] = -velfacex[H-1]`` (odd reflection: velx = d psi/dy = 0 at the
    outflow)."""
    fx = np.concatenate([velfacex, -velfacex[..., -1:, :]], axis=-2)   # (..., H+1, W+1)
    dvely_dx = np.diff(velfacey, axis=-1) / dx                          # (..., H+1, W-1)
    dvelx_dy = np.diff(fx, axis=-2) / dy                                # (..., H, W+1)
    return dvely_dx[..., 1:, :] - dvelx_dy[..., :, 1:-1]                # (..., H, W-1)


def stream_function_from_velocity(velfacex, velfacey, dx, dy):
    """Recover the nodal streamfunction psi of a face-valued velocity field (the
    "inverse curl").

    Uses ``velfacex = d psi/dy`` and ``velfacey = -d psi/dx``, so ``curl(psi)``
    reproduces the solenoidal (divergence-free) part of the input; any divergent
    component is dropped. psi solves the vorticity Poisson equation
    ``laplacian(psi) = -omega`` on the nodes with pool-boiling boundary
    conditions: Dirichlet (``psi = 0``, a streamline) on the closed left, right,
    and bottom walls, and Neumann (``d psi/dy = 0``) at the top outflow. The
    result has shape (..., height + 1, width + 1) with the Dirichlet walls zeroed.
    """
    height, width = _grid_shape_from_faces(velfacex, velfacey)
    omega = _mac_vorticity_nodes(velfacex, velfacey, dx, dy)   # (..., H, W-1)

    eig_x, vec_x, vec_x_inv, eig_y, vec_y, vec_y_inv = _nodal_laplacian_eigsystem(
        height, width, dx, dy
    )
    # laplacian(psi) = -omega, solved separably: psi = Vy ((Vy^-1 (-omega) Vx^-T) / (ly+lx)) Vx^T
    rhs_hat = np.matmul(vec_y_inv, np.matmul(-omega, vec_x_inv.T))
    psi_hat = rhs_hat / (eig_y[:, None] + eig_x[None, :])
    psi_interior = np.matmul(vec_y, np.matmul(psi_hat, vec_x.T))

    psi = np.zeros(omega.shape[:-2] + (height + 1, width + 1), dtype=psi_interior.dtype)
    psi[..., 1:, 1:width] = psi_interior                       # Dirichlet walls stay 0
    return psi


def helmholtz_decomposition(velfacex, velfacey, dx, dy):
    """Helmholtz decomposition of a face-valued velocity field into a nodal
    streamfunction psi (solenoidal part = ``curl(psi)``) and a cell-centered
    potential phi (dilatational part = ``grad(phi)``), plus the face-valued
    velocity residual (harmonic part) not captured by either.
    """
    psi = stream_function_from_velocity(velfacex, velfacey, dx, dy)
    phi = potential_from_velocity(velfacex, velfacey, dx, dy)

    curl_x, curl_y = _curl_psi_faces(psi, dx, dy)
    grad_x, grad_y = _grad_phi_faces(phi, dx, dy)
    return psi, phi, velfacex - curl_x - grad_x, velfacey - curl_y - grad_y


def coupled_helmholtz_decomposition(velfacex, velfacey, dx, dy, gate=None):
    height, width = _grid_shape_from_faces(velfacex, velfacey)

    div = mac_divergence(velfacex, velfacey, dx, dy)
    #if gate is not None:
    #    div = gate * div                        # gate the SOURCE, not grad(phi)
    phi = _solve_potential(div, dx, dy)
    grad_x, grad_y = _grad_phi_faces(phi, dx, dy)
    remainder_x = velfacex - grad_x
    remainder_y = velfacey - grad_y

    psi = stream_function_from_velocity(remainder_x, remainder_y, dx, dy)
    curl_x, curl_y = _curl_psi_faces(psi, dx, dy)

    # Uniform component of the harmonic remainder the Poisson solve left behind,
    # added back analytically (the origin of x/y only shifts psi by a constant,
    # which the curl removes).
    mean_velx = (remainder_x - curl_x).mean(axis=(-2, -1), keepdims=True)
    mean_vely = (remainder_y - curl_y).mean(axis=(-2, -1), keepdims=True)
    y = (np.arange(height + 1) * dy)[:, None]
    x = (np.arange(width + 1) * dx)[None, :]
    psi = psi + mean_velx * y - mean_vely * x

    curl_x, curl_y = _curl_psi_faces(psi, dx, dy)
    return psi, phi, velfacex - curl_x - grad_x, velfacey - curl_y - grad_y


if __name__ == "__main__":
    import h5py
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    dx = dy = 1 / 32
    data_dir = "/share/crsp/lab/amowli/share/BubbleML_staggered/"
    with h5py.File(f"{data_dir}/SingleBubble-Saturated-R515B-2D/Twall_21.hdf5") as handle:
        velfacex = handle["velfacex"][100]      # (H, W+1)
        velfacey = handle["velfacey"][100]      # (H+1, W)
        sdf = handle["dfun"][100]               # (H, W)

    # Hard gate: keep grad(phi) only in a thin band around the interface
    # (|sdf| <= band), zeroing it in both the deep liquid and the deep vapor so
    # its div-free tail is absorbed into curl(psi) instead.
    band = 0.2
    gate = (np.abs(sdf) <= band).astype(velfacex.dtype)

    psi, phi, residual_facex, residual_facey = coupled_helmholtz_decomposition(
        velfacex, velfacey, dx, dy, gate=gate
    )
    curl_x, curl_y = _curl_psi_faces(psi, dx, dy)
    # grad(phi) is ungated: phi already comes from the gated source, so its
    # magnitude is nonlocal but its divergence is confined to the gated band.
    grad_x, grad_y = _grad_phi_faces(phi, dx, dy)

    # Interpolate face vectors to cell centers (for magnitude images) and use the
    # exact MAC divergence for the divergence images.
    def to_center(facex, facey):
        cx = 0.5 * (facex[..., :, 1:] + facex[..., :, :-1])   # (H, W)
        cy = 0.5 * (facey[..., 1:, :] + facey[..., :-1, :])   # (H, W)
        return cx, cy

    velx_cc, vely_cc = to_center(velfacex, velfacey)
    speed = np.sqrt(velx_cc**2 + vely_cc**2)
    residual = np.sqrt(np.sum(np.square(to_center(residual_facex, residual_facey)), axis=0))
    print("relative reconstruction residual:", np.linalg.norm(residual) / np.linalg.norm(speed))

    # How much velocity does the vapor gate throw away? The model outputs
    # curl(psi) + gate * grad(phi), so it zeros (1 - gate) * grad(phi).
    band = 2.0
    gate = np.clip(sdf / band + 1.0, 0.0, 1.0)
    gate = gate * gate * (3.0 - 2.0 * gate)   # smoothstep, matches vapor_gate_from_sdf
    grad_phi_cx, grad_phi_cy = to_center(grad_x, grad_y)
    grad_phi = np.sqrt(grad_phi_cx**2 + grad_phi_cy**2)
    dropped = (1.0 - gate) * grad_phi

    def dropped_fraction(mask):
        return np.linalg.norm(dropped[mask]) / np.linalg.norm(speed[mask]) if np.any(mask) else float("nan")

    print("velocity dropped by the vapor gate, ||(1-gate) grad(phi)|| / ||u||:")
    print(f"  global         : {np.linalg.norm(dropped) / np.linalg.norm(speed):.4e}")
    print(f"  liquid bulk    : {dropped_fraction(sdf < -band):.4e}   (sdf < {-band}, gate ~ 0)")
    print(f"  interface/vapor: {dropped_fraction(sdf >= -band):.4e}")

    # Each column is a face-valued vector field. Row 1 shows its magnitude
    # (interpolated to cell centers), row 2 the magnitude of its exact MAC
    # divergence (at cell centers).
    columns = [
        ("velocity", velfacex, velfacey),
        ("reconstruction", velfacex - residual_facex, velfacey - residual_facey),
        ("error (harmonic)", residual_facex, residual_facey),
        ("divergence-free part", curl_x, curl_y),
        ("non-divergence-free part", grad_x, grad_y),
    ]

    # One color scale per panel: the fields and their divergences span very
    # different magnitudes and would be invisible on a shared scale.
    fig, axes = plt.subplots(2, len(columns), figsize=(20, 9), sharex=True, sharey=True)
    for col, (title, facex, facey) in enumerate(columns):
        center_x, center_y = to_center(facex, facey)
        #mag = np.sqrt(center_x**2 + center_y**2)
        top = axes[0, col].imshow(facey, origin="lower", cmap="RdBu")
        axes[0, col].set_title(title)
        fig.colorbar(top, ax=axes[0, col], fraction=0.046)

        # Divergence spans orders of magnitude -> log-scale colormap, with a
        # per-panel floor so a near-zero panel (the div-free part) does not
        # produce a many-decade colorbar dominated by roundoff.
        div_magnitude = np.abs(mac_divergence(facex, facey, dx, dy))
        vmax = max(float(div_magnitude.max()), 1e-30)
        bottom = axes[1, col].imshow(
            div_magnitude, origin="lower", cmap="viridis",
            norm=LogNorm(vmin=vmax * 1e-6, vmax=vmax, clip=True),
        )
        fig.colorbar(bottom, ax=axes[1, col], fraction=0.046)
    axes[0, 0].set_ylabel("magnitude")
    axes[1, 0].set_ylabel("|divergence|")
    fig.savefig("helmholtz.png", dpi=150, bbox_inches="tight")