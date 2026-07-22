import torch

def divergence_center_from_face(facex, facey, dx, dy):
    """cell-centered divergence of a face-valued field.
    """
    return torch.diff(facex, axis=-1) / dx + torch.diff(facey, axis=-2) / dy


def grad_face_from_center(center, dx, dy):
    """Face-valued gradient of a cell-centered field.
    This assumes BCs Neumann (zero) on the closed left/right/bottom
    walls and Dirichlet (phi = 0) at the top outflow (a half cell above the top
    row of centers)."""
    grad_x = torch.zeros(center.shape[:-1] + (center.shape[-1] + 1,), dtype=center.dtype)
    grad_x[..., 1:-1] = torch.diff(center, axis=-1) / dx            # interior x-faces

    grad_y = torch.zeros(center.shape[:-2] + (center.shape[-2] + 1, center.shape[-1]), dtype=center.dtype)
    grad_y[..., 1:-1, :] = torch.diff(center, axis=-2) / dy         # interior y-faces
    grad_y[..., -1, :] = -center[..., -1, :] / (dy / 2)          # Dirichlet top outflow
    return grad_x, grad_y


def curl_center_from_face(velfacex, velfacey, dx, dy):
    """Exact vorticity ``d vely/dx - d velx/dy`` at the nodes solved for
    (j=1..H, i=1..W-1). The top row uses the Neumann-outflow ghost
    ``velfacex[H] = -velfacex[H-1]`` (odd reflection: velx = d psi/dy = 0 at the
    outflow)."""
    fx = np.concatenate([velfacex, -velfacex[..., -1:, :]], axis=-2)   # (..., H+1, W+1)
    dvely_dx = np.diff(velfacey, axis=-1) / dx                          # (..., H+1, W-1)
    dvelx_dy = np.diff(fx, axis=-2) / dy                                # (..., H, W+1)
    return dvely_dx[..., 1:, :] - dvelx_dy[..., :, 1:-1]                # (..., H, W-1)


# Differentiable cell-centered Poisson solve, ``laplacian(phi) = source``, with
# Neumann boundary conditions on the left/right/bottom walls and a Dirichlet
# ``phi = 0`` outflow at the top. 
# DCT-II in x (Neumann both walls) and DCT-IV in y (Neumann bottom, Dirichlet top)
# diagonalize the 5-point Laplacian.
# These are implemented in torch so it can sit inside the model's forward
# pass and backpropagate. It is written as cached-free matrix transforms (no FFT)
# so it stays inside a ``torch.compile(fullgraph=True)`` graph.


def _dct2_matrix(n: int, device, dtype) -> torch.Tensor:
    # Orthonormal DCT-II basis: C[k, i] = a(k) cos(pi (2i+1) k / (2n)), with
    # a(0) = sqrt(1/n) and a(k) = sqrt(2/n) otherwise. Its transpose is the inverse.
    freqs = torch.arange(n, device=device, dtype=dtype)
    samples = torch.arange(n, device=device, dtype=dtype)
    basis = torch.cos(torch.pi * (2 * samples[None, :] + 1) * freqs[:, None] / (2 * n))
    # DC (k=0) column is scaled by sqrt(1/n), the rest by sqrt(2/n).
    scale = torch.where(
        freqs == 0,
        torch.tensor((1.0 / n) ** 0.5, device=device, dtype=dtype),
        torch.tensor((2.0 / n) ** 0.5, device=device, dtype=dtype),
    )[:, None]
    return basis * scale


def _dct4_matrix(n: int, device, dtype) -> torch.Tensor:
    # Orthonormal DCT-IV basis: C[k, i] = sqrt(2/n) cos(pi (2i+1)(2k+1) / (4n)).
    # DCT-IV is its own inverse basis (the matrix is symmetric-orthonormal).
    freqs = torch.arange(n, device=device, dtype=dtype)
    samples = torch.arange(n, device=device, dtype=dtype)
    basis = torch.cos(torch.pi * (2 * samples[None, :] + 1) * (2 * freqs[:, None] + 1) / (4 * n))
    return basis * (2.0 / n) ** 0.5


def _dst2_matrix(n: int, device, dtype) -> torch.Tensor:
    # Orthonormal DST-II basis: S[k, i] = a(k) sin(pi (2i+1)(k+1) / (2n)), with
    # a(k) = sqrt(2/n) except the Nyquist row (k = n-1) which is sqrt(1/n). Its
    # transpose is the inverse.
    freqs = torch.arange(n, device=device, dtype=dtype)
    samples = torch.arange(n, device=device, dtype=dtype)
    basis = torch.sin(torch.pi * (2 * samples[None, :] + 1) * (freqs[:, None] + 1) / (2 * n))
    scale = torch.where(
        freqs == n - 1,
        torch.tensor((1.0 / n) ** 0.5, device=device, dtype=dtype),
        torch.tensor((2.0 / n) ** 0.5, device=device, dtype=dtype),
    )[:, None]
    return basis * scale


def _dst4_matrix(n: int, device, dtype) -> torch.Tensor:
    # Orthonormal DST-IV basis: S[k, i] = sqrt(2/n) sin(pi (2i+1)(2k+1) / (4n)).
    freqs = torch.arange(n, device=device, dtype=dtype)
    samples = torch.arange(n, device=device, dtype=dtype)
    basis = torch.sin(torch.pi * (2 * samples[None, :] + 1) * (2 * freqs[:, None] + 1) / (4 * n))
    return basis * (2.0 / n) ** 0.5


def solve_poisson_dirichlet_neumann(source: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    """Solve ``laplacian(psi) = source`` on a cell-centered grid with Dirichlet
    (``psi = 0``) on the closed left/right/bottom walls and Neumann
    (``d psi/dy = 0``) at the top outflow -- the streamfunction's boundary
    conditions (each closed wall is a streamline). The eigenbasis is DST-II in x
    (Dirichlet both walls) and DST-IV in y (Dirichlet bottom, Neumann top).
    ``source`` has shape ``(..., H, W)``; leading dims broadcast. Differentiable.
    """
    height, width = source.shape[-2], source.shape[-1]
    device, dtype = source.device, source.dtype

    basis_x = _dst2_matrix(width, device, dtype)     # (W, W)
    basis_y = _dst4_matrix(height, device, dtype)    # (H, H)

    freq_x = torch.arange(width, device=device, dtype=dtype)
    freq_y = torch.arange(height, device=device, dtype=dtype)
    eig_x = -4.0 / dx**2 * torch.sin((freq_x + 1) * torch.pi / (2 * width)) ** 2
    eig_y = -4.0 / dy**2 * torch.sin((2 * freq_y + 1) * torch.pi / (4 * height)) ** 2
    eigenvalues = eig_y[:, None] + eig_x[None, :]    # strictly negative -> invertible

    source_hat = torch.matmul(basis_y, torch.matmul(source, basis_x.transpose(-1, -2)))
    psi_hat = source_hat / eigenvalues
    return torch.matmul(basis_y.transpose(-1, -2), torch.matmul(psi_hat, basis_x))


def solve_poisson_neumann_dirichlet(source: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    """Solve ``laplacian(phi) = source`` on a cell-centered grid.

    NOTE: This is hard-coded for a pool-boiling setup.
    Boundary conditions: Neumann (``d phi/dn = 0``) on the left/right/bottom walls
    and Dirichlet (``phi = 0``) at the top outflow, matching
    
    Args:
        source: (..., H, W), 
        dx: float
        dy: float
    """
    height, width = source.shape[-2], source.shape[-1]
    device, dtype = source.device, source.dtype

    basis_x = _dct2_matrix(width, device, dtype)     # (W, W)
    basis_y = _dct4_matrix(height, device, dtype)    # (H, H)

    freq_x = torch.arange(width, device=device, dtype=dtype)
    freq_y = torch.arange(height, device=device, dtype=dtype)
    eig_x = -4.0 / dx**2 * torch.sin(freq_x * torch.pi / (2 * width)) ** 2
    eig_y = -4.0 / dy**2 * torch.sin((2 * freq_y + 1) * torch.pi / (4 * height)) ** 2
    eigenvalues = eig_y[:, None] + eig_x[None, :]    # strictly negative -> invertible

    # forward transform: source_hat = C4_y @ source @ C2_x^T
    source_hat = torch.matmul(basis_y, torch.matmul(source, basis_x.transpose(-1, -2)))
    phi_hat = source_hat / eigenvalues
    # inverse transform: phi = C4_y^T @ phi_hat @ C2_x
    return torch.matmul(basis_y.transpose(-1, -2), torch.matmul(phi_hat, basis_x))
