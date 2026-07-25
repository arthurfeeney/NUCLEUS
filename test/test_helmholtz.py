import numpy as np

from nucleus.helmholtz import (
    stream_function_from_velocity,
    potential_from_velocity,
    helmholtz_decomposition,
    coupled_helmholtz_decomposition,
    _curl_psi_faces,
    _grad_phi_faces,
)


def _nodes(n, dx, dy):
    node_x = np.arange(n + 1) * dx
    node_y = np.arange(n + 1) * dy
    grid_y, grid_x = np.meshgrid(node_y, node_x, indexing="ij")  # (H+1, W+1)
    return grid_x, grid_y


def _centers(n, dx, dy):
    center_x = (np.arange(n) + 0.5) * dx
    center_y = (np.arange(n) + 0.5) * dy
    grid_y, grid_x = np.meshgrid(center_y, center_x, indexing="ij")  # (H, W)
    return grid_x, grid_y


def _reconstruction_residual(velfacex, velfacey, psi, phi, dx, dy):
    curl_x, curl_y = _curl_psi_faces(psi, dx, dy)
    grad_x, grad_y = _grad_phi_faces(phi, dx, dy)
    res_x = velfacex - curl_x - grad_x
    res_y = velfacey - curl_y - grad_y
    # co-locate faces at cell centers so the two residual components share a grid
    speed = np.sqrt(velfacex[:, :-1] ** 2 + velfacey[:-1, :] ** 2)
    residual = np.sqrt(res_x[:, :-1] ** 2 + res_y[:-1, :] ** 2)
    return np.linalg.norm(residual) / np.linalg.norm(speed)


def test_stream_function_recovers_solenoidal_field():
    # psi that vanishes on the left/right/bottom walls with zero y-derivative at
    # the top outflow: sin(pi x) sin(pi/2 y). Build face velocities from its exact
    # discrete curl and check the solve recovers psi at the nodes.
    n = 64
    dx = dy = 1.0 / n
    node_x, node_y = _nodes(n, dx, dy)

    psi_true = np.sin(np.pi * node_x) * np.sin(0.5 * np.pi * node_y)
    velfacex, velfacey = _curl_psi_faces(psi_true, dx, dy)

    psi = stream_function_from_velocity(velfacex, velfacey, dx, dy)
    # walls are pinned to 0 by construction, so the additive constant is fixed.
    assert np.linalg.norm(psi - psi_true) / np.linalg.norm(psi_true) < 1e-6


def test_potential_recovers_irrotational_field():
    # phi with Neumann walls and Dirichlet top outflow: cos(pi x) cos(pi/2 y).
    n = 64
    dx = dy = 1.0 / n
    center_x, center_y = _centers(n, dx, dy)

    phi_true = np.cos(np.pi * center_x) * np.cos(0.5 * np.pi * center_y)
    velfacex, velfacey = _grad_phi_faces(phi_true, dx, dy)

    phi = potential_from_velocity(velfacex, velfacey, dx, dy)
    assert np.linalg.norm(phi - phi_true) / np.linalg.norm(phi_true) < 1e-6


def test_helmholtz_decomposition_reconstructs_a_mixed_field():
    # Build a face velocity from BOTH potentials and check the decomposition
    # separates them and reconstructs the field (no harmonic part here).
    n = 64
    dx = dy = 1.0 / n
    node_x, node_y = _nodes(n, dx, dy)
    center_x, center_y = _centers(n, dx, dy)

    psi_true = np.sin(np.pi * node_x) * np.sin(0.5 * np.pi * node_y)
    phi_true = np.cos(np.pi * center_x) * np.cos(0.5 * np.pi * center_y)
    curl_x, curl_y = _curl_psi_faces(psi_true, dx, dy)
    grad_x, grad_y = _grad_phi_faces(phi_true, dx, dy)
    velfacex, velfacey = curl_x + grad_x, curl_y + grad_y

    psi, phi, _, _ = helmholtz_decomposition(velfacex, velfacey, dx, dy)

    assert np.linalg.norm(psi - psi_true) / np.linalg.norm(psi_true) < 1e-4
    assert np.linalg.norm(phi - phi_true) / np.linalg.norm(phi_true) < 1e-4
    assert _reconstruction_residual(velfacex, velfacey, psi, phi, dx, dy) < 1e-4


def test_coupled_decomposition_captures_the_harmonic_part():
    # A mixed field plus a uniform (harmonic: div- and curl-free) through-flow.
    # The independent solve drops the harmonic part into the residual; the coupled
    # solve recovers its uniform component and reconstructs u far more completely.
    n = 96
    dx = dy = 1.0 / n
    node_x, node_y = _nodes(n, dx, dy)
    center_x, center_y = _centers(n, dx, dy)

    psi_true = np.sin(np.pi * node_x) * np.sin(0.5 * np.pi * node_y)
    phi_true = np.cos(np.pi * center_x) * np.cos(0.5 * np.pi * center_y)
    curl_x, curl_y = _curl_psi_faces(psi_true, dx, dy)
    grad_x, grad_y = _grad_phi_faces(phi_true, dx, dy)
    velfacex = curl_x + grad_x + 0.3   # uniform through-flow (harmonic)
    velfacey = curl_y + grad_y + 0.2

    psi_i, phi_i, _, _ = helmholtz_decomposition(velfacex, velfacey, dx, dy)
    psi_c, phi_c, _, _ = coupled_helmholtz_decomposition(velfacex, velfacey, dx, dy)

    independent = _reconstruction_residual(velfacex, velfacey, psi_i, phi_i, dx, dy)
    coupled = _reconstruction_residual(velfacex, velfacey, psi_c, phi_c, dx, dy)

    assert coupled < 0.1                # reconstructs u well despite the harmonic flow
    assert coupled < 0.5 * independent  # and much better than the independent solve


def test_decomposition_batches_over_leading_dimensions():
    # The solves should vectorize over arbitrary leading (e.g. time) dimensions.
    n = 32
    dx = dy = 1.0 / n
    node_x, node_y = _nodes(n, dx, dy)

    psi_true = np.sin(np.pi * node_x) * np.sin(0.5 * np.pi * node_y)
    velfacex, velfacey = _curl_psi_faces(psi_true, dx, dy)
    batch_x = np.stack([velfacex, 2.0 * velfacex])
    batch_y = np.stack([velfacey, 2.0 * velfacey])

    psi = stream_function_from_velocity(batch_x, batch_y, dx, dy)
    assert psi.shape == (2, n + 1, n + 1)
    assert np.linalg.norm(psi[0] - psi_true) / np.linalg.norm(psi_true) < 1e-6
    assert np.linalg.norm(psi[1] - 2.0 * psi_true) / np.linalg.norm(psi_true) < 1e-6
