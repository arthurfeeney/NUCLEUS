import h5py
import json
import numpy as np
import os
import pytest
import tempfile
import torch

from nucleus.data.forecast_dataset import ForecastDataset
from nucleus.data.in_mem_forecast_dataset import InMemForecastDataset
from nucleus.data.in_mem_divfree_forecast_dataset import (
    InMemDivFreeForecastDataset,
    GRID_SPACING,
)
from nucleus.data.layout import channel_dim, time_dim
from nucleus.physics.poisson import (
    curl_faces_from_nodes,
    grad_faces_from_centers,
    reconstruct_velocity_from_helmholtz,
)

FIELDS = ["dfun", "temperature", "velx", "vely"]
FIELDS_FACE = ["dfun", "temperature", "velfacex", "velfacey"]


fluid_params = {
    "val1": 1,
    "val2": 2
}
heater_params = {
    "hot1": 1,
    "hot2": 2
}
global_params = {
    "g1": 1,
    "g2": 2
}

@pytest.mark.parametrize("dataset_class", [ForecastDataset, InMemForecastDataset])
@pytest.mark.parametrize("history_time_window", [1, 2, 8, 16])
@pytest.mark.parametrize("future_time_window", [1, 2, 8, 16])
@pytest.mark.parametrize("layout", ["t h w c", "t c h w", "h w t c"])
def test_forecast_dataset(
    dataset_class,
    history_time_window,
    future_time_window,
    layout
):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "sim.hdf5")
        with h5py.File(path, "w") as handle:
            for field in FIELDS:
                handle.create_dataset(field, data=np.random.randn(100, 64, 64))
        json_path = path.replace("hdf5", "json")
        with open(json_path, "w") as handle:
            params = dict(
                bulk_temp=50,
                sat_temp=58,
                x_max=8,
                x_min=-8,
                y_max=16,
                y_min=0,
                num_blocks_x=24,
                num_blocks_y=24,
                nx_block=16,
                ny_block=16
            )
            params.update(fluid_params)
            params.update({"heater": heater_params})
            params.update(global_params)
            json_params = json.dumps(params)
            handle.write(json_params)
            
        dataset = dataset_class(
            [path],
            FIELDS,
            FIELDS,
            future_time_window,
            history_time_window,
            1,
            20,
            fluid_params,
            heater_params,
            global_params,
            layout,
            None,
            True
        )
        
        for i in range(3):
            data = dataset[i]
            assert data.input.shape[time_dim(layout)] == history_time_window
            assert data.target.shape[time_dim(layout)] == future_time_window
            assert data.input.shape[channel_dim(layout)] == 4
            assert len(data.sim_params_tensor) == len(fluid_params) + len(heater_params) + len(global_params)
            assert data.sim_params_tensor[0] == fluid_params["val1"]


def _write_divfree_sim(path, num_frames, height, width, velfacex=None, velfacey=None):
    # Write a synthetic staggered simulation: cell-centered dfun/temperature plus
    # face-staggered velfacex (H, W+1) and velfacey (H+1, W).
    if velfacex is None:
        velfacex = np.random.randn(num_frames, height, width + 1)
    if velfacey is None:
        velfacey = np.random.randn(num_frames, height + 1, width)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("dfun", data=np.random.randn(num_frames, height, width))
        handle.create_dataset("temperature", data=np.random.randn(num_frames, height, width))
        handle.create_dataset("velfacex", data=velfacex)
        handle.create_dataset("velfacey", data=velfacey)
    params = dict(
        bulk_temp=50, sat_temp=58, x_max=8, x_min=-8, y_max=16, y_min=0,
        num_blocks_x=1, num_blocks_y=1, nx_block=width, ny_block=height,
    )
    params.update(fluid_params)
    params.update({"heater": heater_params})
    params.update(global_params)
    with open(path.replace("hdf5", "json"), "w") as handle:
        handle.write(json.dumps(params))

def test_divfree_dataset_psi_phi_reconstruct_velocity():
    # Build face velocities that lie in the decomposition's range (no-penetration
    # walls) from a nodal streamfunction + cell potential, so the dataset's psi/phi
    # channels must reconstruct them via curl(psi) + grad(phi).
    height = width = 32
    # must match the spacing the dataset decomposes with, or psi/phi won't invert
    dx = dy = GRID_SPACING
    length_x, length_y = width * dx, height * dy

    node_x = (torch.arange(width + 1, dtype=torch.float64) * dx)[None, :]
    node_y = (torch.arange(height + 1, dtype=torch.float64) * dy)[:, None]
    psi_true = torch.sin(np.pi * node_x / length_x) * torch.sin(0.5 * np.pi * node_y / length_y)

    center_x = ((torch.arange(width, dtype=torch.float64) + 0.5) * dx)[None, :]
    center_y = ((torch.arange(height, dtype=torch.float64) + 0.5) * dy)[:, None]
    phi_true = torch.cos(np.pi * center_x / length_x) * torch.cos(0.5 * np.pi * center_y / length_y)

    curl_x, curl_y = curl_faces_from_nodes(psi_true, dx, dy)
    grad_x, grad_y = grad_faces_from_centers(phi_true, dx, dy)
    velfacex = (curl_x + grad_x).numpy()   # (H, W+1)
    velfacey = (curl_y + grad_y).numpy()   # (H+1, W)

    num_frames = 5
    velfacex = np.broadcast_to(velfacex, (num_frames, height, width + 1)).copy()
    velfacey = np.broadcast_to(velfacey, (num_frames, height + 1, width)).copy()

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "sim.hdf5")
        _write_divfree_sim(path, num_frames, height, width, velfacex, velfacey)

        dataset = InMemDivFreeForecastDataset(
            [path], FIELDS_FACE, FIELDS_FACE,
            2, 2, 1, 0, fluid_params, heater_params, global_params,
            "t h w c", None, False,
        )
        data = dataset[0]
        inp = data.input   # (T, H+1, W+1, [dfun, temp, velfacex, velfacey, psi, phi])

        # slice each channel back to its true (pre-pad) shape
        psi = inp[..., :height + 1, :width + 1, 4]   # nodal (T, H+1, W+1)
        phi = inp[..., :height, :width, 5]           # (T, H, W)
        out_velfacex = inp[..., :height, :width + 1, 2]   # (T, H, W+1)
        out_velfacey = inp[..., :height + 1, :width, 3]   # (T, H+1, W)

        # the faces stored on disk survive to the channels unchanged (history
        # window is the first 2 frames)
        assert torch.allclose(out_velfacex, torch.from_numpy(velfacex[:2]).float(), atol=1e-5)
        assert torch.allclose(out_velfacey, torch.from_numpy(velfacey[:2]).float(), atol=1e-5)

        # and psi/phi reconstruct them
        rx, ry = reconstruct_velocity_from_helmholtz(psi, phi, dx, dy)
        assert torch.allclose(out_velfacex, rx, atol=1e-3)
        assert torch.allclose(out_velfacey, ry, atol=1e-3)