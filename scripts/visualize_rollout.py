import argparse
import numpy as np
import h5py
from pathlib import Path

from boiling_viz.boiling_video import BoilingVideoBuilder


def load_trajectory(path):
    FIELDS = ["dfun", "temperature", "velx", "vely"]
    with h5py.File(path, "r") as handle:
        return np.concatenate([handle[field] for field in FIELDS], axis=-1)


def vapor_volume(sdf, dx, dy):
    vapor_mask = sdf > 0
    return vapor_mask.astype(float).sum(axis=(-2, -1)) * dx * dy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, type=str)
    args = parser.parse_args()
    path = Path(args.path)
    pred = load_trajectory(path / "pred_trajectory.hdf5")
    gt = load_trajectory(path / "gt_trajectory.hdf5")
    
    builder = BoilingVideoBuilder(pred[::5])
    builder.make_video(
        "./traj.gif", 
        duration=10, 
        colorbars=False,
        step_counter=True,
        field_titles=True,
        transparent_nan=False,
        columnwise=True
    )

    pred_vv = vapor_volume(pred[..., 0], 1/4, 1/4).mean()
    gt_vv = vapor_volume(gt[..., 0], 1/4, 1/4).mean()

    print(pred_vv, gt_vv)

if __name__ == "__main__":
    main()