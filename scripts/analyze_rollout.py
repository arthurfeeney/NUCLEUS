import argparse
import numpy as np
import h5py
from pathlib import Path

def load_trajectory(path):
    FIELDS = ["dfun", "temperature", "velx", "vely"]
    with h5py.File(path, "r") as handle:
        return np.concatenate([handle[field] for field in FIELDS], axis=-1)

def metrics(array: np.array):
    pass

def vapor_volume(sdf, dx, dy):
    vapor_mask = sdf > 0
    return vapor_mask.astype(float).sum(axis=(-2, -1)) * dx * dy

def vapor_volume_at_height(sdf, dx, dy):
    vapor_mask = sdf > 0
    return vapor_mask.astype(float).sum(axis=(-2, -1)) * dx * dy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, type=str)
    args = parser.parse_args()
    path = Path(args.path)
    pred = load_trajectory(path / "pred_trajectory.hdf5")
    gt = load_trajectory(path / "gt_trajectory.hdf5")

    pred_vv = vapor_volume(pred[..., 0], 1/4, 1/4).mean()
    gt_vv = vapor_volume(gt[..., 0], 1/4, 1/4).mean()

    print(pred_vv, gt_vv)

if __name__ == "__main__":
    main()