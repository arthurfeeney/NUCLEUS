import argparse
import numpy as np
import h5py
from pathlib import Path
import torch

from boiling_viz.boiling_video import BoilingVideoBuilder


def load_trajectory(path):
    FIELDS = ["dfun", "temperature", "velx", "vely"]
    with h5py.File(path, "r") as handle:
        return np.stack([handle[field] for field in FIELDS], axis=0)


def vapor_volume(sdf, dx, dy):
    vapor_mask = sdf > 0
    return (vapor_mask.to(torch.float32).sum(axis=(-2, -1)) * dx * dy).mean()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, type=str)
    args = parser.parse_args()
    path = Path(args.path)
    traj = load_trajectory(path)
    
    traj = torch.from_numpy(traj)
    
    print(traj.shape)
    
    down = torch.nn.functional.interpolate(traj, (64, 64), mode="bicubic")
    traj_recon = torch.nn.functional.interpolate(down, (512, 512), mode="bicubic")

    print(vapor_volume(traj[0], 1/32, 1/32))
    print(vapor_volume(down[0], 1/4, 1/4))
    print(vapor_volume(traj_recon[0], 1/32, 1/32))
    


if __name__ == "__main__":
    main()