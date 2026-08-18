from dataclasses import dataclass
from typing import Optional, List

import torch


@dataclass
class Trajectory:
    """A rollout trajectory -- the input container for a model's forward_trajectory.

    Each field carries a batch and time axis, ``(B, T, ...)``. The spatial grids
    are whatever the target model expects: collocated (every field on the same
    ``(H, W)`` grid) for Nucleus2MoE, or the natural staggered/nodal grids for
    Nucleus2MoEDivFree (``velx`` on ``(H, W+1)`` faces, ``vely`` on ``(H+1, W)``,
    ``psi`` nodal ``(H+1, W+1)``, ``phi`` cell-centered ``(H, W)``). Converting
    between grids is the responsibility of whatever builds the Trajectory, since it
    depends on how the data was generated -- it is deliberately not done here.

    ``sim_params`` holds one physical sim-parameter dict per batch element.
    """
    sdf: torch.Tensor
    temp: torch.Tensor
    velx: torch.Tensor
    vely: torch.Tensor
    sim_params: List[dict]
    psi: Optional[torch.Tensor] = None
    phi: Optional[torch.Tensor] = None

    @property
    def num_steps(self) -> int:
        return self.sdf.shape[1]

    def last(self, num_steps: int) -> "Trajectory":
        """The last ``num_steps`` timesteps of every tracked field, including the
        potentials psi/phi when present."""
        tail = lambda field: field[:, -num_steps:] if field is not None else None
        return Trajectory(
            sdf=self.sdf[:, -num_steps:],
            temp=self.temp[:, -num_steps:],
            velx=self.velx[:, -num_steps:],
            vely=self.vely[:, -num_steps:],
            sim_params=self.sim_params,
            psi=tail(self.psi),
            phi=tail(self.phi),
        )

    def extend(
        self,
        sdf: torch.Tensor,
        temp: torch.Tensor,
        velx: torch.Tensor,
        vely: torch.Tensor,
        psi: torch.Tensor,
        phi: torch.Tensor,
    ) -> "Trajectory":
        """Append predicted frames along the time axis, returning a new Trajectory.
        All six fields -- including the potentials psi/phi -- are tracked so they can
        be fed back autoregressively."""
        append = lambda existing, new: torch.cat((existing, new), dim=1) if existing is not None else new
        return Trajectory(
            sdf=torch.cat((self.sdf, sdf), dim=1),
            temp=torch.cat((self.temp, temp), dim=1),
            velx=torch.cat((self.velx, velx), dim=1),
            vely=torch.cat((self.vely, vely), dim=1),
            sim_params=self.sim_params,
            psi=append(self.psi, psi),
            phi=append(self.phi, phi),
        )
