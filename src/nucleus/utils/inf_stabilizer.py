from dataclasses import dataclass

import torch

@dataclass
class NormalizedTempLimits:
    sat_temp: float
    wall_temp: float
    sdf_zero_levelset: float

def clip_temp_by_phase(temp, sdf, limit: NormalizedTempLimits):
    # temp should never be above the heater temperature
    temp = torch.clamp(temp, max=limit.wall_temp + 1e-8)
    # vapor should not be below the sat temp, but the liquid
    # may be above the sat temp. (i.e., just above heater)
    temp = torch.where(sdf > -limit.sdf_zero_levelset, torch.clamp(temp, min=limit.sat_temp - 1e-8), temp)
    return temp
