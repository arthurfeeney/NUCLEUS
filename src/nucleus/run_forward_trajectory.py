from dataclasses import dataclass
import h5py
import json
import torch

from nucleus.data.layout import convert_layout

@dataclass
class TestResults:
    case_name: str
    preds: torch.Tensor
    targets: torch.Tensor
    sim_params: dict

def run_test(cfg, model, normalizer, test_file_path: str, trajectory_steps: int):
    with h5py.File(test_file_path, "r") as handle:
        sdf = torch.from_numpy(handle["dfun"][:])
        temp = torch.from_numpy(handle["temperature"][:])
        velx = torch.from_numpy(handle["velx"][:])
        vely = torch.from_numpy(handle["vely"][:])
        gt_trajectory = torch.stack((sdf, temp, velx, vely), dim=-1)

    initial_state: torch.Tensor = gt_trajectory[cfg.start_time : cfg.start_time + cfg.history_time_window][None, :]
    json_path = test_file_path.replace(".hdf5", ".json")
    with open(json_path, "r") as handle:
        sim_params_dict: dict = json.load(handle)
        
    device = "cuda" if torch.cuda.is_available() else "cpu"

    normalized_initial_state = normalizer.normalize(initial_state, bulk_temp=sim_params_dict["bulk_temp"]).to(device)
    normalized_sim_params_dict = normalizer.normalize_params([sim_params_dict])[0]
    normalized_sim_params_tensor = torch.tensor(
        [normalized_sim_params_dict[param] for param in model.expected_fluid_params] +
        [normalized_sim_params_dict["heater"][param] for param in model.expected_heater_params] +
        [normalized_sim_params_dict[param] for param in model.expected_global_params],
        device=device
    )[None, :]

    with torch.inference_mode():
        normalized_pred_trajectory: torch.Tensor = model.forward_trajectory(
            convert_layout(normalized_initial_state, target_layout=model.layout, source_layout="t h w c"),
            normalized_sim_params_tensor,
            dx=1/4,
            input_time_window_size=8,
            output_time_window_size=8,
            trajectory_steps=trajectory_steps,
            use_sdf_reinit=False,
            return_moe_outputs=False
        )
        
    pred_trajectory = normalizer.unnormalize(normalized_pred_trajectory, bulk_temp=sim_params_dict["bulk_temp"])
    pred_trajectory = convert_layout(pred_trajectory, target_layout="t h w c", source_layout=model.layout)
    pred_trajectory = pred_trajectory.squeeze(0)
    
    case_name = f"{sim_params_dict['setup']}_{sim_params_dict['liquid']}_{sim_params_dict['heater']['wallTemp']}"

    return TestResults(
        case_name,
        pred_trajectory, 
        gt_trajectory[cfg.start_time : cfg.start_time + trajectory_steps],
        sim_params=sim_params_dict
    )