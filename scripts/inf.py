import os
import pathlib
import torch
import h5py
import json
from collections import OrderedDict
from nucleus.models import get_model
import hydra
from omegaconf import DictConfig, OmegaConf
from nucleus.data.normalize import get_normalizer
from nucleus.run_forward_trajectory import run_test, TestResults
from nucleus.utils.set_fp32_precision import set_fp32_precision
from lightning import LightningModule

@hydra.main(version_base=None, config_path="../config", config_name="inference")
def main(cfg: DictConfig):
    set_fp32_precision()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = cfg.model_cfg.name
    model_kwargs = OmegaConf.to_container(cfg.model_cfg.params, resolve=True)
    model = get_model(model_name, **model_kwargs)
    model = model.to(device)
    
    model_data = torch.load(cfg.checkpoint_path, map_location=device, weights_only=False)    
    weight_state_dict = OrderedDict()
    for key, val in model_data["state_dict"].items():
        print(key, val.shape)
        if isinstance(model, LightningModule):
            name = key
        else:
            name = key[6:]
        weight_state_dict[name] = val
    del model_data
    model.load_state_dict(weight_state_dict)
    model.eval()

    normalizer = get_normalizer(OmegaConf.to_container(cfg.normalizer_cfg, resolve=True))
    
    # Rollouts are saved in the directory containing the checkpoint
    save_root = pathlib.Path(cfg.checkpoint_path).parent / "rollouts"
    save_root.mkdir(parents=True, exist_ok=True)
    with open(save_root / "config.yaml", "w") as handle:
        OmegaConf.save(cfg, f=handle.name)
    
    all_test_results = []
    for test_file_path in cfg.data_cfg.test_paths:

        test_results: TestResults = run_test(cfg, model, normalizer, test_file_path, trajectory_steps=cfg.trajectory_steps)
        all_test_results.append(test_results)
     
    for test_result in all_test_results:
        rollout_save_root = save_root / test_result.case_name
        model_save_path = rollout_save_root / "model_cfg.json"
        with open(model_save_path, "w") as handle:
            OmegaConf.save(config=cfg.model_cfg, f=handle.name)
        h5py_save_path = rollout_save_root / "trajectories.hdf5"
        with h5py.File(h5py_save_path, "w") as handle:
            handle.create_dataset("pred_trajectory", data=test_result.preds)
            handle.create_dataset("gt_trajectory", data=test_result.targets)
        json_save_path = rollout_save_root / "sim_params.json"
        with open(json_save_path, "w") as handle:
            json.dump(test_result.sim_params, handle)
        
if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()