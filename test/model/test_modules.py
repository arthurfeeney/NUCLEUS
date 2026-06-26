from hydra import initialize, compose
import os

from nucleus.models.modules import PhaseForecastModule

def test_phase_forecase_module():
    with initialize(version_base=None, config_path="../../config"):
        cfg = compose(config_name="default", overrides=["model_cfg=nucleus2/nucleus2_phase"])
        
    module = PhaseForecastModule(
        None,
        cfg.model_cfg,
        cfg.data_cfg,
        cfg.normalizer_cfg,
        cfg.optim_cfg,
        cfg.scheduler_cfg,
        log_wandb=False,
        normalization_constants=None
    )
    
    assert module is not None