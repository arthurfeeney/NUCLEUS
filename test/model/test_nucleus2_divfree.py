import random

import pytest
import torch
from hydra import compose, initialize

from nucleus.models.modules import get_train_module
from nucleus.models.nucleus2_moe_divfree import (
    GRADIENT_SPACING,
    Nucleus2MoEDivFree,
    Nucleus2MoEDivFreeConfig,
    Nucleus2MoEDivFreeInput,
    Nucleus2MoEDivFreeOutput,
)
from nucleus.data.in_mem_divfree_forecast_dataset import DivFreeBatch, DivFreeData
from nucleus.data.normalize import DivFreeNormalizer, NormalizerConstants
from nucleus.physics.poisson import divergence_centers_from_faces


class _IdentityParamsNormalizer(DivFreeNormalizer):
    """Unit per-field stats make the field normalize/unnormalize identity, and the
    sim-parameter normalization is identity so ``forward_trajectory`` builds its
    conditioning tensor without needing min/max dicts."""

    def normalize_params(self, sim_params_dicts):
        return sim_params_dicts

    def unnormalize_params(self, sim_params_dicts):
        return sim_params_dicts


def make_normalizer() -> _IdentityParamsNormalizer:
    constants = NormalizerConstants(
        max_domain_size=9.0, sdf_mean=0.0, sdf_std=1.0, absmax_temp=1.0,
        temp_mean=0.0, temp_std=1.0, velx_mean=0.0, velx_std=1.0, vely_mean=0.0,
        vely_std=1.0, psi_mean=0.0, psi_std=1.0, phi_mean=0.0, phi_std=1.0,
    )
    return _IdentityParamsNormalizer(constants)


@pytest.fixture
def model() -> Nucleus2MoEDivFree:
    return Nucleus2MoEDivFree(Nucleus2MoEDivFreeConfig(
        patch_size=16, embed_dim=32, num_heads=2, processor_blocks=1,
        num_experts=2, topk=1, moe_intermediate_dim=32, patching="Linear",
    )).eval()


def make_divfree_data(batch_size, time, height, width, sdf_fill=None) -> DivFreeData:
    """Random fields on their natural grids (with a batch dim). ``sdf_fill`` fills
    the sdf with a constant instead of noise (e.g. deep liquid)."""
    def rand(*trailing):
        return torch.randn(batch_size, time, *trailing)

    sdf = rand(height, width) if sdf_fill is None else torch.full(
        (batch_size, time, height, width), float(sdf_fill)
    )
    return DivFreeData(
        sdf=sdf, temperature=rand(height, width),
        velx=rand(height, width + 1), vely=rand(height + 1, width),
        psi=rand(height + 1, width + 1), phi=rand(height, width),
    )


def as_input(data: DivFreeData) -> Nucleus2MoEDivFreeInput:
    return Nucleus2MoEDivFreeInput(
        data.sdf, data.temperature, data.velx, data.vely, data.psi, data.phi
    )


def make_batch(data: DivFreeData, sim_params_tensor: torch.Tensor) -> DivFreeBatch:
    batch_size = data.sdf.shape[0]
    return DivFreeBatch(
        input=data,
        target=None,
        sim_params=[{} for _ in range(batch_size)],
        dx=torch.full((batch_size,), GRADIENT_SPACING),
        dy=torch.full((batch_size,), GRADIENT_SPACING),
        sim_params_tensor=sim_params_tensor,
    )


def run_step(model, data, sim_params, normalizer, input_type):
    """Feed the model input as either a bare ``Nucleus2MoEDivFreeInput`` (with the
    sim-param tensor passed separately) or a ``DivFreeBatch`` (which carries it)."""
    if input_type == "input":
        return model.step(as_input(data), sim_params, normalizer)
    return model.step(make_batch(data, sim_params), normalizer=normalizer)


def make_sim_params_dict(model) -> dict:
    # forward_trajectory assembles its conditioning tensor from a physical sim-param
    # dict; the values are arbitrary for a finiteness/shape check.
    params = {"bulk_temp": 50.0, "sat_temp": 58.0}
    params.update({param: random.random() for param in model.expected_fluid_params})
    params["heater"] = {param: random.random() for param in model.expected_heater_params}
    params.update({param: random.random() for param in model.expected_global_params})
    return params


@pytest.mark.parametrize("input_type", ["input", "batch"])
def test_step_returns_output_on_natural_grids(model, input_type):
    torch.manual_seed(0)
    batch_size, time, height, width = 1, 2, 64, 64
    data = make_divfree_data(batch_size, time, height, width)
    sim_params = torch.randn(batch_size, model.num_sim_params)

    with torch.no_grad():
        output = run_step(model, data, sim_params, make_normalizer(), input_type)

    assert isinstance(output, Nucleus2MoEDivFreeOutput)
    assert output.sdf.shape == (batch_size, time, height, width)
    assert output.temperature.shape == (batch_size, time, height, width)
    assert output.velx.shape == (batch_size, time, height, width + 1)
    assert output.vely.shape == (batch_size, time, height + 1, width)
    assert output.psi.shape == (batch_size, time, height + 1, width + 1)
    assert output.phi.shape == (batch_size, time, height, width)
    for field in (output.sdf, output.temperature, output.velx, output.vely, output.psi, output.phi):
        assert torch.isfinite(field).all()


@pytest.mark.parametrize("input_type", ["input", "batch"])
def test_velocity_is_divergence_free_in_deep_liquid(model, input_type):
    # Deep liquid everywhere -> the interface band mask is zero -> the dilatational
    # part vanishes and the velocity is pure curl(psi), which is divergence free.
    torch.manual_seed(0)
    batch_size, time, height, width = 1, 2, 64, 64
    data = make_divfree_data(batch_size, time, height, width, sdf_fill=-50.0)
    sim_params = torch.randn(batch_size, model.num_sim_params)

    with torch.no_grad():
        output = run_step(model, data, sim_params, make_normalizer(), input_type)

    # velx/vely are the face velocities; the unit normalizer keeps them physical.
    divergence = divergence_centers_from_faces(
        output.velx, output.vely, GRADIENT_SPACING, GRADIENT_SPACING
    )
    interior = divergence[..., :-1, :]  # drop the top outflow row
    assert interior.abs().max() < 1e-2


def make_divfree_batch(batch_size, time, height, width, num_sim_params) -> DivFreeBatch:
    return DivFreeBatch(
        input=make_divfree_data(batch_size, time, height, width),
        target=make_divfree_data(batch_size, time, height, width),
        sim_params=[{} for _ in range(batch_size)],
        dx=torch.full((batch_size,), GRADIENT_SPACING),
        dy=torch.full((batch_size,), GRADIENT_SPACING),
        sim_params_tensor=torch.randn(batch_size, num_sim_params),
    )


def build_divfree_module(extra_overrides=()):
    """Build a small CPU-sized DivFreeForecastModule from the real configs."""
    with initialize(version_base=None, config_path="../../config"):
        cfg = compose(
            config_name="default",
            overrides=[
                "model_cfg=nucleus2/nucleus2_divfree",
                "normalizer_cfg=divfree",
                "data_dir=/tmp",
                "model_cfg.train_module_name=divfree_forecast",
                "model_cfg.params.patch_size=16",
                "model_cfg.params.embed_dim=32",
                "model_cfg.params.processor_blocks=1",
                "model_cfg.params.num_heads=2",
                "model_cfg.params.num_experts=2",
                "model_cfg.params.topk=1",
                "model_cfg.params.moe_intermediate_dim=32",
                "model_cfg.params.activation_dtype=float32",
                *extra_overrides,
            ],
        )

    Module = get_train_module(cfg.model_cfg.train_module_name)
    assert Module.__name__ == "DivFreeForecastModule"
    module = Module(
        None,
        cfg.model_cfg,
        cfg.data_cfg,
        cfg.normalizer_cfg,
        cfg.optim_cfg,
        cfg.scheduler_cfg,
        log_wandb=False,
        normalization_constants=None,
    )
    module.default_log_dict = lambda *args, **kwargs: None  # force no logging
    return module


def test_divfree_forecast_module(model):
    # Build the DivFreeForecastModule around the Nucleus2MoEDivFree model via its
    # config, then run a validation step on a DivFreeBatch.
    module = build_divfree_module()
    assert isinstance(module.model, Nucleus2MoEDivFree)

    batch = make_divfree_batch(2, 2, 64, 64, module.model.num_sim_params)

    @torch.compiler.disable
    def check_step():
        with torch.no_grad():
            loss = module.validation_step(batch, 0)
            assert torch.isfinite(loss)
            assert loss > 0
    check_step()


def test_velocity_enters_loss_after_start_step(monkeypatch):
    # Early in training the velocity fields are held out of the loss; from the
    # start step on they are included. Drive global_step across the threshold and
    # check the velocity mismatch is invisible before and counted after.
    output = Nucleus2MoEDivFreeOutput(
        **{field: torch.zeros(1, 1, 4, 4) for field in
           ("sdf", "temperature", "psi", "phi")},
        velx=torch.zeros(1, 1, 4, 5), vely=torch.zeros(1, 1, 5, 4), moe_outputs=[],
    )
    target = DivFreeData(
        sdf=torch.zeros(1, 1, 4, 4), temperature=torch.zeros(1, 1, 4, 4),
        velx=torch.ones(1, 1, 4, 5), vely=torch.ones(1, 1, 5, 4),
        psi=torch.zeros(1, 1, 4, 4), phi=torch.zeros(1, 1, 4, 4),
    )
    module = build_divfree_module()

    def set_step(step):
        # global_step is a read-only Lightning property; shadow it on the instance's
        # type so _loss_fields sees the value we want.
        monkeypatch.setattr(type(module), "global_step",
                            property(lambda self: step), raising=False)

    set_step(0)
    pred_grids, _ = module._loss_fields(output, target)
    assert len(pred_grids) == 4  # velocity excluded early: mismatch is invisible
    assert module._field_loss(output, target) == 0

    set_step(10**9)
    pred_grids, _ = module._loss_fields(output, target)
    assert len(pred_grids) == 6  # velocity now included: the ones-vs-zeros gap counts
    assert module._field_loss(output, target) > 0


def test_per_field_mae_logs_every_field():
    module = build_divfree_module()
    output = Nucleus2MoEDivFreeOutput(
        **{field: torch.zeros(1, 1, 4, 4) for field in
           ("sdf", "temperature", "psi", "phi")},
        velx=torch.zeros(1, 1, 4, 5), vely=torch.zeros(1, 1, 5, 4), moe_outputs=[],
    )
    target = DivFreeData(
        sdf=torch.ones(1, 1, 4, 4), temperature=torch.ones(1, 1, 4, 4),
        velx=torch.ones(1, 1, 4, 5), vely=torch.ones(1, 1, 5, 4),
        psi=torch.ones(1, 1, 4, 4), phi=torch.ones(1, 1, 4, 4),
    )
    metrics = module._per_field_mae(output, target, "train")
    # Every field is logged (even ones held out of the loss), each MAE == 1 here.
    assert set(metrics) == {f"train/mae_{name}" for name in
                            ("sdf", "temperature", "velx", "vely", "psi", "phi")}
    for value in metrics.values():
        assert value == pytest.approx(1.0)


@pytest.mark.parametrize("trajectory_steps", [8, 24])
def test_forward_trajectory(model, trajectory_steps):
    torch.manual_seed(0)
    batch_size, height, width = 1, 64, 64
    input_window = 8
    initial_state = as_input(make_divfree_data(batch_size, input_window, height, width))

    with torch.inference_mode():
        trajectory = model.forward_trajectory(
            initial_state,
            make_sim_params_dict(model),
            make_normalizer(),
            dx=GRADIENT_SPACING,
            input_time_window_size=input_window,
            output_time_window_size=input_window,
            trajectory_steps=trajectory_steps,
        )

    assert trajectory.isfinite().all()
    assert trajectory.shape[0] == batch_size
    assert trajectory.shape[1] == trajectory_steps
    assert trajectory.shape[-1] == 11  # the dataset's cell-channel layout
