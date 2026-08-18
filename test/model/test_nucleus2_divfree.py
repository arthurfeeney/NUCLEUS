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
    divfree_input_to_cells,
    scale_gradient,
)
from nucleus.data.in_mem_divfree_forecast_dataset import DivFreeBatch, DivFreeData
from nucleus.trajectory import Trajectory
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


def run_step(model, data, sim_params, normalizer, input_type, use_div_gate=False):
    """Feed the model input as either a bare ``Nucleus2MoEDivFreeInput`` (with the
    sim-param tensor passed separately) or a ``DivFreeBatch`` (which carries it)."""
    if input_type == "input":
        return model.step(as_input(data), sim_params, normalizer, use_div_gate=use_div_gate)
    return model.step(make_batch(data, sim_params), normalizer=normalizer, use_div_gate=use_div_gate)


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


def test_step_with_sdf_reinit_gate(model):
    # Redistancing the sdf before the divergence gate is toggled by use_sdf_reinit;
    # the output stays on the natural grids and finite.
    torch.manual_seed(0)
    batch_size, time, height, width = 1, 2, 64, 64
    data = make_divfree_data(batch_size, time, height, width)
    sim_params = torch.randn(batch_size, model.num_sim_params)

    with torch.no_grad():
        output = model.step(as_input(data), sim_params, make_normalizer(), use_sdf_reinit=True)

    assert output.velx.shape == (batch_size, time, height, width + 1)
    for field in (output.sdf, output.velx, output.vely, output.psi, output.phi):
        assert torch.isfinite(field).all()


@pytest.mark.parametrize("input_type", ["input", "batch"])
def test_solenoidal_part_is_divergence_free(model, input_type):
    # With no divergence source, the velocity is pure curl(psi), which is divergence
    # free everywhere (div(curl) = 0 exactly on the MAC grid) regardless of the sdf.
    torch.manual_seed(0)
    batch_size, time, height, width = 1, 2, 64, 64
    with torch.no_grad():
        model.debed_psi.conv_transpose.weight.normal_()  # nonzero solenoidal velocity
        model.debed_div.conv_transpose.weight.zero_()    # no divergent part
    data = make_divfree_data(batch_size, time, height, width)
    sim_params = torch.randn(batch_size, model.num_sim_params)

    with torch.no_grad():
        output = run_step(model, data, sim_params, make_normalizer(), input_type, use_div_gate=True)

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
        cfg.checkpoint_path,
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


def _zero_output():
    return Nucleus2MoEDivFreeOutput(
        **{field: torch.zeros(1, 1, 4, 4) for field in
           ("sdf", "temperature", "psi", "phi", "div_source")},
        velx=torch.zeros(1, 1, 4, 5), vely=torch.zeros(1, 1, 5, 4), moe_outputs=[],
    )


def _target(**overrides):
    base = dict(
        sdf=torch.zeros(1, 1, 4, 4), temperature=torch.zeros(1, 1, 4, 4),
        velx=torch.zeros(1, 1, 4, 5), vely=torch.zeros(1, 1, 5, 4),
        psi=torch.zeros(1, 1, 4, 4), phi=torch.zeros(1, 1, 4, 4),
    )
    base.update(overrides)
    return DivFreeData(**base)


def test_field_loss_is_mean_abs_error_over_supervised_grids():
    # _field_loss is the summed |error| over exactly the grids _loss_fields selects,
    # divided by their total element count. This stays correct regardless of which
    # field set _loss_fields currently supervises.
    module = build_divfree_module()
    output = _zero_output()
    target = _target(
        sdf=torch.ones(1, 1, 4, 4), temperature=torch.ones(1, 1, 4, 4),
        velx=torch.ones(1, 1, 4, 5), vely=torch.ones(1, 1, 5, 4),
        psi=torch.ones(1, 1, 4, 4), phi=torch.ones(1, 1, 4, 4),
    )

    pred_grids, target_grids = module._loss_fields(output, target)
    total_abs_error = sum(
        (pred - tgt).abs().sum() for pred, tgt in zip(pred_grids, target_grids)
    )
    total_elements = sum(pred.numel() for pred in pred_grids)
    expected = total_abs_error / total_elements

    assert torch.allclose(module._field_loss(output, target), expected)
    assert module._field_loss(output, target) > 0


def test_field_loss_ignores_unsupervised_fields():
    # A mismatch in a field that _loss_fields does not select leaves the loss at zero.
    module = build_divfree_module()
    output = _zero_output()

    supervised_fields = {
        name for name in ("sdf", "temperature", "velx", "vely", "psi", "phi")
        if any(
            grid is getattr(output, name)
            for grid in module._loss_fields(output, _target())[0]
        )
    }
    unsupervised = ({"sdf", "temperature", "velx", "vely", "psi", "phi"}
                    - supervised_fields)

    field_shape = {
        "sdf": (4, 4), "temperature": (4, 4), "velx": (4, 5),
        "vely": (5, 4), "psi": (4, 4), "phi": (4, 4),
    }
    for name in unsupervised:
        mismatch = _target(**{name: torch.ones(1, 1, *field_shape[name])})
        assert module._field_loss(output, mismatch) == 0, name


def test_augment_preserves_field_shapes():
    # Per-field augmentation runs each field through the module's augmentations on its
    # own natural grid, preserving shapes and staying finite.
    torch.manual_seed(0)
    module = build_divfree_module()
    data = make_divfree_data(2, 2, 64, 64)
    augmented = module._augment(data)
    for name in ("sdf", "temperature", "velx", "vely", "psi", "phi"):
        assert augmented.__getattribute__(name).shape == data.__getattribute__(name).shape
        assert torch.isfinite(augmented.__getattribute__(name)).all()


def test_per_field_mae_logs_every_field():
    module = build_divfree_module()
    output = Nucleus2MoEDivFreeOutput(
        **{field: torch.zeros(1, 1, 4, 4) for field in
           ("sdf", "temperature", "psi", "phi", "div_source")},
        velx=torch.zeros(1, 1, 4, 5), vely=torch.zeros(1, 1, 5, 4), moe_outputs=[],
    )
    target = DivFreeData(
        sdf=torch.ones(1, 1, 4, 4), temperature=torch.ones(1, 1, 4, 4),
        velx=torch.ones(1, 1, 4, 5), vely=torch.ones(1, 1, 5, 4),
        psi=torch.ones(1, 1, 4, 4), phi=torch.ones(1, 1, 4, 4),
    )
    metrics = module._per_field_mae(output, target, "train")
    # Every field is logged (even ones held out of the loss), plus the divergence source.
    assert set(metrics) == {f"train/mae_{name}" for name in
                            ("sdf", "temperature", "velx", "vely", "psi", "phi", "div_source")}
    for name in ("sdf", "temperature", "velx", "vely", "psi", "phi"):
        assert metrics[f"train/mae_{name}"] == pytest.approx(1.0)
    # The target velocity is spatially constant, so its divergence (the div_source
    # target) is zero, matching the zero prediction.
    assert metrics["train/mae_div_source"] == pytest.approx(0.0)


@pytest.mark.parametrize("trajectory_steps", [8, 24])
def test_forward_trajectory(model, trajectory_steps):
    torch.manual_seed(0)
    batch_size, height, width = 1, 64, 64
    input_window = 8
    # forward_trajectory takes and returns a Trajectory with every field on its
    # natural grid; psi/phi are recomputed from the velocity each step.
    data = make_divfree_data(batch_size, input_window, height, width)
    initial_state = Trajectory(
        sdf=data.sdf, temp=data.temperature, velx=data.velx, vely=data.vely,
        sim_params=[make_sim_params_dict(model)],
    )

    with torch.inference_mode():
        trajectory = model.forward_trajectory(
            initial_state,
            make_normalizer(),
            dx=GRADIENT_SPACING,
            input_time_window_size=input_window,
            output_time_window_size=input_window,
            trajectory_steps=trajectory_steps,
        )

    assert isinstance(trajectory, Trajectory)
    assert trajectory.num_steps == trajectory_steps
    for field in (trajectory.sdf, trajectory.temp, trajectory.velx, trajectory.vely):
        assert field.shape[0] == batch_size
        assert field.shape[1] == trajectory_steps
        assert field.isfinite().all()
    assert trajectory.sdf.shape[-2:] == (height, width)
    assert trajectory.velx.shape[-2:] == (height, width + 1)
    assert trajectory.vely.shape[-2:] == (height + 1, width)


def _save_base_checkpoint(tmp_path, embed_dim=32, num_heads=2, processor_blocks=1,
                          num_experts=2, topk=1, moe_intermediate_dim=32, patch_size=16):
    """Build an unconstrained nucleus2_moe with a trunk matching build_divfree_module's
    and save its raw state dict, returning the path."""
    from nucleus.models._api import get_model

    base = get_model(
        "nucleus2_moe", patch_size=patch_size, embed_dim=embed_dim, num_heads=num_heads,
        processor_blocks=processor_blocks, num_experts=num_experts, topk=topk,
        moe_intermediate_dim=moe_intermediate_dim, patching="Linear",
        activation_dtype="float32",
    )
    path = str(tmp_path / "base.ckpt")
    torch.save(base.state_dict(), path)
    return base, path


def test_warm_start_transfers_trunk_and_keeps_heads_at_init(tmp_path):
    # A divfree module warm-started from an unconstrained base checkpoint should copy
    # the shared transformer trunk (blocks + out_norm) and leave the resized
    # embed/debed and the zero-init psi/phi heads untouched.
    base, path = _save_base_checkpoint(tmp_path)
    module = build_divfree_module(extra_overrides=(f"checkpoint_path={path}",))

    base_state = base.state_dict()
    warm_state = module.model.state_dict()

    transferred, kept = [], []
    for key, warm_tensor in warm_state.items():
        if key == "_extra_state":
            continue
        if key in base_state and base_state[key].shape == warm_tensor.shape:
            assert torch.equal(warm_tensor, base_state[key]), f"{key} was not transferred"
            transferred.append(key)
        else:
            kept.append(key)

    # The trunk (blocks/out_norm) transferred; the resized embed/debed and the
    # divfree-only velocity heads (psi + the two divergent-velocity heads) stayed at
    # their divfree init.
    assert any(key.startswith("blocks.") for key in transferred)
    assert any(key.startswith("out_norm") for key in transferred)
    assert all(key.startswith(("debed_psi", "debed_div")) for key in kept if "debed_" in key and "debed." not in key)
    assert any(key.startswith("embed.") for key in kept)

    # The zero-init psi head is untouched by the warm start (the divergent heads init
    # nonzero, so only psi is asserted zero here).
    assert torch.count_nonzero(module.model.debed_psi.conv_transpose.weight) == 0

    # The base checkpoint is recorded for reproducibility.
    assert module.checkpoint_path == path


def test_warm_start_from_lightning_checkpoint(tmp_path):
    # The same transfer works from a Lightning-style checkpoint (state nested under
    # "state_dict" with a "model." prefix).
    base, _ = _save_base_checkpoint(tmp_path)
    lightning_state = {f"model.{key}": value for key, value in base.state_dict().items()}
    path = str(tmp_path / "lightning.ckpt")
    torch.save({"state_dict": lightning_state}, path)

    module = build_divfree_module(extra_overrides=(f"checkpoint_path={path}",))

    base_state = base.state_dict()
    a_block_key = next(key for key in module.model.state_dict() if key.startswith("blocks."))
    assert torch.equal(module.model.state_dict()[a_block_key], base_state[a_block_key])
    # The divfree config survives -- the base's _extra_state was not applied.
    assert isinstance(module.model, Nucleus2MoEDivFree)


def test_debed_diagnostics_cover_all_heads():
    # The per-head diagnostics should report a grad norm and a weight distribution for
    # each debed head. FlexAttention has no CPU backward, so populate the head gradients
    # directly -- _debed_diagnostics only reads .grad and the weights.
    module = build_divfree_module()
    for head in module._debed_heads().values():
        for param in head.parameters():
            param.grad = torch.randn_like(param)

    scalar_log, histogram_log = module._debed_diagnostics(
        include_scalars=True, include_histograms=True
    )
    for name in ("debed", "debed_psi", "debed_div"):
        assert f"train/{name}/grad_norm" in scalar_log
        assert scalar_log[f"train/{name}/grad_norm"] > 0
        assert torch.isfinite(scalar_log[f"train/{name}/grad_norm"])
        assert f"train/{name}/weight_std" in scalar_log
        assert f"train/{name}/grad_hist" in histogram_log
        assert f"train/{name}/weight_hist" in histogram_log

    # With histograms disabled, only the scalar diagnostics are produced.
    scalars_only, empty_histograms = module._debed_diagnostics(
        include_scalars=True, include_histograms=False
    )
    assert empty_histograms == {}
    assert "train/debed_psi/grad_norm" in scalars_only


@pytest.mark.parametrize("scale", [1.0, 0.5, 1 / 32, 0.0])
def test_scale_gradient_identity_forward_scaled_backward(scale):
    # Forward is the identity; the backward gradient is multiplied by `scale`.
    x = torch.randn(5, requires_grad=True)
    y = scale_gradient(x, scale)
    assert torch.allclose(y, x)
    y.sum().backward()
    assert torch.allclose(x.grad, torch.full_like(x, scale))


def test_potential_grad_scale_defaults_to_dx(model):
    # The knob defaults to dx (full cancellation of the 1/dx reconstruction gain).
    assert model.config.potential_grad_scale == GRADIENT_SPACING


def test_potential_grad_scale_leaves_forward_unchanged(model):
    # scale_gradient is identity in forward, so the reconstructed velocity (and the
    # returned raw potentials) must not depend on potential_grad_scale.
    torch.manual_seed(0)
    data = make_divfree_data(1, 2, 64, 64)
    sim_params = torch.randn(1, model.num_sim_params)
    normalizer = make_normalizer()

    with torch.no_grad():
        model.config.potential_grad_scale = 1.0
        unscaled = model.step(as_input(data), sim_params, normalizer)
        model.config.potential_grad_scale = GRADIENT_SPACING
        scaled = model.step(as_input(data), sim_params, normalizer)

    assert torch.allclose(unscaled.velx, scaled.velx, atol=1e-5)
    assert torch.allclose(unscaled.vely, scaled.vely, atol=1e-5)
    # The returned psi/phi are the raw head outputs (not routed through the scaler).
    assert torch.equal(unscaled.psi, scaled.psi)
    assert torch.equal(unscaled.phi, scaled.phi)


def test_input_embedding_has_six_channels():
    # The embedding consumes sdf, temperature, and the four split face-velocity
    # channels -- psi/phi are not part of the input.
    assert Nucleus2MoEDivFree.expected_fields == [
        "dfun", "temperature", "vel_left", "vel_right", "vel_bottom", "vel_top",
    ]
    data = make_divfree_data(1, 2, 8, 8)
    cells = divfree_input_to_cells(data.sdf, data.temperature, data.velx, data.vely)
    assert cells.shape == (1, 2, 8, 8, 6)


def test_input_potentials_do_not_affect_output(model):
    # psi/phi are excluded from the input embedding, so perturbing them must leave the
    # model output unchanged.
    torch.manual_seed(0)
    data = make_divfree_data(1, 2, 64, 64)
    sim_params = torch.randn(1, model.num_sim_params)
    normalizer = make_normalizer()

    with torch.no_grad():
        baseline = model.step(as_input(data), sim_params, normalizer)
        perturbed = DivFreeData(
            sdf=data.sdf, temperature=data.temperature, velx=data.velx, vely=data.vely,
            psi=torch.randn_like(data.psi), phi=torch.randn_like(data.phi),
        )
        perturbed_output = model.step(as_input(perturbed), sim_params, normalizer)

    for name in ("sdf", "temperature", "velx", "vely", "psi", "phi"):
        assert torch.equal(getattr(baseline, name), getattr(perturbed_output, name)), name


def test_div_gate_confines_divergence_to_band(model):
    # Option B: the model predicts a divergence source; the Poisson reconstruction makes
    # div(velocity) = gated source. So with use_div_gate=True the velocity divergence is
    # ~0 wherever the (predicted-sdf) band gate is zero -- the divergence itself is
    # confined, and grad(phi) still extends smoothly into the bulk. The gate off (the
    # training default) leaves divergence leaking outside the band.
    torch.manual_seed(0)
    with torch.no_grad():
        model.debed.linear.weight.normal_(std=0.3)        # predicted sdf spans an interface
        model.debed_psi.conv_transpose.weight.normal_()   # nonzero solenoidal part
        model.debed_div.conv_transpose.weight.normal_()   # nonzero divergence source

    data = make_divfree_data(1, 2, 64, 64)
    sim_params = torch.randn(1, model.num_sim_params)
    normalizer = make_normalizer()

    with torch.no_grad():
        gated = model.step(as_input(data), sim_params, normalizer, use_div_gate=True)
        ungated = model.step(as_input(data), sim_params, normalizer, use_div_gate=False)

    # The band gate is built from the predicted (physical) sdf the model gated on.
    sdf_physical = normalizer.unnormalize_sdf(gated.sdf)
    in_band = sdf_physical.abs() <= 5 * GRADIENT_SPACING
    assert in_band.any() and (~in_band).any()  # both regions exist for the test to mean anything

    def divergence(output):
        return divergence_centers_from_faces(
            output.velx, output.vely, GRADIENT_SPACING, GRADIENT_SPACING
        )

    # Drop the top outflow row where the Dirichlet BC breaks the exact div = source identity.
    interior = (slice(None), slice(None), slice(None, -1), slice(None))
    gated_div = divergence(gated)[interior]
    outside = ~in_band[interior]
    inside = in_band[interior]

    assert gated_div[outside].abs().max() < 1e-2  # divergence confined: ~0 outside the band
    assert gated_div[inside].abs().max() > 1e-1   # the source injects real divergence in-band

    # Ungated, the same source leaks divergence outside the band.
    assert divergence(ungated)[interior][outside].abs().max() > 1e-1


def test_div_source_supervised_by_target_velocity_divergence():
    # div_source enters the loss with a target computed from the target velocities:
    # vel_std * divergence_centers_from_faces(target.velx, target.vely).
    torch.manual_seed(0)
    module = build_divfree_module()
    output = _zero_output()  # div_source predicted as 0
    target = _target(velx=torch.randn(1, 1, 4, 5), vely=torch.randn(1, 1, 5, 4))

    pred_grids, target_grids = module._loss_fields(output, target)

    # The divergence source is the last (pred, target) pair in the loss.
    assert pred_grids[-1] is output.div_source
    expected_target = module.normalizer.vel_std * divergence_centers_from_faces(
        target.velx, target.vely, GRADIENT_SPACING, GRADIENT_SPACING
    )
    assert torch.allclose(target_grids[-1], expected_target)
    # The target velocity is not divergence-free, so the div_source term contributes.
    assert (pred_grids[-1] - target_grids[-1]).abs().sum() > 0


def test_step_use_mass_transfer_replaces_learned_source(model):
    # use_mass_transfer swaps the learned debed_div output for the physics continuity
    # source (mdot * n.grad(rho)); it needs the physical sim-param dict.
    torch.manual_seed(0)
    data = make_divfree_data(1, 2, 64, 64)
    sim_params = torch.randn(1, model.num_sim_params)
    normalizer = make_normalizer()
    sim_params_dict = make_sim_params_dict(model)
    sim_params_dict.update(
        {"stefan": 0.1, "inv_reynolds": 0.01, "prandtl": 1.0, "thcogas": 0.1, "rhogas": 0.01}
    )
    sim_params_dict["heater"]["wallTemp"] = 80.0  # above bulk_temp so temp non-dim is sane

    with torch.no_grad():
        learned = model.step(as_input(data), sim_params, normalizer)
        physics = model.step(
            as_input(data), sim_params, normalizer,
            use_mass_transfer=True, sim_params_dict=sim_params_dict,
        )

    assert physics.div_source.shape == learned.div_source.shape
    for field in (physics.sdf, physics.velx, physics.vely, physics.div_source):
        assert torch.isfinite(field).all()
    # The physics source is not the learned head output.
    assert not torch.allclose(physics.div_source, learned.div_source)

    # Without the sim-param dict, mass transfer can't run.
    with torch.no_grad(), pytest.raises(AssertionError):
        model.step(as_input(data), sim_params, normalizer, use_mass_transfer=True)
