import pytest
import torch
from hydra import compose, initialize

from nucleus.models import get_model, get_model_class
from nucleus.testing.flashx import flashx_normalizer, random_batch, random_forecast_batch
from nucleus.training.modules.common import grid_weights, interface_weights, pooled_l1_loss, weighted_l1_loss
from nucleus.training.modules.nucleus2_moe import Nucleus2MoETraining, Nucleus2MoETrainingSettings
from nucleus.training.modules.nucleus2_moe_divfree_projection import (
    Nucleus2MoEDivFreeProjectionTraining,
    Nucleus2MoEDivFreeProjectionTrainingSettings,
)
from nucleus.training.modules.nucleus2_moe_staggered import (
    Nucleus2MoEStaggeredTraining,
    Nucleus2MoEStaggeredTrainingSettings,
)

SMALL_MODEL = dict(
    patch_size=4, embed_dim=64, num_heads=2, processor_blocks=1,
    num_experts=2, topk=1, moe_intermediate_dim=32, patching="Linear",
)
TRAININGS = {
    "nucleus2_moe": (Nucleus2MoETraining, Nucleus2MoETrainingSettings),
    "nucleus2_moe_staggered": (Nucleus2MoEStaggeredTraining, Nucleus2MoEStaggeredTrainingSettings),
    "nucleus2_moe_divfree_projection": (Nucleus2MoEDivFreeProjectionTraining, Nucleus2MoEDivFreeProjectionTrainingSettings),
    "nucleus2_moe_divfree_curl": (Nucleus2MoEDivFreeProjectionTraining, Nucleus2MoEDivFreeProjectionTrainingSettings),
}


def _model(name):
    torch.manual_seed(0)
    model_class = get_model_class(name)
    normalizer = flashx_normalizer(model_class.field_names, vector_fields=model_class.vector_fields)
    return get_model(name, normalizer, **SMALL_MODEL)


def _target_with_sdf(model, sdf_physical):
    """A normalized one-sample target whose dfun is sdf_physical, (1, T, H, W)."""
    batch = random_batch(model.field_names, model.config_keys, batch_size=1, timesteps=sdf_physical.shape[1])
    batch = batch.with_fields({"dfun": sdf_physical.to(batch.fields["dfun"].dtype)})
    return model.normalizer.normalize(batch)


def test_cells_near_the_interface_get_the_interface_weight():
    model = _model("nucleus2_moe")
    dx = random_batch(model.field_names, model.config_keys, batch_size=1, timesteps=1).dx("dfun")[0].item()
    sdf = torch.linspace(-10 * dx, 10 * dx, 32).expand(1, 1, 32, 32).clone()
    target = _target_with_sdf(model, sdf)
    weights = interface_weights(model.normalizer, target, interface_weight=5.0, band_cells=3.0)
    near = sdf.abs() < 3 * dx
    assert near.any() and (~near).any()
    torch.testing.assert_close(weights[near], torch.full_like(weights[near], 5.0))
    torch.testing.assert_close(weights[~near], torch.ones_like(weights[~near]))


def test_a_weight_of_one_needs_no_weights():
    model = _model("nucleus2_moe")
    target = _target_with_sdf(model, torch.zeros(1, 1, 32, 32))
    assert interface_weights(model.normalizer, target, interface_weight=1.0, band_cells=3.0) is None


def test_a_face_takes_the_heavier_of_the_cells_it_borders():
    cells = torch.tensor([[1.0, 5.0, 1.0], [1.0, 1.0, 1.0]])
    assert grid_weights(cells, torch.Size([2, 4])).tolist() == [[1.0, 5.0, 5.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
    assert grid_weights(cells, torch.Size([3, 3])).tolist() == [[1.0, 5.0, 1.0], [1.0, 5.0, 1.0], [1.0, 1.0, 1.0]]
    with pytest.raises(ValueError, match="neither"):
        grid_weights(cells, torch.Size([4, 4]))


def test_uniform_weights_give_the_unweighted_loss():
    torch.manual_seed(0)
    pairs = {
        "cells": (torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)),
        "x_faces": (torch.randn(2, 3, 4, 6), torch.randn(2, 3, 4, 6)),
    }
    torch.testing.assert_close(pooled_l1_loss(pairs, torch.ones(2, 3, 4, 5)), pooled_l1_loss(pairs))
    prediction, goal = torch.randn(2, 3, 4, 5, 3), torch.randn(2, 3, 4, 5, 3)
    torch.testing.assert_close(
        weighted_l1_loss(prediction, goal, torch.ones(2, 3, 4, 5)), torch.nn.functional.l1_loss(prediction, goal)
    )


def test_the_weighted_loss_is_dominated_by_the_heavy_cells():
    goal = torch.zeros(1, 1, 2, 2)
    prediction = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
    weights = torch.tensor([[[[9.0, 1.0], [1.0, 1.0]]]])
    # an error of 1 on the heavy cell: 9 / (9 + 3)
    assert pooled_l1_loss({"cells": (prediction, goal)}, weights).item() == pytest.approx(0.75)


@pytest.mark.parametrize("name", TRAININGS)
def test_every_training_trains_with_an_interface_weight(name, differentiable_attention_on_cpu):
    training_class, settings_class = TRAININGS[name]
    model = _model(name)
    settings = settings_class(target_timesteps=2, interface_loss_weight=5.0, interface_loss_band_cells=3.0)
    output = training_class(model, settings).training_step(
        random_forecast_batch(model.field_names, model.config_keys, target_timesteps=2), step=0
    )
    assert torch.isfinite(output.loss)
    output.loss.backward()


@pytest.mark.parametrize("name", TRAININGS)
def test_the_train_configs_hold_the_interface_settings(name):
    with initialize(version_base=None, config_path="../../config"):
        cfg = compose(config_name=f"train_{name}", overrides=["split_dir=x", "log_dir=y"])
    assert cfg.training.interface_loss_weight == 1.0
    assert cfg.training.interface_loss_band_cells == 3.0
