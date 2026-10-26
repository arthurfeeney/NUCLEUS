import pytest
import torch

from nucleus.layers.mlp import DenseMLP
from nucleus.layers.moe.topk_moe import TopkMoE, TopkRouterWithBias
from nucleus.models import get_model, get_model_class, load_model
from nucleus.models.nucleus2_moe_base import Nucleus2MoEBaseConfig, config_from_dict, config_to_dict
from nucleus.testing.flashx import flashx_normalizer, random_batch, random_forecast_batch
from nucleus.testing.reference_attention import reference_natten
from nucleus.training.modules.nucleus2_moe import Nucleus2MoETraining, Nucleus2MoETrainingSettings

MODELS = ("nucleus2_moe", "nucleus2_moe_staggered", "nucleus2_moe_divfree_curl", "nucleus2_moe_divfree_projection")
SMALL_MODEL = dict(
    patch_size=4, embed_dim=64, num_heads=2, processor_blocks=2,
    num_experts=4, topk=2, moe_intermediate_dim=64, patching="Linear",
)


def _model(name, **overrides):
    torch.manual_seed(0)
    model_class = get_model_class(name)
    normalizer = flashx_normalizer(model_class.field_names, vector_fields=model_class.vector_fields)
    return get_model(name, normalizer, **(SMALL_MODEL | overrides))


@pytest.mark.parametrize("name", MODELS)
def test_a_dense_model_predicts_without_moe_outputs(name):
    model = _model(name, mlp_type="dense")
    assert not any(isinstance(module, (TopkMoE, TopkRouterWithBias)) for module in model.modules())
    assert all(isinstance(block.mlp, DenseMLP) for block in model.blocks)
    batch = model.normalizer.normalize(random_batch(model.field_names, model.config_keys, batch_size=2))
    with torch.no_grad(), reference_natten():
        output = model(batch)
    assert output.moe_outputs == []
    for name_, tensor in output.prediction.fields.items():
        assert tensor.shape == batch.fields[name_].shape
        assert torch.isfinite(tensor).all()


def test_the_dense_mlp_has_the_parameters_of_one_expert():
    dense = _model("nucleus2_moe", mlp_type="dense").blocks[0].mlp
    moe = _model("nucleus2_moe").blocks[0].mlp
    assert dense.w1.shape == moe.w1.shape[1:]
    assert dense.w2.shape == moe.w2.shape[1:]


def test_the_dense_mlp_keeps_the_input_dtype():
    mlp = DenseMLP(hidden_dim=8, intermediate_dim=16)
    out = mlp(torch.randn(2, 3, 8, dtype=torch.bfloat16))
    assert out.shape == (2, 3, 8)
    assert out.dtype == torch.bfloat16


def test_a_config_without_mlp_type_is_an_moe():
    config_dict = config_to_dict(Nucleus2MoEBaseConfig(**SMALL_MODEL))
    del config_dict["mlp_type"]
    assert config_from_dict(Nucleus2MoEBaseConfig, config_dict).mlp_type == "moe"


def test_an_unknown_mlp_type_is_rejected():
    with pytest.raises(ValueError, match="unknown mlp_type"):
        _model("nucleus2_moe", mlp_type="mixture")


def test_a_dense_checkpoint_loads_as_dense(tmp_path):
    model = _model("nucleus2_moe", mlp_type="dense")
    path = tmp_path / "checkpoint.pt"
    torch.save({"model": model.state_dict()}, path)
    loaded = load_model(path)
    assert loaded.config.mlp_type == "dense"
    for name, tensor in model.state_dict().items():
        if isinstance(tensor, torch.Tensor):
            torch.testing.assert_close(loaded.state_dict()[name], tensor)


def test_a_dense_model_trains_without_router_losses_or_metrics():
    model = _model("nucleus2_moe", mlp_type="dense", processor_blocks=1)
    training = Nucleus2MoETraining(model, Nucleus2MoETrainingSettings(target_timesteps=2))
    batch = random_forecast_batch(model.field_names, model.config_keys, target_timesteps=2)
    with reference_natten():
        output = training.training_step(batch, step=0)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert not any("_moe/" in name for name in output.metrics)
    assert all(param.grad is not None for param in model.blocks[0].mlp.parameters())
