import pytest
import torch

from nucleus.models import get_model
from nucleus.models.nucleus2_moe import Nucleus2MoE
from nucleus.models.nucleus2_moe_base import Nucleus2MoEBaseConfig, config_from_dict, config_to_dict
from nucleus.testing.flashx import flashx_normalizer, random_batch
from nucleus.testing.reference_attention import reference_natten

PATCH_SIZE = 4
SMALL_MODEL = dict(
    patch_size=PATCH_SIZE, embed_dim=64, num_heads=2, processor_blocks=1,
    num_experts=2, topk=1, moe_intermediate_dim=32, patching="Linear",
)


def _model(**overrides):
    torch.manual_seed(0)
    return get_model("nucleus2_moe", flashx_normalizer(Nucleus2MoE.field_names), **(SMALL_MODEL | overrides))


def _changed_patches(model) -> torch.Tensor:
    """(patches_y, patches_x) whether a patch's prediction changes when only the
    input patch in row 0, column 0 changes."""
    batch = model.normalizer.normalize(random_batch(Nucleus2MoE.field_names, Nucleus2MoE.config_keys, batch_size=1))
    cells = batch.stack_field_cells(model.field_names)
    perturbed = cells.clone()
    perturbed[:, :, :PATCH_SIZE, :PATCH_SIZE] += 1.0
    with torch.no_grad(), reference_natten():
        before, _ = model.step(cells, batch.config_tensor())
        after, _ = model.step(perturbed, batch.config_tensor())
    difference = (after - before).abs().amax(dim=(0, 1, -1))  # (H, W)
    height, width = difference.shape
    patches = difference.view(height // PATCH_SIZE, PATCH_SIZE, width // PATCH_SIZE, PATCH_SIZE)
    return patches.amax(dim=(1, 3)) > 0


def test_the_kernel_size_sets_how_far_a_block_sees():
    narrow = _changed_patches(_model(natten_kernel_size=3))
    wide = _changed_patches(_model(natten_kernel_size=5))
    # one block reaches one patch away with a kernel of 3, two with a kernel of 5
    assert narrow[1, 1] and not narrow[2, 2]
    assert wide[2, 2] and not wide[3, 3]


def test_a_config_without_a_kernel_size_keeps_the_original_3():
    config_dict = config_to_dict(Nucleus2MoEBaseConfig(**SMALL_MODEL))
    del config_dict["natten_kernel_size"]
    assert config_from_dict(Nucleus2MoEBaseConfig, config_dict).natten_kernel_size == 3


@pytest.mark.parametrize("mlp_type", ["moe", "dense"])
def test_a_wide_kernel_model_predicts_finite_fields(mlp_type):
    model = _model(natten_kernel_size=5, mlp_type=mlp_type, processor_blocks=2)
    assert all(block.attention.kernel_size == 5 for block in model.blocks)
    batch = model.normalizer.normalize(random_batch(Nucleus2MoE.field_names, Nucleus2MoE.config_keys, batch_size=2))
    with torch.no_grad(), reference_natten():
        output = model(batch)
    assert all(torch.isfinite(tensor).all() for tensor in output.prediction.fields.values())
