import torch

from nucleus.noise import PhaseFlip, _dilate_mask


def test_dilate_mask_grows_single_cell_into_disk():
    mask = torch.zeros(1, 1, 5, 5, dtype=torch.bool)
    mask[0, 0, 2, 2] = True
    dilated = _dilate_mask(mask, radius=1)
    # radius-1 disk is a plus/cross: center + 4 edge neighbors, corners excluded
    assert dilated.sum().item() == 5
    assert dilated[0, 0, 2, 1] and dilated[0, 0, 1, 2]
    assert not dilated[0, 0, 1, 1] and not dilated[0, 0, 3, 3]


def test_dilate_mask_larger_radius_is_rounded():
    mask = torch.zeros(1, 1, 11, 11, dtype=torch.bool)
    mask[0, 0, 5, 5] = True
    dilated = _dilate_mask(mask, radius=3)
    # corners of the 7x7 bounding box are outside the disk, so dilation is not square
    assert not dilated[0, 0, 2, 2]
    assert dilated[0, 0, 5, 2] and dilated[0, 0, 2, 5]


def test_phase_flip_preserves_shape_and_binary():
    torch.manual_seed(0)
    phase = (torch.rand(2, 3, 16, 16) > 0.5).to(torch.int32)
    flip = PhaseFlip(active_prob=1.0)
    out = flip(phase)
    assert out.shape == phase.shape
    assert out.dtype == phase.dtype
    assert set(out.unique().tolist()) <= {0, 1}


def test_phase_flip_inactive_returns_input_unchanged():
    phase = (torch.rand(1, 2, 8, 8) > 0.5).to(torch.int32)
    flip = PhaseFlip(active_prob=0.0)
    assert torch.equal(flip(phase), phase)


def test_phase_flip_protects_bubble_interior():
    phase = torch.zeros(1, 1, 16, 16, dtype=torch.int32)
    phase[0, 0, 4:12, 4:12] = 1  # solid vapor block
    flip = PhaseFlip(
        active_prob=1.0,
        interface_flip_prob=1.0,
        bulk_flip_prob=0.0,
        resize_prob=0.0,
        interface_radius=1,
    )
    out = flip(phase)
    # cells more than interface_radius from any liquid cell must stay vapor
    assert out[0, 0, 6:10, 6:10].all()


def test_phase_flip_bulk_nucleation_fills_liquid():
    phase = torch.zeros(1, 1, 16, 16, dtype=torch.int32)  # all liquid, no bubbles
    flip = PhaseFlip(
        active_prob=1.0,
        bulk_flip_prob=1.0,
        interface_flip_prob=0.0,
        resize_prob=0.0,
    )
    assert (flip(phase) == 1).all()


def test_phase_flip_grow_increases_vapor():
    phase = torch.zeros(1, 1, 16, 16, dtype=torch.int32)
    phase[0, 0, 6:10, 6:10] = 1
    flip = PhaseFlip(
        active_prob=1.0,
        bulk_flip_prob=0.0,
        interface_flip_prob=0.0,
        resize_prob=1.0,
        max_resize_radius=1,
    )
    # resize randomly grows or shrinks; over several draws the count must change
    counts = {flip(phase).sum().item() for _ in range(20)}
    assert counts != {16}
