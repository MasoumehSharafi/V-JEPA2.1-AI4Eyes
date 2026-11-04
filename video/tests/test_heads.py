"""Shape and logic tests for the reimplemented detection head. CPU only,
random tensors -- no checkpoint, no video, no GPU required."""
import pytest
import torch

from ai4eyes_video.heads import (
    ActionFormerHead,
    FPN1D,
    Target,
    assign_targets,
    diou_loss_1d,
    generate_points,
    nms_1d,
    sigmoid_focal_loss,
)

STRIDES = (1, 2, 4, 8, 16, 32)
REGRESSION_RANGES = ((0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 10_000))


def test_fpn1d_produces_halving_levels():
    fpn = FPN1D(in_channels=64, out_channels=32, num_levels=6)
    x = torch.randn(2, 64, 160)
    levels = fpn(x)
    assert len(levels) == 6
    expected_lengths = [160, 80, 40, 20, 10, 5]
    for level, expected in zip(levels, expected_lengths):
        assert level.shape == (2, 32, expected)


def test_action_former_head_shapes():
    fpn = FPN1D(in_channels=64, out_channels=32, num_levels=6)
    head = ActionFormerHead(in_channels=32, num_classes=2)
    levels = fpn(torch.randn(2, 64, 160))
    cls_outputs, reg_outputs = head(levels)
    assert len(cls_outputs) == len(reg_outputs) == 6
    for cls, reg, level in zip(cls_outputs, reg_outputs, levels):
        assert cls.shape == (2, 2, level.shape[-1])
        assert reg.shape == (2, 2, level.shape[-1])
        assert (reg >= 0).all()  # regression distances are ReLU'd


def test_generate_points_lengths_and_spacing():
    points = generate_points([160, 80, 40], (1, 2, 4), device=torch.device("cpu"))
    assert [p.shape[0] for p in points] == [160, 80, 40]
    assert points[0][0].item() == 0.5  # first level-0 point center = stride/2
    assert points[1][0].item() == 1.0  # first level-1 point center = stride/2 = 2/2


def test_assign_targets_marks_positives_near_target():
    points = generate_points([160], (1,), device=torch.device("cpu"))
    targets = [Target(start=10.0, end=14.0, label=0)]
    cls_t, reg_t, pos_mask = assign_targets(points, ((0, 4),), (1,), targets, num_classes=1)
    assert pos_mask.any(), "at least one point should be assigned to the target"
    assert cls_t[pos_mask, 0].min() == 1
    assert (reg_t[pos_mask] >= 0).all()
    # Points far from the target must not be positive.
    assert not pos_mask[0].item()  # point near t=0.5, far from [10,14]


def test_sigmoid_focal_loss_is_near_zero_for_confident_correct_logits():
    targets = torch.tensor([[1.0, 0.0]])
    confident_correct = torch.tensor([[10.0, -10.0]])
    confident_wrong = torch.tensor([[-10.0, 10.0]])
    loss_correct = sigmoid_focal_loss(confident_correct, targets)
    loss_wrong = sigmoid_focal_loss(confident_wrong, targets)
    assert loss_correct.item() < loss_wrong.item()
    assert loss_correct.item() < 1e-2


def test_diou_loss_zero_for_identical_boxes():
    pred = torch.tensor([[2.0, 3.0]])
    target = torch.tensor([[2.0, 3.0]])
    loss = diou_loss_1d(pred, target)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_nms_1d_removes_overlapping_lower_score():
    segments = torch.tensor([[0.0, 10.0], [1.0, 11.0], [50.0, 60.0]])
    scores = torch.tensor([0.9, 0.8, 0.95])
    keep = nms_1d(segments, scores, iou_threshold=0.5)
    kept_indices = set(keep.tolist())
    assert 2 in kept_indices  # the far-away, highest-score segment survives
    assert 0 in kept_indices  # the higher-scoring of the two overlapping segments survives
    assert 1 not in kept_indices  # suppressed by segment 0 (high overlap, lower score)


def test_nms_1d_empty_input():
    keep = nms_1d(torch.zeros((0, 2)), torch.zeros((0,)))
    assert keep.numel() == 0
