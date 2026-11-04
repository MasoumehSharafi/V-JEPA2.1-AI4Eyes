"""A from-scratch, anchor-free temporal action detection head, in the style
of ActionFormer (Zhang et al., 2022): a multi-scale 1D feature pyramid,
per-level shared conv classification/regression towers, point-based label
assignment with center sampling, focal loss + DIoU loss, and greedy 1D NMS.

This is NOT a copy of any specific codebase's implementation -- it exists
so this repo has zero dependency on any non-MIT detection framework. It
reproduces the same high-level architecture used to produce the results
in ../README.md, but is a fresh implementation, so exact metric
reproduction may differ slightly from those numbers.
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_STRIDES = (1, 2, 4, 8, 16, 32)
DEFAULT_REGRESSION_RANGES = ((0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 10_000))


class FPN1D(nn.Module):
    """Projects a single-scale feature sequence [B,C_in,T] into a pyramid of
    `len(strides)` levels via repeated stride-2 conv downsampling, each
    projected to `out_channels`."""

    def __init__(self, in_channels: int, out_channels: int = 512, num_levels: int = 6):
        super().__init__()
        self.in_proj = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.downsamples = nn.ModuleList(
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
            for _ in range(num_levels - 1)
        )
        self.norms = nn.ModuleList(nn.GroupNorm(32, out_channels) for _ in range(num_levels))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """x: [B, C_in, T] -> list of [B, out_channels, T_l], T_l halving each level."""
        feat = F.relu(self.in_proj(x))
        levels = [self.norms[0](feat)]
        for i, down in enumerate(self.downsamples):
            feat = F.relu(down(feat))
            levels.append(self.norms[i + 1](feat))
        return levels


class ActionFormerHead(nn.Module):
    """Shared conv towers producing per-point classification logits and
    (left, right) regression distances, applied independently at every
    pyramid level."""

    def __init__(self, in_channels: int, num_classes: int, num_convs: int = 2, prior_prob: float = 0.01):
        super().__init__()
        self.num_classes = num_classes

        def tower():
            layers = []
            for _ in range(num_convs):
                layers += [nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
            return nn.Sequential(*layers)

        self.cls_tower = tower()
        self.reg_tower = tower()
        self.cls_logits = nn.Conv1d(in_channels, num_classes, kernel_size=3, padding=1)
        self.reg_offsets = nn.Conv1d(in_channels, 2, kernel_size=3, padding=1)

        # Standard focal-loss prior-probability init so training doesn't
        # start with an enormous classification loss from all-positive bias.
        bias_value = -torch.log(torch.tensor((1 - prior_prob) / prior_prob))
        nn.init.constant_(self.cls_logits.bias, bias_value.item())

    def forward(self, levels: list[torch.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        cls_outputs, reg_outputs = [], []
        for feat in levels:
            cls_outputs.append(self.cls_logits(self.cls_tower(feat)))  # [B, num_classes, T_l]
            reg_outputs.append(F.relu(self.reg_offsets(self.reg_tower(feat))))  # [B, 2, T_l], distances >= 0
        return cls_outputs, reg_outputs


def generate_points(level_lengths: list[int], strides: tuple[int, ...], device) -> list[torch.Tensor]:
    """Per level, point centers in the base (level-0) temporal coordinate
    system: stride/2, stride*1.5, stride*2.5, ... """
    points = []
    for length, stride in zip(level_lengths, strides):
        centers = (torch.arange(length, device=device, dtype=torch.float32) + 0.5) * stride
        points.append(centers)
    return points


@dataclass
class Target:
    """One ground-truth instance in window-local FRAME coordinates."""

    start: float
    end: float
    label: int


def assign_targets(
    points: list[torch.Tensor],
    regression_ranges: tuple[tuple[float, float], ...],
    strides: tuple[int, ...],
    targets: list[Target],
    num_classes: int,
    center_sample_radius: float = 1.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns, concatenated across all levels:
        cls_targets: [sum_T, num_classes] multi-hot (0/1)
        reg_targets: [sum_T, 2] (left, right) distances, meaningful only where positive
        pos_mask:    [sum_T] bool, True at points assigned to at least one target
    """
    all_points = torch.cat(points)  # [sum_T]
    all_ranges = torch.cat(
        [
            torch.tensor([lo, hi], dtype=torch.float32, device=all_points.device).expand(len(p), 2)
            for p, (lo, hi) in zip(points, regression_ranges)
        ]
    )
    all_strides = torch.cat(
        [torch.full((len(p),), stride, dtype=torch.float32, device=all_points.device) for p, stride in zip(points, strides)]
    )
    n_points = all_points.shape[0]
    cls_targets = torch.zeros(n_points, num_classes, device=all_points.device)
    reg_targets = torch.zeros(n_points, 2, device=all_points.device)
    pos_mask = torch.zeros(n_points, dtype=torch.bool, device=all_points.device)
    best_area = torch.full((n_points,), float("inf"), device=all_points.device)

    for target in targets:
        center = (target.start + target.end) / 2
        radius = center_sample_radius * all_strides
        in_center = (all_points >= center - radius) & (all_points <= center + radius)
        left = all_points - target.start
        right = target.end - all_points
        in_segment = (left >= 0) & (right >= 0)
        max_dist = torch.maximum(left, right)
        in_range = (max_dist >= all_ranges[:, 0]) & (max_dist < all_ranges[:, 1])
        eligible = in_center & in_segment & in_range
        area = target.end - target.start
        better = eligible & (area < best_area)
        if better.any():
            best_area = torch.where(better, torch.full_like(best_area, area), best_area)
            reg_targets[better] = torch.stack([left[better], right[better]], dim=1)
            cls_targets[better] = 0  # a closer (smaller-area) target overrides any previous assignment
            cls_targets[better, target.label] = 1
            pos_mask = pos_mask | better

    return cls_targets, reg_targets, pos_mask


def sigmoid_focal_loss(logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.sum()


def diou_loss_1d(pred_lr: torch.Tensor, target_lr: torch.Tensor) -> torch.Tensor:
    """1D Distance-IoU loss on (left, right) distances from a shared point,
    summed (not averaged) over the batch of positive points."""
    pred_len = pred_lr[:, 0] + pred_lr[:, 1]
    target_len = target_lr[:, 0] + target_lr[:, 1]
    inter = torch.minimum(pred_lr[:, 0], target_lr[:, 0]) + torch.minimum(pred_lr[:, 1], target_lr[:, 1])
    inter = inter.clamp(min=0)
    union = pred_len + target_len - inter
    iou = inter / union.clamp(min=1e-6)

    enclosing = torch.maximum(pred_lr[:, 0], target_lr[:, 0]) + torch.maximum(pred_lr[:, 1], target_lr[:, 1])
    center_dist = (pred_lr[:, 0] - target_lr[:, 0] - (pred_lr[:, 1] - target_lr[:, 1])).abs() / 2
    diou = iou - (center_dist**2) / (enclosing.clamp(min=1e-6) ** 2)
    return (1 - diou).sum()


def nms_1d(segments: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5, max_segments: int = 2000) -> torch.Tensor:
    """Greedy 1D NMS. segments: [N,2] (start,end); scores: [N]. Returns
    kept indices, sorted by descending score. Simpler than soft-NMS, but
    standard and well-understood -- see heads.py module docstring."""
    if segments.numel() == 0:
        return torch.empty((0,), dtype=torch.int64)
    order = torch.argsort(scores, descending=True)
    keep = []
    while order.numel() > 0 and len(keep) < max_segments:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        inter = (
            torch.minimum(segments[i, 1], segments[rest, 1]) - torch.maximum(segments[i, 0], segments[rest, 0])
        ).clamp(min=0)
        union = (segments[i, 1] - segments[i, 0]) + (segments[rest, 1] - segments[rest, 0]) - inter
        iou = inter / union.clamp(min=1e-6)
        order = rest[iou <= iou_threshold]
    return torch.tensor(keep, dtype=torch.int64)
