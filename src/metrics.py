"""
This module defines the IoU metrics for evaluating circle overlap.

It includes:
- iou: the standard NumPy-based implementation for evaluating the overlap between two CircleParams instances.
- iou_torch: a PyTorch vectorized implementation for computing the IoU over batches of circle predictions and targets.
"""

# Libraries
import numpy as np
import torch

# Local modules
from src.tools import CircleParams


def iou_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Vectorized batch computation of IoU between two sets of circles using PyTorch.
    Inputs:
        pred (Tensor): Tensor of shape (B, 3), each row as (row, col, radius).
        target (Tensor): Tensor of shape (B, 3), same format as pred.
    Output:
        Tensor: IoU values for each prediction-target pair (B,)
    """
    assert pred.shape == target.shape and pred.shape[1] == 3, "Input must be (B, 3)"

    r1, r2 = pred[:, 2], target[:, 2]
    d = torch.norm(pred[:, :2] - target[:, :2], dim=1)

    no_overlap = d > (r1 + r2)
    inside = d <= torch.abs(r1 - r2)

    smaller = torch.min(r1, r2)
    larger = torch.max(r1, r2)
    iou_inside = (smaller ** 2) / (larger ** 2)

    r1_sq = r1 ** 2
    r2_sq = r2 ** 2
    d1 = (r1_sq - r2_sq + d ** 2) / (2 * d)
    d2 = d - d1

    # Safe clipping for numerical stability
    safe_d1 = torch.clamp(d1 / r1, -1 + 1e-6, 1 - 1e-6)
    safe_d2 = torch.clamp(d2 / r2, -1 + 1e-6, 1 - 1e-6)

    sector_area1 = r1_sq * torch.acos(safe_d1)
    triangle_area1 = d1 * torch.sqrt(r1_sq - d1 ** 2)
    sector_area2 = r2_sq * torch.acos(safe_d2)
    triangle_area2 = d2 * torch.sqrt(r2_sq - d2 ** 2)

    intersection = sector_area1 + sector_area2 - (triangle_area1 + triangle_area2)
    union = torch.pi * (r1_sq + r2_sq) - intersection
    iou_partial = intersection / union

    return torch.where(no_overlap, torch.zeros_like(d),
                       torch.where(inside, iou_inside, iou_partial))


def iou(a: CircleParams, b: CircleParams) -> float:
    """Calculate the intersection over union of two circles"""
    r1, r2 = a.radius, b.radius
    d = np.linalg.norm(np.array([a.row, a.col]) - np.array([b.row, b.col]))
    if d > r1 + r2:
        # If the distance between the centers is greater than the sum of the radii, then the circles don't intersect
        return 0.0
    if d <= abs(r1 - r2):
        # If the distance between the centers is less than the absolute difference of the radii, then one circle is
        # inside the other
        larger_r, smaller_r = max(r1, r2), min(r1, r2)
        return smaller_r ** 2 / larger_r ** 2
    r1_sq, r2_sq = r1**2, r2**2
    d1 = (r1_sq - r2_sq + d**2) / (2 * d)
    d2 = d - d1
    sector_area1 = r1_sq * np.arccos(d1 / r1)
    triangle_area1 = d1 * np.sqrt(r1_sq - d1**2)
    sector_area2 = r2_sq * np.arccos(d2 / r2)
    triangle_area2 = d2 * np.sqrt(r2_sq - d2**2)
    intersection = sector_area1 + sector_area2 - (triangle_area1 + triangle_area2)
    union = np.pi * (r1_sq + r2_sq) - intersection
    return intersection / union