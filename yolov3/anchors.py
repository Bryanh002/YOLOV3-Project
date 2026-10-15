"""Dimension clusters for YOLOv3 anchors.

YOLOv3 keeps the YOLOv2 prior boxes. Those priors are not a fixed table in the
training procedure: k-means is run on the ground-truth widths and heights, and
the distance is `1 - IoU` of the two boxes when both sit at the origin. The
nine published COCO anchors are that clustering, already scaled to a 416 input.
Training from scratch on another label set should cluster that set instead of
reusing the COCO numbers.
"""

from __future__ import annotations

from pathlib import Path

import torch

from yolov3.ops import wh_iou


def kmeans_anchors(
    widths_heights: torch.Tensor,
    k: int = 9,
    iterations: int = 25,
    seed: int = 0,
) -> tuple[tuple[float, float], ...]:
    """Return `k` anchors sorted from smallest area to largest.

    `widths_heights` is (N, 2) in any consistent unit. Pixels at the network
    input size match the unit used by the published COCO anchors.
    """
    if widths_heights.ndim != 2 or widths_heights.shape[1] != 2:
        raise ValueError("widths_heights must have shape (N, 2)")
    boxes = widths_heights.detach().float().cpu()
    keep = torch.isfinite(boxes).all(dim=1) & (boxes > 0).all(dim=1)
    boxes = boxes[keep]
    if boxes.shape[0] == 0:
        raise ValueError("Anchor clustering needs at least one positive box")
    clusters = min(int(k), int(boxes.shape[0]))
    if clusters < 1:
        raise ValueError("k must be positive")

    generator = torch.Generator().manual_seed(seed)
    initial = torch.randperm(boxes.shape[0], generator=generator)[:clusters]
    centers = boxes[initial].clone()

    for _ in range(iterations):
        overlap = wh_iou(boxes, centers)
        assignment = overlap.argmax(dim=1)
        updated = []
        for index in range(clusters):
            members = boxes[assignment == index]
            if members.numel() == 0:
                # An empty cluster is restarted on the box its current center fits worst.
                worst = int(overlap.max(dim=1).values.argmin())
                updated.append(boxes[worst])
            else:
                updated.append(members.mean(dim=0))
        new_centers = torch.stack(updated, dim=0)
        if torch.allclose(new_centers, centers, atol=1e-3):
            centers = new_centers
            break
        centers = new_centers

    order = torch.argsort(centers[:, 0] * centers[:, 1])
    centers = centers[order]
    return tuple((float(width), float(height)) for width, height in centers)


def mean_best_iou(widths_heights: torch.Tensor, anchors) -> float:
    boxes = widths_heights.detach().float().cpu()
    priors = torch.tensor(anchors, dtype=torch.float32)
    return float(wh_iou(boxes, priors).max(dim=1).values.mean())


def normalized_wh_from_labels(label_dir: str | Path) -> torch.Tensor:
    """Read YOLO label files and return normalized (width, height) rows."""
    rows: list[tuple[float, float]] = []
    folder = Path(label_dir)
    paths = sorted(folder.glob("*.txt"))
    if not paths:
        raise FileNotFoundError(f"No label files in {folder}")
    for path in paths:
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            width, height = float(parts[3]), float(parts[4])
            if width > 0 and height > 0:
                rows.append((width, height))
    if not rows:
        raise ValueError(f"No boxes found in {folder}")
    return torch.tensor(rows, dtype=torch.float32)


def anchors_from_labels(label_dir: str | Path, img_size: int, k: int = 9, seed: int = 0):
    """Cluster label boxes into `k` priors measured in network-input pixels."""
    normalized = normalized_wh_from_labels(label_dir)
    return kmeans_anchors(normalized * float(img_size), k=k, seed=seed)
