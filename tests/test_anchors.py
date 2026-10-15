"""IoU k-means, the procedure YOLOv3 uses to choose its nine priors."""

from pathlib import Path

import torch

from yolov3.anchors import anchors_from_labels, kmeans_anchors, mean_best_iou
from yolov3.model import YOLOv3


def test_kmeans_recovers_three_box_sizes():
    generator = torch.Generator().manual_seed(1)
    groups = [
        torch.tensor([12.0, 14.0]) + torch.randn(40, 2, generator=generator),
        torch.tensor([48.0, 30.0]) + torch.randn(40, 2, generator=generator),
        torch.tensor([90.0, 80.0]) + torch.randn(40, 2, generator=generator),
    ]
    boxes = torch.cat(groups, dim=0).clamp(min=1)
    anchors = kmeans_anchors(boxes, k=3, iterations=30, seed=2)
    assert len(anchors) == 3
    areas = [width * height for width, height in anchors]
    assert areas == sorted(areas)
    # Each cluster center should land near the size that generated it.
    expected = [(12.0, 14.0), (48.0, 30.0), (90.0, 80.0)]
    for width, height in expected:
        nearest = min(anchors, key=lambda anchor: abs(anchor[0] - width) + abs(anchor[1] - height))
        assert abs(nearest[0] - width) < 8
        assert abs(nearest[1] - height) < 8
    assert mean_best_iou(boxes, anchors) > 0.7


def test_kmeans_is_deterministic():
    boxes = torch.tensor([[10.0, 10.0], [11.0, 12.0], [40.0, 20.0], [42.0, 18.0], [80.0, 70.0], [78.0, 75.0]])
    assert kmeans_anchors(boxes, k=3, seed=4) == kmeans_anchors(boxes, k=3, seed=4)


def test_label_files_are_clustered_in_network_pixels(tmp_path: Path):
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "a.txt").write_text("0 0.5 0.5 0.25 0.25\n0 0.2 0.2 0.10 0.10\n")
    anchors = anchors_from_labels(labels, img_size=200, k=2, seed=0)
    assert len(anchors) == 2
    # 0.10 * 200 = 20 and 0.25 * 200 = 50, smallest first.
    assert anchors[0][0] < anchors[1][0]


def test_custom_anchors_are_what_the_heads_decode_with():
    anchors = tuple((float(8 * (index + 1)), float(6 * (index + 1))) for index in range(9))
    model = YOLOv3(num_classes=3, anchors=anchors)
    assert model.anchors == anchors
    assert model.yolo_layers[2].anchors == anchors
    # The mask for small objects is the first three clusters.
    assert model.yolo_layers[2].mask == (0, 1, 2)
