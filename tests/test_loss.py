"""Anchor assignment, the ignore mask, and a short overfit of the training loss."""

import torch

from yolov3.config import ANCHOR_MASKS, ANCHORS
from yolov3.loss import YOLOv3Loss
from yolov3.model import YOLOv3
from yolov3.ops import wh_iou


def _predictions(batch=1, num_classes=80, size=416):
    grids = (size // 32, size // 16, size // 8)
    channels = 3 * (5 + num_classes)
    return [torch.zeros(batch, channels, grid, grid) for grid in grids]


def _best_anchor(width, height):
    ious = wh_iou(torch.tensor([[width, height]]), torch.tensor(ANCHORS, dtype=torch.float32))
    return int(ious.argmax())


def test_one_ground_truth_box_is_assigned_to_a_single_anchor():
    criterion = YOLOv3Loss(80, ANCHORS, ANCHOR_MASKS)
    large = torch.tensor([[0.0, 16.0, 0.5, 0.5, 300 / 416, 300 / 416]])
    criterion(_predictions(), large, 416)
    assert criterion.last_positive_counts == [1, 0, 0]
    assert _best_anchor(300, 300) == 8
    assert 8 in ANCHOR_MASKS[0]

    small = torch.tensor([[0.0, 0.0, 0.25, 0.75, 12 / 416, 14 / 416]])
    criterion(_predictions(), small, 416)
    assert criterion.last_positive_counts == [0, 0, 1]
    assert _best_anchor(12, 14) == 0
    assert 0 in ANCHOR_MASKS[2]


def test_ignore_mask_drops_predictions_that_already_overlap_an_object():
    criterion = YOLOv3Loss(1, ANCHORS, ANCHOR_MASKS, ignore_thresh=0.5)
    noobj = torch.ones(1, 1, 1, 2, dtype=torch.bool)
    pred_xyxy = torch.tensor([[[[[30.0, 30.0, 70.0, 70.0], [0.0, 0.0, 5.0, 5.0]]]]])
    # Normalized center 0.5, size 0.4 on a 100px canvas is the first box.
    targets = torch.tensor([[0.0, 0.0, 0.5, 0.5, 0.4, 0.4]])
    criterion._apply_ignore(noobj, pred_xyxy, targets, 100)
    assert bool(noobj[0, 0, 0, 0]) is False
    assert bool(noobj[0, 0, 0, 1]) is True


def test_empty_targets_still_backpropagate():
    criterion = YOLOv3Loss(2, ANCHORS, ANCHOR_MASKS)
    predictions = [tensor.requires_grad_(True) for tensor in _predictions(num_classes=2, size=128)]
    loss = criterion(predictions, torch.zeros(0, 6), 128)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(tensor.grad is not None and torch.isfinite(tensor.grad).all() for tensor in predictions)


def test_box_loss_falls_when_one_example_is_overfit():
    torch.manual_seed(0)
    model = YOLOv3(num_classes=1).train()
    criterion = YOLOv3Loss(1, ANCHORS, ANCHOR_MASKS, ignore_thresh=0.7)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    image = torch.rand(1, 3, 128, 128)
    target = torch.tensor([[0.0, 0.0, 0.5, 0.5, 0.45, 0.45]])
    box_losses = []
    for _ in range(12):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(image), target, 128)
        loss.backward()
        optimizer.step()
        box_losses.append(criterion.last_parts["xy"] + criterion.last_parts["wh"])
    assert box_losses[-1] < box_losses[0] * 0.75, box_losses
