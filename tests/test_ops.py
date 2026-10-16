"""Decoding, letterbox geometry, and class-wise NMS."""

import pytest
import torch
from PIL import Image

from yolov3.config import ANCHORS
from yolov3.model import ScaleSpec
from yolov3.ops import (
    box_iou,
    decode_predictions,
    decode_scale,
    greedy_nms,
    letterbox_image,
    non_max_suppression,
    undo_letterbox,
    wh_iou,
    xywh_to_xyxy,
)


def test_iou_matches_a_hand_computed_overlap():
    same = torch.tensor([[0.0, 0.0, 10.0, 20.0], [2.0, 2.0, 8.0, 10.0]])
    overlap = box_iou(same, same)
    assert torch.allclose(overlap.diag(), torch.ones(2))
    # 10x10 and 10x5 share a 10x5 region: intersection 50, union 100.
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    other = torch.tensor([[0.0, 0.0, 10.0, 5.0]])
    assert torch.allclose(box_iou(boxes, other), torch.tensor([[0.5]]))
    wh = wh_iou(torch.tensor([[10.0, 20.0]]), torch.tensor([[10.0, 20.0], [10.0, 10.0]]))
    assert float(wh[0, 0]) == pytest.approx(1.0)
    assert float(wh[0, 1]) == pytest.approx(0.5)


def test_decode_uses_the_paper_box_equations():
    raw = torch.zeros(1, 3 * 85, 13, 13)
    decoded = decode_scale(raw, ANCHORS, (6, 7, 8), 416)
    # sigmoid(0) = 0.5 and the stride is 32, so the corner cell sits at (16, 16).
    # exp(0) leaves the anchor size unchanged.
    xywh = decoded[0, 0, 0, 0, :4]
    assert torch.allclose(xywh, torch.tensor([16.0, 16.0, 116.0, 90.0]), atol=1e-4)
    assert torch.allclose(decoded[0, :, :, :, 4], torch.full((3, 13, 13), 0.5))


def test_nms_is_per_class_and_suppresses_overlaps():
    prediction = torch.tensor(
        [[
            [15.0, 15.0, 10.0, 10.0, 0.95, 0.90, 0.05, 0.0],
            [16.0, 16.0, 10.0, 10.0, 0.90, 0.85, 0.05, 0.0],
            [16.0, 16.0, 10.0, 10.0, 0.90, 0.05, 0.80, 1.0],
        ]]
    )
    kept = non_max_suppression(prediction, conf_thresh=0.25, iou_thresh=0.4)[0]
    classes = sorted(int(class_id) for class_id in kept[:, 5])
    assert classes == [0, 1]
    assert float(kept[:, 4].max()) == pytest.approx(0.95 * 0.90)


def test_greedy_nms_keeps_disjoint_boxes():
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
    scores = torch.tensor([0.2, 0.9])
    assert greedy_nms(boxes, scores, 0.5) == [1, 0]


def test_letterbox_preserves_aspect_and_roundtrips_a_box():
    image = Image.new("RGB", (200, 100), (0, 0, 0))
    canvas, meta = letterbox_image(image, 128)
    assert canvas.size == (128, 128)
    assert meta.scale_x == pytest.approx(128 / 200)
    assert meta.pad_x == 0
    assert meta.pad_y == (128 - 64) // 2
    source = torch.tensor([[0.0, 0.0, 200.0, 100.0]])
    network = source.clone()
    network[:, [0, 2]] = source[:, [0, 2]] * meta.scale_x + meta.pad_x
    network[:, [1, 3]] = source[:, [1, 3]] * meta.scale_y + meta.pad_y
    restored = undo_letterbox(network, meta, image.size)
    assert torch.allclose(restored, source, atol=1e-4)


def test_decode_predictions_tags_each_scale():
    specs = [
        ScaleSpec((6, 7, 8), 0, "large"),
        ScaleSpec((3, 4, 5), 1, "medium"),
        ScaleSpec((0, 1, 2), 2, "small"),
    ]
    raw = [torch.zeros(1, 255, 4, 4), torch.zeros(1, 255, 8, 8), torch.zeros(1, 255, 16, 16)]
    decoded = decode_predictions(raw, specs, 128)
    assert decoded.shape[0] == 1
    assert decoded.shape[-1] == 6 + 80
    scales = {int(value) for value in decoded[0, :, -1].unique()}
    assert scales == {0, 1, 2}
    assert torch.isfinite(xywh_to_xyxy(decoded[..., :4])).all()
