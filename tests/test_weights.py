"""Darknet weight round trip, and a detection check when official weights are present."""

from pathlib import Path

import pytest
import torch
from PIL import Image

from yolov3.config import DARKNET_WEIGHTS_SHA256
from yolov3.engine import Detector
from yolov3.model import YOLOv3
from yolov3.weights import darknet_weight_count, load_darknet_weights, save_darknet_weights

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights" / "yolov3.weights"
DOG = ROOT / "samples" / "photos" / "park_dog.jpg"


def test_darknet_roundtrip_restores_every_parameter(tmp_path: Path):
    torch.manual_seed(1)
    source = YOLOv3(num_classes=80)
    path = tmp_path / "roundtrip.weights"
    save_darknet_weights(source, path, seen=7)
    restored = YOLOv3(num_classes=80)
    header = load_darknet_weights(restored, path)
    assert header["seen"] == 7
    assert header["floats"] == darknet_weight_count(source)
    for first, second in zip(source.parameters(), restored.parameters()):
        assert torch.equal(first, second)
    for first, second in zip(source.buffers(), restored.buffers()):
        assert torch.equal(first, second)


@pytest.mark.skipif(not WEIGHTS.is_file(), reason="official yolov3.weights is not downloaded")
def test_official_weights_detect_a_dog():
    import hashlib

    digest = hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()
    assert digest == DARKNET_WEIGHTS_SHA256
    detector = Detector(WEIGHTS)
    assert detector.model.training is False
    image = Image.open(DOG)
    detections = detector.detect(image, conf_thresh=0.25, iou_thresh=0.45, img_size=416)
    names = [det["class_name"] for det in detections]
    assert "dog" in names, names
    dog = max((det for det in detections if det["class_name"] == "dog"), key=lambda det: det["confidence"])
    assert dog["confidence"] > 0.5
    assert dog["x2"] > dog["x1"]
    assert dog["scale"] in {"large", "medium", "small"}
