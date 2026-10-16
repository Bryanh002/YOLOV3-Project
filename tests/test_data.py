"""Label geometry, COCO id mapping, and the training schedule."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from yolov3.config import COCO_CATEGORY_IDS, COCO_CLASSES
from yolov3.data import (
    COCODataset,
    YOLODataset,
    apply_letterbox,
    collate_yolo,
    horizontal_flip,
    load_dataset_yaml,
    prepare_sample,
    read_yolo_label,
)
from yolov3.model import YOLOv3
from yolov3.train import learning_rate_at, sgd_param_groups


def test_letterbox_box_stays_on_the_painted_object():
    image = Image.new("RGB", (200, 100), (0, 0, 0))
    for x in range(40, 120):
        for y in range(20, 70):
            image.putpixel((x, y), (255, 255, 255))
    boxes = np.array([[40, 20, 120, 70]], dtype=np.float32)
    canvas, moved, meta = apply_letterbox(image, boxes, 128)
    box = moved[0]
    cx = int((box[0] + box[2]) / 2)
    cy = int((box[1] + box[3]) / 2)
    assert canvas.getpixel((cx, cy))[0] > 200
    # The top-left pixel is letterbox padding, which Darknet fills with mid-gray.
    assert canvas.getpixel((0, 0)) == (128, 128, 128)
    # A pixel on the resized background, outside the white rectangle, stays dark.
    assert canvas.getpixel((2, meta.pad_y + 2))[0] < 20


def test_horizontal_flip_mirrors_a_box():
    image = Image.new("RGB", (100, 40), (1, 2, 3))
    boxes = np.array([[10.0, 5.0, 30.0, 25.0]])
    flipped, moved = horizontal_flip(image, boxes)
    assert flipped.size == (100, 40)
    assert moved.tolist() == [[70.0, 5.0, 90.0, 25.0]]


def test_yolo_dataset_and_collate(tmp_path: Path):
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    Image.new("RGB", (80, 60), (20, 30, 40)).save(images / "one.png")
    (labels / "one.txt").write_text("1 0.5 0.5 0.25 0.5\n")
    (tmp_path / "dataset.yaml").write_text(
        yaml.safe_dump({"path": ".", "train": "images", "val": "images", "names": ["disc", "tile"]})
    )
    spec = load_dataset_yaml(tmp_path / "dataset.yaml")
    assert spec["names"] == ["disc", "tile"]
    dataset = YOLODataset(spec["root"], augment=False, img_size=128)
    image, boxes, classes = dataset[0]
    assert image.shape == (3, 128, 128)
    assert float(image.max()) <= 1.0
    assert classes.tolist() == [1]
    assert boxes.shape == (1, 4)
    # The source box is centered, so letterbox padding keeps the center near 0.5, 0.5.
    assert float(boxes[0, 0]) == pytest.approx(0.5, abs=0.02)
    batch_images, targets = collate_yolo([(image, boxes, classes), (image, boxes[:0], classes[:0])])
    assert batch_images.shape[0] == 2
    assert targets.shape == (1, 6)
    assert int(targets[0, 0]) == 0
    loaded_boxes, loaded_classes = read_yolo_label(labels / "one.txt", (80, 60))
    assert loaded_classes.tolist() == [1]
    assert loaded_boxes.shape == (1, 4)


def test_prepare_sample_without_augmentation_is_deterministic():
    image = Image.new("RGB", (64, 32), (255, 0, 0))
    boxes = np.array([[0.0, 0.0, 64.0, 32.0]], dtype=np.float32)
    classes = np.array([0], dtype=np.int64)
    first = prepare_sample(image, boxes, classes, 128, augment=False)
    second = prepare_sample(image, boxes, classes, 128, augment=False)
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])


def test_coco_category_ids_line_up_with_class_names(tmp_path: Path):
    assert len(COCO_CLASSES) == 80
    assert len(COCO_CATEGORY_IDS) == 80
    assert COCO_CLASSES[COCO_CATEGORY_IDS.index(1)] == "person"
    assert COCO_CLASSES[COCO_CATEGORY_IDS.index(18)] == "dog"
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (40, 30), (8, 8, 8)).save(image_dir / "a.jpg")
    payload = {
        "images": [{"id": 1, "file_name": "a.jpg", "width": 40, "height": 30}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 18, "bbox": [5, 5, 10, 8], "iscrowd": 0},
            {"id": 2, "image_id": 1, "category_id": 18, "bbox": [0, 0, 4, 4], "iscrowd": 1},
            {"id": 3, "image_id": 1, "category_id": 12, "bbox": [1, 1, 4, 4], "iscrowd": 0},
        ],
    }
    ann = tmp_path / "instances.json"
    ann.write_text(json.dumps(payload))
    dataset = COCODataset(image_dir, ann, img_size=128, augment=False)
    _image, boxes, classes = dataset[0]
    assert classes.tolist() == [COCO_CLASSES.index("dog")]
    assert boxes.shape[0] == 1


def test_learning_rate_follows_burn_in_and_the_step_schedule():
    assert learning_rate_at(500, 0.001, 1000) == pytest.approx(0.001 * (0.5 ** 4))
    assert learning_rate_at(1000, 0.001, 1000) == pytest.approx(0.001)
    assert learning_rate_at(400000, 0.001, 1000) == pytest.approx(0.0001)
    assert learning_rate_at(450000, 0.001, 1000) == pytest.approx(0.00001)


def test_weight_decay_skips_biases_and_batch_norm():
    model = YOLOv3()
    groups = sgd_param_groups(model, 5e-4)
    decaying = {id(param) for param in groups[0]["params"]}
    assert id(model.backbone.stem.conv.weight) in decaying
    assert id(model.backbone.stem.bn.weight) not in decaying
    assert id(model.head1[1].conv.bias) not in decaying
    assert groups[0]["weight_decay"] == 5e-4
    assert groups[1]["weight_decay"] == 0.0
