"""YOLO-format and COCO datasets, plus the training augmentations from the config.

Images are resized with letterboxing rather than stretched. Pixels are scaled
to 0..1 and are not ImageNet-normalized. Augmentation is horizontal flip,
aspect jitter, and the hue / saturation / exposure shifts from the training
config.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset

from yolov3.config import COCO_CATEGORY_IDS, EXPOSURE, HUE, JITTER, SATURATION
from yolov3.ops import Letterbox, letterbox_image, pil_to_tensor

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(folder: Path) -> list[Path]:
    return sorted(path for path in Path(folder).rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def read_yolo_label(path: Path, image_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Load a label file into pixel xyxy boxes and integer class ids."""
    width, height = image_size
    boxes = []
    classes = []
    if path.is_file():
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            class_id = int(float(parts[0]))
            cx, cy, bw, bh = (float(value) for value in parts[1:5])
            boxes.append(
                [
                    (cx - bw / 2) * width,
                    (cy - bh / 2) * height,
                    (cx + bw / 2) * width,
                    (cy + bh / 2) * height,
                ]
            )
            classes.append(class_id)
    if not boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.asarray(boxes, dtype=np.float32), np.asarray(classes, dtype=np.int64)


def horizontal_flip(image: Image.Image, boxes: np.ndarray) -> tuple[Image.Image, np.ndarray]:
    width, _height = image.size
    flipped = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if boxes.size == 0:
        return flipped, boxes
    out = boxes.copy()
    out[:, 0] = width - boxes[:, 2]
    out[:, 2] = width - boxes[:, 0]
    return flipped, out


def apply_letterbox(image: Image.Image, boxes: np.ndarray, size: int) -> tuple[Image.Image, np.ndarray, Letterbox]:
    canvas, meta = letterbox_image(image, size)
    if boxes.size == 0:
        return canvas, boxes.reshape(0, 4), meta
    out = boxes.copy()
    out[:, [0, 2]] = boxes[:, [0, 2]] * meta.scale_x + meta.pad_x
    out[:, [1, 3]] = boxes[:, [1, 3]] * meta.scale_y + meta.pad_y
    return canvas, out, meta


def jitter_place(
    image: Image.Image,
    boxes: np.ndarray,
    size: int,
    jitter: float = JITTER,
) -> tuple[Image.Image, np.ndarray]:
    """Aspect jitter followed by a random crop or pad onto the square canvas.

    This is the training-time counterpart of letterboxing. Boxes are moved
    with the same scale and translation as the pixels.
    """
    width, height = image.size
    new_w = max(1.0, width + random.uniform(-width * jitter, width * jitter))
    new_h = max(1.0, height + random.uniform(-height * jitter, height * jitter))
    scale = size / max(new_w, new_h)
    scale *= random.uniform(0.8, 1.2)
    resized_w = max(1, int(round(width * scale * (new_w / width))))
    resized_h = max(1, int(round(height * scale * (new_h / height))))
    resized = image.resize((resized_w, resized_h), Image.Resampling.BILINEAR)
    if boxes.size:
        out = boxes.copy()
        out[:, [0, 2]] *= resized_w / width
        out[:, [1, 3]] *= resized_h / height
    else:
        out = boxes.reshape(0, 4)

    if resized_w > size:
        pad_x = random.randint(size - resized_w, 0)
    else:
        pad_x = random.randint(0, size - resized_w)
    if resized_h > size:
        pad_y = random.randint(size - resized_h, 0)
    else:
        pad_y = random.randint(0, size - resized_h)

    canvas = Image.new("RGB", (size, size), (128, 128, 128))
    canvas.paste(resized, (pad_x, pad_y))
    if out.size:
        out[:, [0, 2]] += pad_x
        out[:, [1, 3]] += pad_y
    return canvas, out


def distort_hsv(
    image: Image.Image,
    hue: float = HUE,
    saturation: float = SATURATION,
    exposure: float = EXPOSURE,
) -> Image.Image:
    hsv = np.array(image.convert("HSV"), dtype=np.float32)
    hsv[..., 0] = (hsv[..., 0] + random.uniform(-hue, hue) * 255.0) % 256.0
    if saturation > 1:
        hsv[..., 1] *= random.uniform(1.0 / saturation, saturation)
    if exposure > 1:
        hsv[..., 2] *= random.uniform(1.0 / exposure, exposure)
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2], 0, 255)
    return Image.fromarray(hsv.astype(np.uint8), mode="HSV").convert("RGB")


def clip_boxes(
    boxes: np.ndarray,
    classes: np.ndarray,
    size: int,
    min_pixels: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    if boxes.size == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    clipped = boxes.copy()
    clipped[:, [0, 2]] = np.clip(clipped[:, [0, 2]], 0, size)
    clipped[:, [1, 3]] = np.clip(clipped[:, [1, 3]], 0, size)
    wh = clipped[:, 2:4] - clipped[:, 0:2]
    keep = (wh[:, 0] >= min_pixels) & (wh[:, 1] >= min_pixels)
    return clipped[keep].astype(np.float32), classes[keep]


def xyxy_to_norm_xywh(boxes: np.ndarray, size: int) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    xywh = np.empty_like(boxes, dtype=np.float32)
    xywh[:, 0] = ((boxes[:, 0] + boxes[:, 2]) / 2) / size
    xywh[:, 1] = ((boxes[:, 1] + boxes[:, 3]) / 2) / size
    xywh[:, 2] = (boxes[:, 2] - boxes[:, 0]) / size
    xywh[:, 3] = (boxes[:, 3] - boxes[:, 1]) / size
    return xywh


def prepare_sample(
    image: Image.Image,
    boxes: np.ndarray,
    classes: np.ndarray,
    img_size: int,
    augment: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image = image.convert("RGB")
    if augment and random.random() < 0.5:
        image, boxes = horizontal_flip(image, boxes)
    if augment:
        image, boxes = jitter_place(image, boxes, img_size)
    else:
        image, boxes, _meta = apply_letterbox(image, boxes, img_size)
    if augment:
        image = distort_hsv(image)
    boxes, classes = clip_boxes(boxes, classes, img_size)
    normalized = xyxy_to_norm_xywh(boxes, img_size)
    return (
        pil_to_tensor(image),
        torch.from_numpy(normalized),
        torch.from_numpy(classes.astype(np.int64)),
    )


def collate_yolo(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    targets = []
    for index, (_image, boxes, classes) in enumerate(batch):
        if boxes.numel() == 0:
            continue
        batch_index = torch.full((boxes.shape[0], 1), index, dtype=torch.float32)
        targets.append(torch.cat((batch_index, classes.view(-1, 1).float(), boxes), dim=1))
    if targets:
        target_tensor = torch.cat(targets, dim=0)
    else:
        target_tensor = torch.zeros((0, 6), dtype=torch.float32)
    return images, target_tensor


class YOLODataset(Dataset):
    """Images with sibling YOLO label files: `class cx cy w h`, all normalized."""

    def __init__(
        self,
        root: str | Path,
        image_dir: str = "images",
        label_dir: str = "labels",
        img_size: int = 416,
        augment: bool = False,
    ) -> None:
        self.root = Path(root)
        self.image_paths = list_images(self.root / image_dir)
        if not self.image_paths:
            raise FileNotFoundError(f"No images under {self.root / image_dir}")
        self.label_dir = self.root / label_dir
        self.img_size = img_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.image_paths)

    def label_path_for(self, image_path: Path) -> Path:
        return self.label_dir / f"{image_path.stem}.txt"

    def __getitem__(self, index: int):
        path = self.image_paths[index]
        image = Image.open(path).convert("RGB")
        boxes, classes = read_yolo_label(self.label_path_for(path), image.size)
        return prepare_sample(image, boxes, classes, self.img_size, self.augment)


def load_dataset_yaml(path: str | Path) -> dict:
    path = Path(path)
    data = yaml.safe_load(path.read_text())
    root = path.parent
    base = data.get("path") or "."
    base_path = Path(base)
    if not base_path.is_absolute():
        base_path = root / base_path
    names = data.get("names", [])
    if isinstance(names, dict):
        names = [names[key] for key in sorted(names, key=lambda item: int(item))]
    return {
        "root": base_path,
        "train": data.get("train", "images"),
        "val": data.get("val", "images"),
        "names": list(names),
    }


class COCODataset(Dataset):
    """COCO instances JSON, mapped onto the 80 detection classes.

    Crowd boxes are skipped. Category ids that are not in the 80-class list
    are skipped. Boxes are converted to the same training tensors as YOLODataset.
    """

    def __init__(
        self,
        image_dir: str | Path,
        annotation_file: str | Path,
        img_size: int = 416,
        augment: bool = False,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.img_size = img_size
        self.augment = augment
        with Path(annotation_file).open() as handle:
            payload = json.load(handle)
        index_of = {category_id: index for index, category_id in enumerate(COCO_CATEGORY_IDS)}
        images = {image["id"]: image for image in payload["images"]}
        grouped: dict[int, list[tuple[int, list[float]]]] = {image_id: [] for image_id in images}
        for ann in payload["annotations"]:
            if ann.get("iscrowd", 0):
                continue
            class_index = index_of.get(ann["category_id"])
            if class_index is None:
                continue
            grouped.setdefault(ann["image_id"], []).append((class_index, ann["bbox"]))
        self.records = []
        for image_id, image in images.items():
            path = self.image_dir / image["file_name"]
            if path.is_file():
                self.records.append((path, image["width"], image["height"], grouped.get(image_id, [])))
        if not self.records:
            raise FileNotFoundError(f"No COCO images from {annotation_file} were found in {self.image_dir}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        path, width, height, annotations = self.records[index]
        image = Image.open(path).convert("RGB")
        boxes = []
        classes = []
        # Prefer the actual decoded size if it differs from the JSON.
        width, height = image.size
        for class_index, bbox in annotations:
            x, y, bw, bh = bbox
            if bw <= 1 or bh <= 1:
                continue
            boxes.append([x, y, x + bw, y + bh])
            classes.append(class_index)
        if boxes:
            box_array = np.asarray(boxes, dtype=np.float32)
            class_array = np.asarray(classes, dtype=np.int64)
        else:
            box_array = np.zeros((0, 4), dtype=np.float32)
            class_array = np.zeros((0,), dtype=np.int64)
        return prepare_sample(image, box_array, class_array, self.img_size, self.augment)
