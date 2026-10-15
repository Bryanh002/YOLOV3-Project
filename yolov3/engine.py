"""Shared inference path for the command line tool and the viewer."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image, ImageDraw

from yolov3.config import COCO_CLASSES, class_color
from yolov3.model import YOLOv3
from yolov3.ops import decode_predictions, letterbox_image, non_max_suppression, pil_to_tensor, scale_name, undo_letterbox
from yolov3.weights import load_model_weights


class Detector:
    def __init__(self, weights: str | Path, device: str | None = None) -> None:
        self.weights_path = Path(weights)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.class_names = list(COCO_CLASSES)
        num_classes = 80
        anchors = None
        checkpoint = None
        if self.weights_path.suffix.lower() in {".pt", ".pth"}:
            checkpoint = torch.load(self.weights_path, map_location="cpu", weights_only=False)
            names = checkpoint.get("class_names") if isinstance(checkpoint, dict) else None
            if names:
                self.class_names = list(names)
                num_classes = len(self.class_names)
            elif isinstance(checkpoint, dict) and "num_classes" in checkpoint:
                num_classes = int(checkpoint["num_classes"])
            if isinstance(checkpoint, dict) and checkpoint.get("anchors"):
                anchors = checkpoint["anchors"]
        # Every detection runs this module. A weight file only fills parameters.
        self.model = YOLOv3(num_classes=num_classes, anchors=anchors).to(self.device)
        self.origin = "trained" if anchors is not None else "darknet"
        if checkpoint is None:
            load_model_weights(self.model, self.weights_path)
        else:
            state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
            self.model.load_state_dict(state)
        self.model.eval()

    @property
    def parameter_count(self) -> int:
        return sum(param.numel() for param in self.model.parameters())

    @torch.no_grad()
    def detect(
        self,
        image: Image.Image,
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
        img_size: int = 416,
    ) -> list[dict]:
        image = image.convert("RGB")
        canvas, meta = letterbox_image(image, img_size)
        tensor = pil_to_tensor(canvas).unsqueeze(0).to(self.device)
        predictions = self.model(tensor)
        decoded = decode_predictions(predictions, self.model.yolo_layers, img_size)
        detections = non_max_suppression(decoded, conf_thresh, iou_thresh)[0].cpu()
        if detections.numel():
            detections = detections.clone()
            detections[:, :4] = undo_letterbox(detections[:, :4], meta, image.size)
        return [self._row_to_dict(row) for row in detections]

    def _row_to_dict(self, row: torch.Tensor) -> dict:
        class_id = int(row[5].item())
        name = self.class_names[class_id] if 0 <= class_id < len(self.class_names) else str(class_id)
        return {
            "x1": round(float(row[0]), 2),
            "y1": round(float(row[1]), 2),
            "x2": round(float(row[2]), 2),
            "y2": round(float(row[3]), 2),
            "confidence": round(float(row[4]), 4),
            "class_id": class_id,
            "class_name": name,
            "scale": scale_name(int(row[6].item())),
            "color": class_color(class_id),
        }


def draw_detections(image: Image.Image, detections: list[dict]) -> Image.Image:
    canvas = image.convert("RGB")
    pen = ImageDraw.Draw(canvas)
    for det in detections:
        color = det["color"]
        box = [det["x1"], det["y1"], det["x2"], det["y2"]]
        pen.rectangle(box, outline=color, width=3)
        label = f"{det['class_name']} {det['confidence']:.2f}"
        pen.text((det["x1"] + 4, det["y1"] + 4), label, fill=color)
    return canvas
