"""Run YOLOv3 on an image or a directory and write annotated copies."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from PIL import Image

from yolov3.data import IMAGE_SUFFIXES
from yolov3.engine import Detector, draw_detections


def iter_sources(source: Path):
    if source.is_dir():
        for path in sorted(source.rglob("*")):
            if path.suffix.lower() in IMAGE_SUFFIXES:
                yield path
    else:
        yield source


def detect(args) -> None:
    detector = Detector(args.weights, device=args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for path in iter_sources(Path(args.source)):
        image = Image.open(path).convert("RGB")
        started = time.perf_counter()
        detections = detector.detect(image, args.conf, args.iou, args.img_size)
        elapsed = (time.perf_counter() - started) * 1000
        annotated = draw_detections(image, detections)
        destination = output / path.name
        annotated.save(destination)
        summary = ", ".join(f"{det['class_name']} {det['confidence']:.2f}" for det in detections) or "no objects"
        print(f"{path.name}: {len(detections)} boxes in {elapsed:.0f} ms  {summary}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Detect objects with YOLOv3")
    parser.add_argument("--weights", required=True, help="Darknet .weights file or a training checkpoint")
    parser.add_argument("--source", required=True, help="Image file or directory")
    parser.add_argument("--output", default="runs/detect")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--img-size", type=int, default=416)
    parser.add_argument("--device", default=None)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    detect(parse_args(argv))


if __name__ == "__main__":
    main()
