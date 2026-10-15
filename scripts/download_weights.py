"""Download the official YOLOv3 Darknet weights into weights/yolov3.weights."""

from __future__ import annotations

import argparse
from pathlib import Path

from yolov3.weights import download_official_weights

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "weights" / "yolov3.weights"))
    args = parser.parse_args()
    path = download_official_weights(args.output)
    print(f"Ready: {path}")


if __name__ == "__main__":
    main()
