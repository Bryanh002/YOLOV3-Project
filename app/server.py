"""Browser workbench for YOLOv3.

The page lists the sample photographs and the labeled synthetic scenes, runs
letterboxed inference, and draws the boxes back on the original image.
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from yolov3.config import MULTI_SCALE_SIZES
from yolov3.data import IMAGE_SUFFIXES, load_dataset_yaml, read_yolo_label
from yolov3.engine import Detector
from yolov3.weights import download_official_weights

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / "static"
SAMPLES = ROOT / "samples"
WEIGHTS_PATH = ROOT / "weights" / "yolov3.weights"

state: dict = {"detector": None, "error": None, "active": None, "catalog": []}
lock = threading.Lock()

NO_WEIGHTS = (
    "No checkpoint yet. Train this reimplementation with "
    "`python -m yolov3.train --data samples/synthetic/dataset.yaml --anchors kmeans --burn-in 0`, "
    "or download the Darknet COCO file to check that the same network reads those weights."
)


def weight_catalog() -> list[dict]:
    """Weight files the reimplemented network can run. None of these are a library model."""
    specs = [
        (ROOT / "runs" / "shapes" / "best.pt", "shapes", "Trained here on the shape plates", "trained"),
        (WEIGHTS_PATH, "coco", "Darknet file, COCO classes", "darknet"),
        (ROOT / "weights" / "yolov3.pt", "coco-pt", "Checkpoint, COCO classes", "checkpoint"),
    ]
    found = []
    seen = set()
    for path, weight_id, label, kind in specs:
        if not path.is_file():
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        found.append({"id": weight_id, "path": path, "label": label, "kind": kind})
    return found


def try_load(weight_id: str | None = None) -> None:
    catalog = weight_catalog()
    state["catalog"] = catalog
    if not catalog:
        state["detector"] = None
        state["active"] = None
        state["error"] = NO_WEIGHTS
        return
    chosen = None
    if weight_id:
        chosen = next((item for item in catalog if item["id"] == weight_id), None)
        if chosen is None:
            raise FileNotFoundError(weight_id)
    if chosen is None:
        env = os.environ.get("YOLOV3_WEIGHTS")
        if env:
            env_path = Path(env).resolve()
            chosen = next((item for item in catalog if item["path"].resolve() == env_path), None)
    if chosen is None:
        chosen = next((item for item in catalog if item["id"] == "coco"), catalog[0])
    state["detector"] = Detector(chosen["path"])
    state["active"] = chosen["id"]
    state["error"] = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try_load()
    yield


app = FastAPI(title="YOLOv3 bench", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")
app.mount("/samples", StaticFiles(directory=SAMPLES), name="samples")


def synthetic_names() -> list[str]:
    yaml_path = SAMPLES / "synthetic" / "dataset.yaml"
    if not yaml_path.is_file():
        return []
    return load_dataset_yaml(yaml_path)["names"]


def list_samples() -> list[dict]:
    items = []
    folders = (
        (SAMPLES / "photos", "photo"),
        (SAMPLES / "synthetic" / "images", "synthetic"),
    )
    for folder, kind in folders:
        if not folder.is_dir():
            continue
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            relative = path.relative_to(SAMPLES).as_posix()
            items.append(
                {
                    "id": relative,
                    "name": path.stem.replace("_", " "),
                    "kind": kind,
                    "url": f"/samples/{relative}",
                }
            )
    return items


def resolve_sample(sample_id: str) -> Path:
    root = SAMPLES.resolve()
    path = (SAMPLES / sample_id).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Unknown sample") from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Unknown sample")
    return path


def ground_truth(path: Path, size: tuple[int, int]) -> list[dict]:
    if path.parent.name != "images":
        return []
    label_path = path.parent.parent / "labels" / f"{path.stem}.txt"
    names = synthetic_names()
    boxes, classes = read_yolo_label(label_path, size)
    labels = []
    for box, class_id in zip(boxes, classes):
        name = names[int(class_id)] if 0 <= int(class_id) < len(names) else str(int(class_id))
        labels.append(
            {
                "x1": round(float(box[0]), 2),
                "y1": round(float(box[1]), 2),
                "x2": round(float(box[2]), 2),
                "y2": round(float(box[3]), 2),
                "class_id": int(class_id),
                "class_name": name,
            }
        )
    return labels


def open_image(raw: bytes) -> Image.Image:
    from io import BytesIO

    try:
        return Image.open(BytesIO(raw)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="That file is not an image") from exc


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


def _catalog_payload() -> list[dict]:
    return [
        {
            "id": item["id"],
            "label": item["label"],
            "kind": item["kind"],
            "active": item["id"] == state["active"],
        }
        for item in state["catalog"]
    ]


@app.get("/api/health")
def health() -> dict:
    detector = state["detector"]
    payload = {
        "ready": detector is not None,
        "error": state["error"],
        "available": _catalog_payload(),
        "classes": [],
        "img_sizes": list(MULTI_SCALE_SIZES),
    }
    if detector is None:
        return payload
    payload.update(
        {
            "weights": str(detector.weights_path),
            "origin": detector.origin,
            "device": str(detector.device),
            "parameters": detector.parameter_count,
            "classes": detector.class_names,
        }
    )
    return payload


@app.post("/api/weights/select")
def select_weights(weight_id: str = Form(...)) -> dict:
    try:
        with lock:
            try_load(weight_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown weight set {exc}") from exc
    if state["detector"] is None:
        raise HTTPException(status_code=409, detail=state["error"] or NO_WEIGHTS)
    return health()


@app.get("/api/samples")
def samples() -> dict:
    return {"samples": list_samples()}


@app.post("/api/weights/download")
def download_weights() -> dict:
    try:
        download_official_weights(WEIGHTS_PATH)
        os.environ["YOLOV3_WEIGHTS"] = str(WEIGHTS_PATH)
        try_load()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    if state["detector"] is None:
        raise HTTPException(status_code=500, detail=state["error"] or "Weights did not load")
    return health()


@app.post("/api/detect")
async def detect(
    sample: str | None = Form(default=None),
    file: UploadFile | None = File(default=None),
    conf: float = Form(0.25),
    iou: float = Form(0.45),
    imgsz: int = Form(416),
) -> dict:
    detector = state["detector"]
    if detector is None:
        raise HTTPException(status_code=409, detail=state["error"] or "Weights are not loaded")
    if imgsz not in MULTI_SCALE_SIZES:
        raise HTTPException(status_code=400, detail="Image size must be a multiple of 32 from 320 to 608")
    if not 0.0 <= conf <= 1.0 or not 0.0 <= iou <= 1.0:
        raise HTTPException(status_code=400, detail="Confidence and IoU thresholds must be between 0 and 1")

    labels: list[dict] = []
    if file is not None and file.filename:
        raw = await file.read()
        if len(raw) > 15 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Image is larger than 15 MB")
        image = open_image(raw)
        source_name = file.filename
    elif sample:
        path = resolve_sample(sample)
        image = Image.open(path).convert("RGB")
        labels = ground_truth(path, image.size)
        source_name = path.name
    else:
        raise HTTPException(status_code=400, detail="Choose a sample or upload an image")

    started = time.perf_counter()
    with lock:
        detections = detector.detect(image, conf_thresh=conf, iou_thresh=iou, img_size=imgsz)
    elapsed_ms = (time.perf_counter() - started) * 1000
    return {
        "source": source_name,
        "image": {"width": image.size[0], "height": image.size[1]},
        "detections": detections,
        "labels": labels,
        "elapsed_ms": round(elapsed_ms, 1),
        "img_size": imgsz,
    }


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Open the YOLOv3 detection workbench")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--weights", default=None)
    args = parser.parse_args()
    if args.weights:
        os.environ["YOLOV3_WEIGHTS"] = args.weights
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
