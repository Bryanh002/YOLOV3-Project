"""Read and write Darknet `.weights` files.

The header is three int32 version numbers followed by an int64 image count
when `major * 10 + minor >= 2`. Each convolutional layer is then stored as
batch-norm bias, weight, running mean, and running variance, then the
convolution kernel. Prediction layers have no batch norm, so they store a
bias followed by the kernel.
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import numpy as np
import torch

from yolov3.config import DARKNET_WEIGHTS_SHA256, DARKNET_WEIGHTS_URL, DARKNET_WEIGHT_FLOATS


def darknet_weight_count(model) -> int:
    """Number of float32 values a Darknet export of `model` contains."""
    count = 0
    for layer in model.iter_convs():
        if layer.batch_normalize:
            count += 4 * layer.bn.bias.numel()
        else:
            count += layer.conv.bias.numel()
        count += layer.conv.weight.numel()
    return count


def _read_header(handle) -> tuple[tuple[int, int, int], int]:
    header = np.fromfile(handle, dtype=np.int32, count=3)
    if header.size != 3:
        raise RuntimeError("Weight file is missing its Darknet header")
    major, minor, revision = (int(value) for value in header)
    version = (major, minor, revision)
    if version[0] * 10 + version[1] >= 2 and version[0] < 1000 and version[1] < 1000:
        seen = np.fromfile(handle, dtype=np.int64, count=1)
    else:
        seen = np.fromfile(handle, dtype=np.int32, count=1)
    seen_count = int(seen[0]) if seen.size else 0
    return version, seen_count


def _take(weights: np.ndarray, pointer: int, count: int) -> tuple[torch.Tensor, int]:
    end = pointer + count
    if end > weights.size:
        raise RuntimeError(
            f"Weight file ended at float {pointer} while a layer still needed {count} values"
        )
    tensor = torch.from_numpy(weights[pointer:end].copy())
    return tensor, end


def load_darknet_weights(model, path: str | Path) -> dict:
    path = Path(path)
    with path.open("rb") as handle:
        version, seen = _read_header(handle)
        weights = np.fromfile(handle, dtype=np.float32)

    pointer = 0
    convs = list(model.iter_convs())
    for index, layer in enumerate(convs):
        try:
            if layer.batch_normalize:
                count = layer.bn.bias.numel()
                values, pointer = _take(weights, pointer, count)
                layer.bn.bias.data.copy_(values.view_as(layer.bn.bias))
                values, pointer = _take(weights, pointer, count)
                layer.bn.weight.data.copy_(values.view_as(layer.bn.weight))
                values, pointer = _take(weights, pointer, count)
                layer.bn.running_mean.copy_(values.view_as(layer.bn.running_mean))
                values, pointer = _take(weights, pointer, count)
                layer.bn.running_var.copy_(values.view_as(layer.bn.running_var))
            else:
                count = layer.conv.bias.numel()
                values, pointer = _take(weights, pointer, count)
                layer.conv.bias.data.copy_(values.view_as(layer.conv.bias))
            count = layer.conv.weight.numel()
            values, pointer = _take(weights, pointer, count)
            layer.conv.weight.data.copy_(values.view_as(layer.conv.weight))
        except RuntimeError as exc:
            raise RuntimeError(f"Failed while reading convolutional layer {index}") from exc

    if pointer != weights.size:
        raise RuntimeError(
            f"Weight file has {weights.size} floats but this model consumed {pointer}. "
            "The architecture does not match the file."
        )
    return {"version": version, "seen": seen, "floats": int(weights.size)}


def save_darknet_weights(model, path: str | Path, seen: int = 0) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blobs: list[np.ndarray] = []
    for layer in model.iter_convs():
        if layer.batch_normalize:
            blobs.append(layer.bn.bias.detach().cpu().numpy().reshape(-1))
            blobs.append(layer.bn.weight.detach().cpu().numpy().reshape(-1))
            blobs.append(layer.bn.running_mean.detach().cpu().numpy().reshape(-1))
            blobs.append(layer.bn.running_var.detach().cpu().numpy().reshape(-1))
        else:
            blobs.append(layer.conv.bias.detach().cpu().numpy().reshape(-1))
        blobs.append(layer.conv.weight.detach().cpu().numpy().reshape(-1))
    with path.open("wb") as handle:
        np.array([0, 2, 0], dtype=np.int32).tofile(handle)
        np.array([seen], dtype=np.int64).tofile(handle)
        np.concatenate(blobs).astype(np.float32).tofile(handle)


def load_checkpoint(model, path: str | Path) -> dict:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state)
    return checkpoint if isinstance(checkpoint, dict) else {"model": state}


def load_model_weights(model, path: str | Path) -> dict:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".weights":
        if model.num_classes != 80:
            raise ValueError("Official Darknet YOLOv3 weights are trained for 80 classes")
        return load_darknet_weights(model, path)
    if suffix in {".pt", ".pth"}:
        return load_checkpoint(model, path)
    raise ValueError(f"Unrecognized weight file: {path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_official_weights(destination: str | Path, url: str = DARKNET_WEIGHTS_URL) -> Path:
    """Download yolov3.weights and check it against the published checksum."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and _sha256(destination) == DARKNET_WEIGHTS_SHA256:
        return destination
    partial = destination.with_suffix(destination.suffix + ".partial")
    urllib.request.urlretrieve(url, partial)
    digest = _sha256(partial)
    if digest != DARKNET_WEIGHTS_SHA256:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"Downloaded weights checksum {digest} does not match {DARKNET_WEIGHTS_SHA256}"
        )
    partial.replace(destination)
    return destination


def assert_official_architecture(model) -> None:
    count = darknet_weight_count(model)
    if count != DARKNET_WEIGHT_FLOATS:
        raise AssertionError(
            f"Model stores {count} Darknet floats, official YOLOv3 stores {DARKNET_WEIGHT_FLOATS}"
        )
