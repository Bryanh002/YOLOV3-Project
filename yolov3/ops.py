"""Box geometry, YOLOv3 decoding, letterbox preprocessing, and class-wise NMS."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from yolov3.config import SCALE_NAMES


@dataclass(frozen=True)
class Letterbox:
    scale_x: float
    scale_y: float
    pad_x: int
    pad_y: int
    size: int


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    out = boxes.clone()
    out[..., 0] = boxes[..., 0] - boxes[..., 2] / 2
    out[..., 1] = boxes[..., 1] - boxes[..., 3] / 2
    out[..., 2] = boxes[..., 0] + boxes[..., 2] / 2
    out[..., 3] = boxes[..., 1] + boxes[..., 3] / 2
    return out


def xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    out = boxes.clone()
    out[..., 0] = (boxes[..., 0] + boxes[..., 2]) / 2
    out[..., 1] = (boxes[..., 1] + boxes[..., 3]) / 2
    out[..., 2] = boxes[..., 2] - boxes[..., 0]
    out[..., 3] = boxes[..., 3] - boxes[..., 1]
    return out


def box_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU of xyxy boxes. Returns a tensor shaped (N, M)."""
    if boxes_a.numel() == 0 or boxes_b.numel() == 0:
        return boxes_a.new_zeros((boxes_a.shape[0], boxes_b.shape[0]))
    top_left = torch.max(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = torch.min(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    wh = (bottom_right - top_left).clamp(min=0)
    intersection = wh[..., 0] * wh[..., 1]
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]).clamp(min=0) * (boxes_a[:, 3] - boxes_a[:, 1]).clamp(min=0)
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]).clamp(min=0) * (boxes_b[:, 3] - boxes_b[:, 1]).clamp(min=0)
    union = area_a[:, None] + area_b[None, :] - intersection
    return intersection / union.clamp(min=1e-9)


def wh_iou(wh_a: torch.Tensor, wh_b: torch.Tensor) -> torch.Tensor:
    """IoU of width/height pairs, treating every box as centered at the origin.

    YOLOv3 chooses the responsible anchor with this comparison, before the
    box has a cell offset.
    """
    if wh_a.numel() == 0 or wh_b.numel() == 0:
        return wh_a.new_zeros((wh_a.shape[0], wh_b.shape[0]))
    inter_w = torch.min(wh_a[:, None, 0], wh_b[None, :, 0])
    inter_h = torch.min(wh_a[:, None, 1], wh_b[None, :, 1])
    intersection = inter_w.clamp(min=0) * inter_h.clamp(min=0)
    area_a = wh_a[:, 0] * wh_a[:, 1]
    area_b = wh_b[:, 0] * wh_b[:, 1]
    union = area_a[:, None] + area_b[None, :] - intersection
    return intersection / union.clamp(min=1e-9)


def norm_xywh_to_xyxy(boxes: torch.Tensor, img_size: int) -> torch.Tensor:
    """Normalized center xywh to pixel xyxy on a square canvas."""
    xywh = boxes.clone()
    xywh[..., 0:4] = boxes[..., 0:4] * float(img_size)
    return xywh_to_xyxy(xywh)


def decode_scale(prediction: torch.Tensor, anchors, mask, img_size: int) -> torch.Tensor:
    """Decode one raw head into pixel xywh, objectness, and class probabilities.

    The returned tensor has shape (batch, anchors, grid_y, grid_x, 5 + classes).
    Box equations are the ones in the paper:

        bx = sigmoid(tx) + cx
        by = sigmoid(ty) + cy
        bw = pw * exp(tw)
        bh = ph * exp(th)
    """
    batch, channels, grid_y, grid_x = prediction.shape
    num_anchors = len(mask)
    if channels % num_anchors != 0:
        raise ValueError(f"Prediction channels {channels} are not divisible by {num_anchors} anchors")
    num_attrs = channels // num_anchors
    pred = prediction.view(batch, num_anchors, num_attrs, grid_y, grid_x).permute(0, 1, 3, 4, 2).contiguous()

    stride = float(img_size) / float(grid_y)
    if img_size % grid_y != 0 or grid_y != grid_x:
        raise ValueError(f"Cannot decode a {grid_y}x{grid_x} map at image size {img_size}")

    grid_y_idx, grid_x_idx = torch.meshgrid(
        torch.arange(grid_y, device=pred.device, dtype=pred.dtype),
        torch.arange(grid_x, device=pred.device, dtype=pred.dtype),
        indexing="ij",
    )
    grid = torch.stack((grid_x_idx, grid_y_idx), dim=-1).view(1, 1, grid_y, grid_x, 2)
    anchor_wh = pred.new_tensor([anchors[int(index)] for index in mask]).view(1, num_anchors, 1, 1, 2)

    xy = (pred[..., 0:2].sigmoid() + grid) * stride
    # Clamp before exp so a wild logit cannot overflow the ignore-mask IoU.
    wh = pred[..., 2:4].clamp(-12.0, 8.0).exp() * anchor_wh
    obj = pred[..., 4:5].sigmoid()
    classes = pred[..., 5:].sigmoid()
    return torch.cat((xy, wh, obj, classes), dim=-1)


def decode_predictions(predictions, specs, img_size: int) -> torch.Tensor:
    """Concatenate every scale into (batch, boxes, 6 + classes).

    The last channel is the scale index (0 large, 1 medium, 2 small) so the
    viewer can say which head fired. It is not a class score.
    """
    decoded = []
    for prediction, spec in zip(predictions, specs):
        boxes = decode_scale(prediction, spec.anchors, spec.mask, img_size)
        batch = boxes.shape[0]
        flat = boxes.reshape(batch, -1, boxes.shape[-1])
        scale = flat.new_full((batch, flat.shape[1], 1), float(spec.scale_index))
        decoded.append(torch.cat((flat, scale), dim=-1))
    return torch.cat(decoded, dim=1)


def greedy_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thresh: float) -> list[int]:
    if boxes.numel() == 0:
        return []
    order = scores.argsort(descending=True)
    keep: list[int] = []
    while order.numel():
        current = int(order[0])
        keep.append(current)
        if order.numel() == 1:
            break
        rest = order[1:]
        overlap = box_iou(boxes[current].unsqueeze(0), boxes[rest]).squeeze(0)
        order = rest[overlap <= iou_thresh]
    return keep


def non_max_suppression(
    prediction: torch.Tensor,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45,
    max_det: int = 300,
) -> list[torch.Tensor]:
    """Per-class greedy NMS, which is what Darknet does at inference.

    Each returned tensor is (detections, 7): x1, y1, x2, y2, score, class, scale.
    Scores are objectness times the chosen class probability.
    """
    if not 0.0 <= conf_thresh <= 1.0:
        raise ValueError("conf_thresh must be between 0 and 1")
    if not 0.0 <= iou_thresh <= 1.0:
        raise ValueError("iou_thresh must be between 0 and 1")

    num_classes = prediction.shape[-1] - 6
    results = []
    for image in prediction:
        # Independent logistic classifiers: every class above the threshold is
        # a candidate, not only the argmax. NMS then runs separately per class.
        objectness = image[:, 4:5]
        class_scores = image[:, 5 : 5 + num_classes]
        scores = objectness * class_scores
        box_index, class_index = torch.where(scores > conf_thresh)
        if box_index.numel() == 0:
            results.append(image.new_zeros((0, 7)))
            continue

        boxes = xywh_to_xyxy(image[box_index, :4])
        confidence = scores[box_index, class_index]
        class_id = class_index
        scale = image[box_index, -1]
        # Drop boxes that collapsed to nothing after decoding.
        positive = (boxes[:, 2] - boxes[:, 0] > 1) & (boxes[:, 3] - boxes[:, 1] > 1)
        boxes, confidence, class_id, scale = (
            boxes[positive],
            confidence[positive],
            class_id[positive],
            scale[positive],
        )
        if boxes.numel() == 0:
            results.append(image.new_zeros((0, 7)))
            continue

        kept = []
        for class_index in class_id.unique():
            members = class_id == class_index
            chosen = greedy_nms(boxes[members], confidence[members], iou_thresh)
            if not chosen:
                continue
            chosen_t = torch.tensor(chosen, device=boxes.device, dtype=torch.long)
            member_boxes = boxes[members][chosen_t]
            member_scores = confidence[members][chosen_t]
            member_classes = class_id[members][chosen_t].float()
            member_scales = scale[members][chosen_t]
            kept.append(
                torch.stack(
                    (
                        member_boxes[:, 0],
                        member_boxes[:, 1],
                        member_boxes[:, 2],
                        member_boxes[:, 3],
                        member_scores,
                        member_classes,
                        member_scales,
                    ),
                    dim=1,
                )
            )
        if not kept:
            results.append(image.new_zeros((0, 7)))
            continue
        detections = torch.cat(kept, dim=0)
        order = detections[:, 4].argsort(descending=True)
        results.append(detections[order][:max_det])
    return results


def letterbox_image(
    image: Image.Image,
    size: int,
    color: tuple[int, int, int] = (128, 128, 128),
) -> tuple[Image.Image, Letterbox]:
    """Fit `image` inside a square canvas without stretching it.

    Darknet fills the border with mid-gray. The returned scale and pad map
    network pixels back to the original image.
    """
    if size % 32 != 0:
        raise ValueError("Letterbox size must be a multiple of 32")
    image = image.convert("RGB")
    width, height = image.size
    ratio = min(size / width, size / height)
    resized_w = max(1, int(round(width * ratio)))
    resized_h = max(1, int(round(height * ratio)))
    resized = image.resize((resized_w, resized_h), Image.Resampling.BILINEAR)
    pad_x = (size - resized_w) // 2
    pad_y = (size - resized_h) // 2
    canvas = Image.new("RGB", (size, size), color)
    canvas.paste(resized, (pad_x, pad_y))
    meta = Letterbox(
        scale_x=resized_w / width,
        scale_y=resized_h / height,
        pad_x=pad_x,
        pad_y=pad_y,
        size=size,
    )
    return canvas, meta


def undo_letterbox(boxes: torch.Tensor, meta: Letterbox, original_size: tuple[int, int]) -> torch.Tensor:
    """Map xyxy boxes from the network canvas back to the original image."""
    if boxes.numel() == 0:
        return boxes
    width, height = original_size
    out = boxes.clone()
    out[:, 0] = (boxes[:, 0] - meta.pad_x) / meta.scale_x
    out[:, 2] = (boxes[:, 2] - meta.pad_x) / meta.scale_x
    out[:, 1] = (boxes[:, 1] - meta.pad_y) / meta.scale_y
    out[:, 3] = (boxes[:, 3] - meta.pad_y) / meta.scale_y
    out[:, [0, 2]] = out[:, [0, 2]].clamp(0, width)
    out[:, [1, 3]] = out[:, [1, 3]].clamp(0, height)
    return out


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """RGB image to a float tensor in 0..1. YOLOv3 does not use ImageNet normalization."""
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def scale_name(scale_index: int) -> str:
    if 0 <= int(scale_index) < len(SCALE_NAMES):
        return SCALE_NAMES[int(scale_index)]
    return "unknown"
