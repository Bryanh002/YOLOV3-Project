"""YOLOv3 training loss.

For every ground-truth box the single best of the nine anchors, measured by
width/height IoU, is responsible. Other predictions whose decoded box overlaps
that object above the ignore threshold contribute neither object nor no-object
loss. Coordinate targets are the inverted box equations from the paper, class
scores are independent logistics, and small boxes are up-weighted by (2 - w h).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolov3.ops import box_iou, decode_scale, norm_xywh_to_xyxy, wh_iou


class YOLOv3Loss(nn.Module):
    def __init__(self, num_classes: int, anchors, masks, ignore_thresh: float = 0.7) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.anchors = tuple(tuple(anchor) for anchor in anchors)
        self.masks = tuple(tuple(mask) for mask in masks)
        self.ignore_thresh = ignore_thresh
        self.last_parts: dict[str, float] = {}

    def forward(self, predictions, targets: torch.Tensor, img_size: int) -> torch.Tensor:
        if len(predictions) != len(self.masks):
            raise ValueError(f"Expected {len(self.masks)} prediction scales, got {len(predictions)}")
        device = predictions[0].device
        if targets is None or targets.numel() == 0:
            targets = predictions[0].new_zeros((0, 6))
        else:
            targets = targets.to(device).float()

        anchor_tensor = predictions[0].new_tensor(self.anchors)
        if targets.shape[0]:
            gt_wh = targets[:, 4:6] * float(img_size)
            anchor_ious = wh_iou(gt_wh, anchor_tensor)
            best_ious, best_anchor = anchor_ious.max(dim=1)
        else:
            best_ious = predictions[0].new_zeros((0,))
            best_anchor = torch.zeros((0,), dtype=torch.long, device=device)

        batch = predictions[0].shape[0]
        totals = {name: predictions[0].sum() * 0.0 for name in ("xy", "wh", "obj", "cls")}
        positive_counts = []

        for prediction, mask in zip(predictions, self.masks):
            parts, count = self._loss_one_scale(
                prediction,
                targets,
                best_anchor,
                best_ious,
                mask,
                img_size,
            )
            positive_counts.append(count)
            for name in totals:
                totals[name] = totals[name] + parts[name]
        self.last_positive_counts = positive_counts

        total = (totals["xy"] + totals["wh"] + totals["obj"] + totals["cls"]) / batch
        self.last_parts = {name: float(value.detach() / batch) for name, value in totals.items()}
        self.last_parts["loss"] = float(total.detach())
        return total

    def _loss_one_scale(self, prediction, targets, best_anchor, best_ious, mask, img_size: int):
        batch, _, grid_y, grid_x = prediction.shape
        num_anchors = len(mask)
        num_classes = self.num_classes
        device = prediction.device

        raw = (
            prediction.view(batch, num_anchors, 5 + num_classes, grid_y, grid_x)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )
        obj_mask = torch.zeros(batch, num_anchors, grid_y, grid_x, dtype=torch.bool, device=device)
        noobj_mask = torch.ones(batch, num_anchors, grid_y, grid_x, dtype=torch.bool, device=device)
        target_xy = torch.zeros(batch, num_anchors, grid_y, grid_x, 2, device=device)
        target_wh = torch.zeros(batch, num_anchors, grid_y, grid_x, 2, device=device)
        target_cls = torch.zeros(batch, num_anchors, grid_y, grid_x, num_classes, device=device)
        box_weight = torch.ones(batch, num_anchors, grid_y, grid_x, device=device)
        slot_quality = torch.zeros(batch, num_anchors, grid_y, grid_x, device=device)

        with torch.no_grad():
            decoded = decode_scale(prediction.detach(), self.anchors, mask, img_size)
            pred_xyxy = decoded[..., 0:4]
            pred_xyxy = torch.cat(
                (
                    pred_xyxy[..., 0:1] - pred_xyxy[..., 2:3] / 2,
                    pred_xyxy[..., 1:2] - pred_xyxy[..., 3:4] / 2,
                    pred_xyxy[..., 0:1] + pred_xyxy[..., 2:3] / 2,
                    pred_xyxy[..., 1:2] + pred_xyxy[..., 3:4] / 2,
                ),
                dim=-1,
            )
        self._apply_ignore(noobj_mask, pred_xyxy, targets, img_size)

        local_index = {int(anchor_index): offset for offset, anchor_index in enumerate(mask)}
        for row in range(targets.shape[0]):
            anchor_index = int(best_anchor[row])
            if anchor_index not in local_index:
                continue
            image_index = int(targets[row, 0])
            class_index = int(targets[row, 1])
            if image_index < 0 or image_index >= batch or class_index < 0 or class_index >= num_classes:
                continue
            gx = float(targets[row, 2]) * grid_x
            gy = float(targets[row, 3]) * grid_y
            if gx < 0 or gy < 0 or gx >= grid_x or gy >= grid_y:
                continue
            cell_x = min(int(gx), grid_x - 1)
            cell_y = min(int(gy), grid_y - 1)
            gt_w = float(targets[row, 4]) * img_size
            gt_h = float(targets[row, 5]) * img_size
            if gt_w <= 1 or gt_h <= 1:
                continue
            local = local_index[anchor_index]
            quality = float(best_ious[row])
            if obj_mask[image_index, local, cell_y, cell_x] and quality <= float(
                slot_quality[image_index, local, cell_y, cell_x]
            ):
                continue
            anchor_w, anchor_h = self.anchors[anchor_index]
            obj_mask[image_index, local, cell_y, cell_x] = True
            noobj_mask[image_index, local, cell_y, cell_x] = False
            slot_quality[image_index, local, cell_y, cell_x] = quality
            target_xy[image_index, local, cell_y, cell_x, 0] = gx - cell_x
            target_xy[image_index, local, cell_y, cell_x, 1] = gy - cell_y
            target_wh[image_index, local, cell_y, cell_x, 0] = torch.log(
                prediction.new_tensor(gt_w / anchor_w)
            )
            target_wh[image_index, local, cell_y, cell_x, 1] = torch.log(
                prediction.new_tensor(gt_h / anchor_h)
            )
            target_cls[image_index, local, cell_y, cell_x, class_index] = 1.0
            area = float(targets[row, 4] * targets[row, 5])
            box_weight[image_index, local, cell_y, cell_x] = max(0.0, 2.0 - area)

        raw_xy = raw[..., 0:2]
        raw_wh = raw[..., 2:4]
        raw_obj = raw[..., 4]
        raw_cls = raw[..., 5:]
        positives = obj_mask
        negatives = noobj_mask

        obj_bce = F.binary_cross_entropy_with_logits(raw_obj, torch.zeros_like(raw_obj), reduction="none")
        loss_obj = (obj_bce * negatives).sum()
        loss_obj = loss_obj + F.binary_cross_entropy_with_logits(
            raw_obj, torch.ones_like(raw_obj), reduction="none"
        )[positives].sum()

        if torch.any(positives):
            loss_xy = ((raw_xy.sigmoid() - target_xy).pow(2).sum(dim=-1) * box_weight)[positives].sum()
            loss_wh = ((raw_wh - target_wh).pow(2).sum(dim=-1) * box_weight)[positives].sum()
            loss_cls = F.binary_cross_entropy_with_logits(raw_cls, target_cls, reduction="none").sum(dim=-1)[
                positives
            ].sum()
        else:
            # Keep the prediction in the graph when an image has no boxes on this scale.
            zero = prediction.sum() * 0.0
            loss_xy = zero
            loss_wh = zero
            loss_cls = zero

        return {"xy": loss_xy, "wh": loss_wh, "obj": loss_obj, "cls": loss_cls}, int(obj_mask.sum().item())

    def _apply_ignore(self, noobj_mask, pred_xyxy, targets, img_size: int) -> None:
        batch = pred_xyxy.shape[0]
        for image_index in range(batch):
            ground_truth = targets[targets[:, 0] == image_index]
            if ground_truth.numel() == 0:
                continue
            gt_xyxy = norm_xywh_to_xyxy(ground_truth[:, 2:6], img_size)
            flat = pred_xyxy[image_index].reshape(-1, 4)
            best = box_iou(flat, gt_xyxy).max(dim=1).values.view_as(noobj_mask[image_index])
            noobj_mask[image_index][best > self.ignore_thresh] = False
