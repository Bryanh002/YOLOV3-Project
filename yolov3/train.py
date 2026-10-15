"""Train YOLOv3 with the Darknet schedule.

The optimizer is SGD with momentum 0.9. Weight decay is applied to convolution
kernels only. The learning rate follows the burn-in curve `lr * (step / burn_in)^4`
and then the step drops at batches 400000 and 450000. Pass `--multi-scale` to
resize each tenth batch to a random multiple of 32 between 320 and 608.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from yolov3.anchors import anchors_from_labels, mean_best_iou, normalized_wh_from_labels
from yolov3.config import (
    ANCHOR_MASKS,
    ANCHORS,
    BURN_IN,
    IGNORE_THRESH,
    LEARNING_RATE,
    LR_SCALES,
    LR_STEPS,
    MOMENTUM,
    MULTI_SCALE_SIZES,
    WEIGHT_DECAY,
)
from yolov3.data import COCODataset, YOLODataset, collate_yolo, load_dataset_yaml
from yolov3.loss import YOLOv3Loss
from yolov3.model import YOLOv3
from yolov3.ops import box_iou, decode_predictions, non_max_suppression, xywh_to_xyxy


def learning_rate_at(step: int, base_lr: float, burn_in: int, steps=LR_STEPS, scales=LR_SCALES) -> float:
    """Darknet burn-in, then multiplicative drops. `step` is the 1-based batch index."""
    if burn_in and step <= burn_in:
        return base_lr * (step / burn_in) ** 4
    lr = base_lr
    for boundary, scale in zip(steps, scales):
        if step >= boundary:
            lr *= scale
    return lr


def sgd_param_groups(model: torch.nn.Module, weight_decay: float):
    decay, plain = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith(".bias"):
            plain.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": plain, "weight_decay": 0.0},
    ]


def make_loader(dataset, batch_size: int, shuffle: bool, workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_yolo,
        drop_last=False,
    )


def build_datasets(args):
    names = None
    if args.data:
        spec = load_dataset_yaml(args.data)
        names = spec["names"]
        train_set = YOLODataset(spec["root"], spec["train"], "labels", args.img_size, augment=not args.no_augment)
        val_set = YOLODataset(spec["root"], spec["val"], "labels", args.img_size, augment=False)
        return train_set, val_set, names
    if args.coco_train:
        train_set = COCODataset(args.coco_train, args.coco_train_ann, args.img_size, augment=True)
        val_set = COCODataset(args.coco_val, args.coco_val_ann, args.img_size, augment=False)
        from yolov3.config import COCO_CLASSES

        return train_set, val_set, list(COCO_CLASSES)
    raise SystemExit("Pass --data dataset.yaml or the COCO image and annotation paths")


def run_epoch(model, loader, criterion, optimizer, device, img_size, multi_scale, burn_in, base_lr, global_step, grad_clip):
    model.train()
    total = 0.0
    seen = 0
    current_size = img_size
    for batch_index, (images, targets) in enumerate(loader):
        global_step += 1
        if multi_scale and batch_index % 10 == 0:
            current_size = random.choice(MULTI_SCALE_SIZES)
        if images.shape[-1] != current_size:
            images = F.interpolate(images, size=current_size, mode="bilinear", align_corners=False)
        images = images.to(device)
        targets = targets.to(device)
        lr = learning_rate_at(global_step, base_lr, burn_in)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        predictions = model(images)
        loss = criterion(predictions, targets, current_size)
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += float(loss.detach()) * images.shape[0]
        seen += images.shape[0]
        if batch_index % 20 == 0:
            parts = criterion.last_parts
            print(
                f"  batch {batch_index:04d}  loss {float(loss.detach()):.4f}  "
                f"xy {parts['xy']:.3f} wh {parts['wh']:.3f} obj {parts['obj']:.3f} cls {parts['cls']:.3f}  "
                f"lr {lr:.6f}  size {current_size}",
                flush=True,
            )
    return total / max(seen, 1), global_step


@torch.no_grad()
def evaluate(model, loader, criterion, device, img_size, conf_thresh=0.25, iou_thresh=0.5):
    model.eval()
    total = 0.0
    seen = 0
    true_positives = 0
    ground_truths = 0
    predictions_kept = 0
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        predictions = model(images)
        loss = criterion(predictions, targets, img_size)
        total += float(loss.detach()) * images.shape[0]
        seen += images.shape[0]
        decoded = decode_predictions(predictions, model.yolo_layers, img_size)
        detections = non_max_suppression(decoded, conf_thresh, iou_thresh)
        matched, gt_count, pred_count = match_recall(detections, targets, img_size, iou_thresh)
        true_positives += matched
        ground_truths += gt_count
        predictions_kept += pred_count
    recall = true_positives / ground_truths if ground_truths else 0.0
    precision = true_positives / predictions_kept if predictions_kept else 0.0
    return {
        "loss": total / max(seen, 1),
        "recall": recall,
        "precision": precision,
        "ground_truths": ground_truths,
    }


def match_recall(detections, targets, img_size: int, iou_thresh: float) -> tuple[int, int, int]:
    """Greedy one-to-one matching of predictions to ground truth at one IoU."""
    matched = 0
    gt_count = 0
    pred_count = 0
    batch = len(detections)
    for image_index in range(batch):
        gt = targets[targets[:, 0] == image_index]
        gt_count += int(gt.shape[0])
        det = detections[image_index]
        pred_count += int(det.shape[0])
        if gt.numel() == 0 or det.numel() == 0:
            continue
        gt_xyxy = xywh_to_xyxy(gt[:, 2:6] * img_size)
        used = set()
        order = det[:, 4].argsort(descending=True)
        for row in det[order]:
            class_id = int(row[5].item())
            candidates = [
                index
                for index in range(gt.shape[0])
                if index not in used and int(gt[index, 1].item()) == class_id
            ]
            if not candidates:
                continue
            gt_boxes = gt_xyxy[candidates]
            overlaps = box_iou(row[:4].unsqueeze(0), gt_boxes).squeeze(0)
            best = int(overlaps.argmax())
            if float(overlaps[best]) >= iou_thresh:
                used.add(candidates[best])
                matched += 1
    return matched, gt_count, pred_count


def save_checkpoint(path: Path, model, optimizer, epoch: int, class_names, img_size: int, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "class_names": list(class_names) if class_names is not None else None,
            "num_classes": model.num_classes,
            "anchors": [list(pair) for pair in model.anchors],
            "img_size": img_size,
            "metrics": metrics,
        },
        path,
    )


def resolve_anchors(args, img_size: int):
    """Pick priors. A Darknet COCO file implies the published clusters.

    Training from scratch on a YOLO label directory clusters those boxes with
    the IoU k-means unless `--anchors coco` is set.
    """
    weights = str(args.weights or "")
    if weights.endswith(".weights"):
        if args.anchors == "kmeans":
            raise SystemExit("A Darknet COCO weight file uses the published anchors, not a new clustering")
        return ANCHORS
    if args.anchors == "coco" or (args.coco_train and args.anchors != "kmeans"):
        return ANCHORS
    if not args.data:
        if args.anchors == "kmeans":
            raise SystemExit("k-means anchors need --data pointing at a YOLO label set")
        return ANCHORS
    spec = load_dataset_yaml(args.data)
    label_dir = spec["root"] / "labels"
    anchors = anchors_from_labels(label_dir, img_size, k=9)
    wh = normalized_wh_from_labels(label_dir) * float(img_size)
    print(
        f"k-means anchors at {img_size}px, mean best IoU {mean_best_iou(wh, anchors):.3f}: {anchors}",
        flush=True,
    )
    return anchors


def train(args) -> None:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_set, val_set, class_names = build_datasets(args)
    num_classes = len(class_names) if class_names else 80
    anchors = resolve_anchors(args, args.img_size)
    model = YOLOv3(num_classes=num_classes, anchors=anchors).to(device)
    if args.weights:
        from yolov3.weights import load_model_weights

        load_model_weights(model, args.weights)
    criterion = YOLOv3Loss(num_classes, model.anchors, ANCHOR_MASKS, ignore_thresh=args.ignore_thresh)
    groups = sgd_param_groups(model, args.weight_decay if args.optimizer == "sgd" else 0.0)
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(groups, lr=args.lr)
    else:
        optimizer = torch.optim.SGD(groups, lr=args.lr, momentum=args.momentum)
    train_loader = make_loader(train_set, args.batch_size, True, args.workers)
    val_loader = make_loader(val_set, args.batch_size, False, args.workers)
    save_dir = Path(args.save_dir)
    best_loss = float("inf")
    best_recall = -1.0
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}", flush=True)
        train_loss, global_step = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            args.img_size,
            args.multi_scale,
            args.burn_in,
            args.lr,
            global_step,
            args.grad_clip,
        )
        metrics = evaluate(model, val_loader, criterion, device, args.img_size)
        metrics["train_loss"] = train_loss
        print(
            f"  train {train_loss:.4f}  val {metrics['loss']:.4f}  "
            f"precision {metrics['precision']:.3f}  recall {metrics['recall']:.3f}",
            flush=True,
        )
        save_checkpoint(save_dir / "last.pt", model, optimizer, epoch, class_names, args.img_size, metrics)
        recall = metrics["recall"]
        loss_value = metrics["loss"]
        improved = recall > best_recall or (recall == best_recall and loss_value < best_loss)
        if improved and loss_value == loss_value:
            best_recall = recall
            best_loss = loss_value
            save_checkpoint(save_dir / "best.pt", model, optimizer, epoch, class_names, args.img_size, metrics)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train a YOLOv3 detector")
    parser.add_argument("--data", help="YOLO dataset.yaml with train/val image folders and class names")
    parser.add_argument("--coco-train", help="COCO train image directory")
    parser.add_argument("--coco-train-ann", help="COCO instances_train JSON")
    parser.add_argument("--coco-val", help="COCO val image directory")
    parser.add_argument("--coco-val-ann", help="COCO instances_val JSON")
    parser.add_argument("--weights", help="Optional Darknet .weights or checkpoint to fine-tune")
    parser.add_argument(
        "--anchors",
        choices=("kmeans", "coco"),
        default=None,
        help="kmeans clusters the training labels; coco uses the nine published priors",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=416)
    parser.add_argument("--optimizer", choices=("sgd", "adam"), default="sgd")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--momentum", type=float, default=MOMENTUM)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--burn-in", type=int, default=BURN_IN)
    parser.add_argument("--ignore-thresh", type=float, default=IGNORE_THRESH)
    parser.add_argument("--multi-scale", action="store_true")
    parser.add_argument("--no-augment", action="store_true", help="Skip flip, jitter, and color shifts")
    parser.add_argument("--grad-clip", type=float, default=0.0, help="Clip the global gradient norm. 0 disables it")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-dir", default="runs/train")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    train(parse_args(argv))


if __name__ == "__main__":
    main()
