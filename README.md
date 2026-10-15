# YOLOv3

A from-scratch YOLOv3 detector: Darknet-53, three prediction scales, the paper's
box parameterization, and a browser workbench for running it on sample images.

The original notebooks in `legacy/` used a ResNet backbone, a single detection
head, and a loss that did not assign anchors. This package replaces that.

## Model

Darknet-53 is the residual backbone: a 3×3 stem, then stages with 1, 2, 8, 8,
and 4 residual blocks. Each block is a 1×1 bottleneck, a 3×3 convolution, and
a shortcut. Downsampling is a stride-2 convolution, not pooling.

The heads read strides 8, 16, and 32. The stride-32 map predicts the three
largest anchors. It is reduced, nearest-neighbor upsampled, and concatenated
with the stride-16 map, then the same thing happens for stride 8. Each head
emits three boxes with the paper's equations

```
bx = σ(tx) + cx
by = σ(ty) + cy
bw = pw · exp(tw)
bh = ph · exp(th)
```

Class scores are independent logistics, not a softmax. The layer order matches
`yolov3/cfg/yolov3.cfg`, including the YOLO layers at 82, 94, and 106, so the
official Darknet weight file loads directly. Input pixels are divided by 255.
There is no ImageNet mean or standard deviation.

## Loss and training

Each ground-truth box is given to the single anchor, among all nine, with the
best width/height IoU. Predictions that are not responsible but overlap a
ground-truth box above the ignore threshold (0.7, as in the released config)
are left out of the no-object term. Coordinate targets are the inverted box
equations. Center offsets use squared error after the sigmoid, width and height
use squared error in log space, and objectness and classes use binary
cross-entropy. Boxes are weighted by `(2 - w·h)` so small objects count more.

The training loop uses SGD with momentum 0.9, weight decay `5e-4` on
convolution kernels only, the burn-in curve `lr * (step / burn_in)^4`, and the
step drops at batches 400000 and 450000. `--multi-scale` resizes every tenth
batch to a multiple of 32 between 320 and 608. Augmentation is a horizontal
flip, aspect jitter, and the hue, saturation, and exposure shifts from the
config.

## Setup

```bash
python -m pip install -r requirements.txt
python scripts/download_weights.py
python scripts/generate_samples.py
```

`download_weights.py` checks the file against the published SHA-256. The
weights are about 237 MB and are gitignored. CPU PyTorch is enough for the
viewer; training on COCO wants a GPU.

## Viewer

```bash
python -m app.server
```

Open http://127.0.0.1:8000. The filmstrip has photographs and labeled synthetic
scenes. Confidence, NMS IoU, and the network size (320, 416, 512, or 608) are
live. Boxes are drawn in the original image after undoing the letterbox, and
each one is tagged with the head that produced it: large, medium, or small.
Arrow keys move through the plates. Images can be dropped on the stage.

If the weights are missing, the page can download them.

## Detect and train

```bash
python -m yolov3.detect --weights weights/yolov3.weights --source samples/photos --img-size 416
python -m yolov3.train --data samples/synthetic/dataset.yaml --epochs 20 --batch-size 4 --img-size 416 --burn-in 0
```

COCO training takes image directories plus `instances_*.json`:

```bash
python -m yolov3.train \
  --coco-train /data/coco/train2017 --coco-train-ann /data/coco/annotations/instances_train2017.json \
  --coco-val /data/coco/val2017 --coco-val-ann /data/coco/annotations/instances_val2017.json
```

Category ids are mapped onto the 80-class `coco.names` order used by the
official weights. Crowd boxes are skipped.

## Tests

```bash
python -m pytest
```

The suite checks output shapes, the 62,001,757-float Darknet layout, anchor
assignment, the ignore mask, letterbox geometry, class-wise NMS, and a short
overfit of the box loss. When `weights/yolov3.weights` is present it also
checks that the park photograph is detected as a dog.

## Samples

`samples/photos` is the set to open in the viewer. `samples/synthetic` is a
tiny labeled set (`disc`, `tile`, `wedge`) for the training loop and for the
ground-truth overlay. Sources are listed in `samples/ATTRIBUTION.txt`.
