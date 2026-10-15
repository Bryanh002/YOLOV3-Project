# YOLOv3

A re-implementation of YOLOv3. Detection does not call a pretrained detector
or a detection library. Darknet-53, the three-scale head, anchor assignment,
the loss, k-means priors, and class-wise NMS are all in this repository.
PyTorch supplies convolution and batch norm only.

The original notebooks in `legacy/` used a ResNet backbone, a single detection
head, and a loss that did not assign anchors. This package replaces that.

A Darknet `yolov3.weights` file is optional. It is the same tensor layout, so
it can be loaded to compare against the COCO-trained reference. It is not the
implementation. Training from scratch clusters whatever labels you give it.

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
`yolov3/cfg/yolov3.cfg`, including the YOLO layers at 82, 94, and 106, which is
why a Darknet weight file can be read without a second architecture. Input
pixels are divided by 255. There is no ImageNet mean or standard deviation.

Priors come from k-means on the training boxes. The distance is `1 - IoU`
with both boxes centered at the origin, which is the clustering YOLOv3
inherited from YOLOv2. `--anchors coco` keeps the nine published COCO clusters
instead, which is what a Darknet COCO weight file expects.

## Loss and training

Each ground-truth box is given to the single anchor, among all nine, with the
best width/height IoU. Predictions that are not responsible but overlap a
ground-truth box above the ignore threshold (0.7, as in the released config)
are left out of the no-object term. Coordinate targets are the inverted box
equations. Center offsets use squared error after the sigmoid, width and height
use squared error in log space, and objectness and classes use binary
cross-entropy. Boxes are weighted by `(2 - w·h)` so small objects count more.

The training loop defaults to the paper's SGD: momentum 0.9, weight decay
`5e-4` on convolution kernels only, the burn-in curve `lr * (step / burn_in)^4`,
and the step drops at batches 400000 and 450000. `--optimizer adam` is there
for a small labeled set, where the summed objectness term otherwise needs a
long SGD run. `--multi-scale` resizes every tenth batch to a multiple of 32
between 320 and 608. Augmentation is a horizontal flip, aspect jitter, and the
hue, saturation, and exposure shifts from the config.

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
python -m yolov3.train --data samples/synthetic/dataset.yaml --optimizer adam --epochs 40 --batch-size 4 --img-size 320 --burn-in 0 --lr 0.001 --grad-clip 10 --no-augment --workers 0 --save-dir runs/shapes
python -m yolov3.detect --weights runs/shapes/best.pt --source samples/synthetic/images --img-size 320
python -m yolov3.detect --weights weights/yolov3.weights --source samples/photos --img-size 416
```

The first command fits the nine priors to the shape labels, then trains the
loss from random initialization. The workbench lists that checkpoint separately
from a Darknet COCO file when both are on disk.

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
