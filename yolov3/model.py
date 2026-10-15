"""Darknet-53 YOLOv3.

The layer order matches the released detection config so official Darknet
weights load without a conversion table. Feature maps at strides 8, 16, and
32 are fused by nearest-neighbor upsampling and concatenation, then each
scale predicts three anchor boxes.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from yolov3.config import ANCHOR_MASKS, ANCHORS, SCALE_NAMES


class ConvLayer(nn.Module):
    """Convolution used everywhere in YOLOv3.

    Hidden layers are conv + batch norm + leaky ReLU (slope 0.1), with no
    bias on the convolution. The three prediction layers are linear and keep
    a bias, which is what the Darknet weight file stores.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        batch_normalize: bool = True,
        activation: str = "leaky",
    ) -> None:
        super().__init__()
        if activation not in {"leaky", "linear"}:
            raise ValueError(f"Unsupported activation {activation!r}")
        padding = (kernel_size - 1) // 2
        self.batch_normalize = batch_normalize
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not batch_normalize,
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1) if batch_normalize else None
        self.act = nn.LeakyReLU(0.1, inplace=True) if activation == "leaky" else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class ResidualBlock(nn.Module):
    """1x1 bottleneck followed by a 3x3 projection, added back to the input."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = channels // 2
        self.conv1 = ConvLayer(channels, hidden, kernel_size=1)
        self.conv2 = ConvLayer(hidden, channels, kernel_size=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.conv1(x))


class Darknet53(nn.Module):
    """Feature extractor. Returns stride-8, stride-16, and stride-32 maps."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = ConvLayer(3, 32, kernel_size=3)
        stages = nn.ModuleList()
        in_channels = 32
        # Residual repeats at each resolution: 1, 2, 8, 8, 4.
        for out_channels, repeats in ((64, 1), (128, 2), (256, 8), (512, 8), (1024, 4)):
            stages.append(
                nn.ModuleDict(
                    {
                        "down": ConvLayer(in_channels, out_channels, kernel_size=3, stride=2),
                        "blocks": nn.ModuleList(ResidualBlock(out_channels) for _ in range(repeats)),
                    }
                )
            )
            in_channels = out_channels
        self.stages = stages

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        features = []
        for index, stage in enumerate(self.stages):
            x = stage["down"](x)
            for block in stage["blocks"]:
                x = block(x)
            # Keep the maps that the detection heads read: 256, 512, and 1024.
            if index >= 2:
                features.append(x)
        stride8, stride16, stride32 = features
        return stride8, stride16, stride32


class ConvSet(nn.Module):
    """Five-convolution stack in front of each detection head."""

    def __init__(self, in_channels: int, channels: int) -> None:
        super().__init__()
        self.conv1 = ConvLayer(in_channels, channels, kernel_size=1)
        self.conv2 = ConvLayer(channels, channels * 2, kernel_size=3)
        self.conv3 = ConvLayer(channels * 2, channels, kernel_size=1)
        self.conv4 = ConvLayer(channels, channels * 2, kernel_size=3)
        self.conv5 = ConvLayer(channels * 2, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x


def normalize_anchors(anchors) -> tuple[tuple[float, float], ...]:
    if anchors is None:
        return ANCHORS
    pairs = tuple((float(width), float(height)) for width, height in anchors)
    if len(pairs) != 9:
        raise ValueError(f"YOLOv3 uses nine anchors, got {len(pairs)}")
    return pairs


class ScaleSpec:
    """Anchor mask for one YOLO head. Not a module; it has no parameters."""

    def __init__(self, mask: tuple[int, ...], scale_index: int, name: str, anchors=None) -> None:
        self.mask = tuple(mask)
        self.anchors = normalize_anchors(anchors)
        self.scale_index = scale_index
        self.name = name


class YOLOv3(nn.Module):
    """Full detector.

    `forward` returns raw prediction tensors, one per scale, in large,
    medium, small order. Channels are `3 * (5 + num_classes)`: the box
    offsets, objectness logit, and independent class logits. `anchors` are
    either the nine published COCO clusters or a fresh k-means fit on the
    training labels. They are not learned parameters.
    """

    def __init__(self, num_classes: int = 80, anchors=None) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError("num_classes must be positive")
        self.num_classes = num_classes
        self.anchors = normalize_anchors(anchors)
        self.backbone = Darknet53()
        out_channels = 3 * (5 + num_classes)

        self.neck1 = ConvSet(1024, 512)
        self.head1 = nn.Sequential(
            ConvLayer(512, 1024, kernel_size=3),
            ConvLayer(1024, out_channels, kernel_size=1, batch_normalize=False, activation="linear"),
        )
        self.lateral1 = ConvLayer(512, 256, kernel_size=1)

        self.neck2 = ConvSet(256 + 512, 256)
        self.head2 = nn.Sequential(
            ConvLayer(256, 512, kernel_size=3),
            ConvLayer(512, out_channels, kernel_size=1, batch_normalize=False, activation="linear"),
        )
        self.lateral2 = ConvLayer(256, 128, kernel_size=1)

        self.neck3 = ConvSet(128 + 256, 128)
        self.head3 = nn.Sequential(
            ConvLayer(128, 256, kernel_size=3),
            ConvLayer(256, out_channels, kernel_size=1, batch_normalize=False, activation="linear"),
        )
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.yolo_layers = [
            ScaleSpec(mask, index, name, self.anchors)
            for index, (mask, name) in enumerate(zip(ANCHOR_MASKS, SCALE_NAMES))
        ]
        self._init_weights()

    def _init_weights(self) -> None:
        # He init matches Darknet's `sqrt(2 / fan_in)` Gaussian, with the
        # leaky slope used by every hidden layer. Objectness starts near
        # 0.01 so the huge background set does not dominate the first steps.
        obj_bias = math.log(0.01 / 0.99)
        for layer in self.iter_convs():
            nn.init.kaiming_normal_(layer.conv.weight, mode="fan_in", nonlinearity="leaky_relu", a=0.1)
            if layer.conv.bias is not None:
                nn.init.zeros_(layer.conv.bias)
            if layer.bn is not None:
                nn.init.ones_(layer.bn.weight)
                nn.init.zeros_(layer.bn.bias)
        for layer in self.prediction_convs():
            bias = layer.conv.bias.data.view(3, 5 + self.num_classes)
            bias[:, 4].fill_(obj_bias)
            layer.conv.bias.data.copy_(bias.reshape(-1))

    def prediction_convs(self) -> tuple[ConvLayer, ConvLayer, ConvLayer]:
        return self.head1[1], self.head2[1], self.head3[1]

    def iter_convs(self):
        """Convolutional layers in Darknet weight-file order."""
        yield self.backbone.stem
        for stage in self.backbone.stages:
            yield stage["down"]
            for block in stage["blocks"]:
                yield block.conv1
                yield block.conv2
        for neck, head, lateral in (
            (self.neck1, self.head1, self.lateral1),
            (self.neck2, self.head2, self.lateral2),
            (self.neck3, self.head3, None),
        ):
            yield neck.conv1
            yield neck.conv2
            yield neck.conv3
            yield neck.conv4
            yield neck.conv5
            yield head[0]
            yield head[1]
            if lateral is not None:
                yield lateral

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        if x.shape[-1] != x.shape[-2] or x.shape[-1] % 32 != 0:
            raise ValueError(
                f"YOLOv3 expects a square input whose size is a multiple of 32, got {tuple(x.shape[-2:])}"
            )
        stride8, stride16, stride32 = self.backbone(x)

        large = self.neck1(stride32)
        pred_large = self.head1(large)
        up_large = self.upsample(self.lateral1(large))

        medium = self.neck2(torch.cat((up_large, stride16), dim=1))
        pred_medium = self.head2(medium)
        up_medium = self.upsample(self.lateral2(medium))

        small = self.neck3(torch.cat((up_medium, stride8), dim=1))
        pred_small = self.head3(small)
        return [pred_large, pred_medium, pred_small]
