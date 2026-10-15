"""Architecture checks against the released YOLOv3 configuration."""

from pathlib import Path

import pytest
import torch

from yolov3.config import ANCHOR_MASKS, ANCHORS, DARKNET_WEIGHT_FLOATS
from yolov3.model import YOLOv3
from yolov3.weights import darknet_weight_count

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "yolov3" / "cfg" / "yolov3.cfg"


def parse_cfg(path: Path):
    blocks = []
    current = None
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            if current is not None:
                blocks.append(current)
            current = {"type": line[1:-1]}
        else:
            key, value = line.split("=", 1)
            current[key.strip()] = value.strip()
    if current is not None:
        blocks.append(current)
    return blocks


def test_output_shapes_follow_the_three_strides():
    model = YOLOv3().eval()
    with torch.no_grad():
        large, medium, small = model(torch.zeros(2, 3, 416, 416))
    assert tuple(large.shape) == (2, 255, 13, 13)
    assert tuple(medium.shape) == (2, 255, 26, 26)
    assert tuple(small.shape) == (2, 255, 52, 52)
    with torch.no_grad():
        outputs = model(torch.zeros(1, 3, 608, 608))
    assert [tuple(t.shape[-2:]) for t in outputs] == [(19, 19), (38, 38), (76, 76)]


def test_weight_layout_matches_the_official_file():
    model = YOLOv3()
    convs = list(model.iter_convs())
    assert len(convs) == 75
    assert darknet_weight_count(model) == DARKNET_WEIGHT_FLOATS
    prediction = model.prediction_convs()
    assert all(not layer.batch_normalize and layer.conv.bias is not None for layer in prediction)
    hidden = [layer for layer in convs if layer not in prediction]
    assert all(layer.batch_normalize and layer.conv.bias is None for layer in hidden)
    # Low objectness prior, so background cells start easy.
    assert float(prediction[0].conv.bias.detach().view(3, -1)[0, 4]) < -4


def test_cfg_convolution_filters_match_the_module():
    blocks = parse_cfg(CFG)[1:]
    filters = [int(block["filters"]) for block in blocks if block["type"] == "convolutional"]
    model = YOLOv3()
    assert filters == [layer.conv.out_channels for layer in model.iter_convs()]
    yolo = [block for block in blocks if block["type"] == "yolo"]
    masks = [tuple(int(part) for part in block["mask"].split(",")) for block in yolo]
    assert masks == list(ANCHOR_MASKS)
    anchor_line = [part.strip() for part in yolo[0]["anchors"].split(",")]
    parsed = tuple((float(anchor_line[i]), float(anchor_line[i + 1])) for i in range(0, len(anchor_line), 2))
    assert parsed == ANCHORS
    assert [index for index, block in enumerate(blocks) if block["type"] == "yolo"] == [82, 94, 106]


def test_residual_block_is_an_identity_when_the_branch_is_zero():
    model = YOLOv3().eval()
    block = model.backbone.stages[0]["blocks"][0]
    torch.nn.init.zeros_(block.conv1.conv.weight)
    torch.nn.init.zeros_(block.conv2.conv.weight)
    source = torch.randn(1, 64, 8, 8)
    with torch.no_grad():
        assert torch.allclose(block(source), source, atol=1e-6)


def test_input_must_be_a_square_multiple_of_32():
    model = YOLOv3().eval()
    with pytest.raises(ValueError):
        model(torch.zeros(1, 3, 100, 100))
    with pytest.raises(ValueError):
        model(torch.zeros(1, 3, 416, 320))
