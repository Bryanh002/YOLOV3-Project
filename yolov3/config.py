"""Constants from YOLOv3: the COCO anchors, class list, and training schedule."""

from __future__ import annotations

# Pixel anchors at the 416 reference resolution, smallest cluster first.
# Masks assign the last three to the stride-32 head, the middle three to
# stride 16, and the first three to stride 8.
ANCHORS: tuple[tuple[float, float], ...] = (
    (10, 13),
    (16, 30),
    (33, 23),
    (30, 61),
    (62, 45),
    (59, 119),
    (116, 90),
    (156, 198),
    (373, 326),
)

# Prediction order follows the network: large objects, then medium, then small.
ANCHOR_MASKS: tuple[tuple[int, int, int], ...] = (
    (6, 7, 8),
    (3, 4, 5),
    (0, 1, 2),
)

SCALE_NAMES: tuple[str, ...] = ("large", "medium", "small")

# Released yolov3.cfg. The paper text uses 0.5; the trained weights use 0.7.
IGNORE_THRESH = 0.7
TRUTH_THRESH = 1.0

DEFAULT_IMG_SIZE = 416
NETWORK_STRIDE = 32
MULTI_SCALE_SIZES: tuple[int, ...] = tuple(range(320, 608 + 1, 32))

# Darknet [net] block used to train the released model.
LEARNING_RATE = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
BURN_IN = 1000
MAX_BATCHES = 500200
LR_STEPS = (400000, 450000)
LR_SCALES = (0.1, 0.1)

HUE = 0.1
SATURATION = 1.5
EXPOSURE = 1.5
JITTER = 0.3

# Official weight file. SHA-256 covers the bytes published by pjreddie.
DARKNET_WEIGHTS_URL = "https://pjreddie.com/media/files/yolov3.weights"
DARKNET_WEIGHTS_SHA256 = "523e4e69e1d015393a1b0a441cef1d9c7659e3eb2d7e15f793f060a21b32f297"
DARKNET_WEIGHT_FLOATS = 62_001_757

# coco.names order. This is the label order baked into yolov3.weights.
COCO_CLASSES: tuple[str, ...] = (
    "person",
    "bicycle",
    "car",
    "motorbike",
    "aeroplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "sofa",
    "pottedplant",
    "bed",
    "diningtable",
    "toilet",
    "tvmonitor",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)

# COCO category ids, aligned with COCO_CLASSES. The annotation dump skips
# ids that were removed from the 80-class detection set.
COCO_CATEGORY_IDS: tuple[int, ...] = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90,
)

CLASS_PALETTE: tuple[str, ...] = (
    "#dff25a",
    "#4ad6c8",
    "#ffb020",
    "#ff5d73",
    "#8eb6ff",
    "#7dffb3",
    "#ff8ad4",
    "#ffe27a",
    "#c9a6ff",
    "#ff9f6e",
    "#6ee7ff",
    "#f2f2f2",
)


def class_color(class_id: int) -> str:
    return CLASS_PALETTE[int(class_id) % len(CLASS_PALETTE)]
