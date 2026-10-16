"""Draw the labeled synthetic scenes used by the viewer and the training set.

Running this again reproduces the same files. Photographs in samples/photos
are separate; this script only writes samples/synthetic.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "samples" / "synthetic"
IMAGES = OUT / "images"
LABELS = OUT / "labels"

COLORS = {
    0: (214, 92, 74),
    1: (54, 122, 168),
    2: (214, 168, 74),
}
NAMES = {0: "disc", 1: "tile", 2: "wedge"}


def draw_shape(pen: ImageDraw.ImageDraw, class_id: int, box, color) -> None:
    x, y, w, h = box
    shadow = (x + 7, y + 8, x + w + 7, y + h + 8)
    shade = tuple(max(0, channel - 40) for channel in color)
    if class_id == 0:
        pen.ellipse(shadow, fill=shade)
        pen.ellipse((x, y, x + w, y + h), fill=color)
    elif class_id == 1:
        radius = int(min(w, h) * 0.18)
        pen.rounded_rectangle(shadow, radius=radius, fill=shade)
        pen.rounded_rectangle((x, y, x + w, y + h), radius=radius, fill=color)
    else:
        def triangle(origin):
            ox, oy = origin
            return [(ox + w / 2, oy + 4), (ox + w - 4, oy + h - 4), (ox + 4, oy + h - 4)]

        pen.polygon(triangle((x + 7, y + 8)), fill=shade)
        pen.polygon(triangle((x, y)), fill=color)


def write_scene(name: str, size: tuple[int, int], background: tuple[int, int, int], objects: list[tuple]) -> None:
    image = Image.new("RGB", size, background)
    pen = ImageDraw.Draw(image)
    width, height = size
    lines = []
    for class_id, x, y, w, h in objects:
        draw_shape(pen, class_id, (x, y, w, h), COLORS[class_id])
        cx = (x + w / 2) / width
        cy = (y + h / 2) / height
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w / width:.6f} {h / height:.6f}")
    image.save(IMAGES / f"{name}.png")
    (LABELS / f"{name}.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    LABELS.mkdir(parents=True, exist_ok=True)
    scenes = [
        ("large_disc", (640, 480), (236, 230, 218), [(0, 170, 90, 300, 300)]),
        ("three_tiles", (800, 600), (226, 232, 228), [
            (1, 60, 80, 180, 140),
            (1, 310, 220, 200, 160),
            (1, 540, 70, 160, 220),
        ]),
        ("portrait_wedges", (480, 720), (232, 224, 214), [
            (2, 70, 60, 180, 160),
            (2, 210, 280, 200, 180),
            (2, 50, 500, 150, 140),
        ]),
        ("wide_mixed", (960, 540), (220, 226, 232), [
            (0, 40, 160, 200, 200),
            (1, 300, 80, 220, 160),
            (2, 600, 180, 240, 200),
            (0, 820, 300, 100, 100),
        ]),
        ("small_crowd", (512, 512), (240, 236, 228), [
            (0, 30, 40, 48, 48),
            (1, 120, 70, 56, 40),
            (2, 230, 36, 60, 54),
            (0, 360, 80, 40, 40),
            (1, 70, 220, 70, 50),
            (2, 200, 250, 80, 70),
            (0, 360, 280, 52, 52),
            (1, 40, 380, 90, 60),
            (2, 280, 400, 70, 64),
        ]),
        ("corner_wedge", (700, 500), (228, 226, 220), [(2, 20, 18, 120, 110)]),
        ("paired_shapes", (640, 640), (234, 228, 220), [
            (0, 50, 50, 160, 160),
            (0, 400, 420, 140, 140),
            (1, 360, 60, 180, 130),
            (1, 70, 400, 150, 120),
            (2, 230, 240, 170, 150),
            (2, 450, 300, 130, 120),
        ]),
        ("strip_of_discs", (900, 400), (222, 230, 226), [
            (0, 40, 140, 110, 110),
            (0, 220, 90, 150, 150),
            (0, 460, 160, 90, 90),
            (0, 680, 120, 160, 160),
        ]),
    ]
    for name, size, background, objects in scenes:
        write_scene(name, size, background, objects)
    print(f"Wrote {len(scenes)} scenes to {OUT}")
    print("classes:", ", ".join(f"{index} {name}" for index, name in NAMES.items()))


if __name__ == "__main__":
    main()
