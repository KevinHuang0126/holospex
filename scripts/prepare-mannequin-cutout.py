"""Reproduce the user-authorized background cutout of the known mannequin photo.

Requires Pillow only. Run from any directory:
    python scripts/prepare-mannequin-cutout.py --qa

The fixed source remains unchanged. The derivative retains every RGB value and
the original 894 x 569 pixel coordinate system; only alpha changes. White pixels
connected to the canvas border or documented empty cable/finger gaps are removed.
Near-white pixels immediately along that boundary receive a conservative alpha
transition. This is a matte for this specific photo, not a general anatomy mask.
"""

from argparse import ArgumentParser
from collections import deque
from hashlib import sha256
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "assets/demo/mannequin.png"
TARGET = ROOT / "assets/demo/mannequin-cutout.png"
SOURCE_SHA256 = "cbeb22f2c521d8ecad5be3ed9061bd3d707c065c01b02dbf5eb426f38dbf4c51"

# Empty background enclosed by demonstration cable loops in this exact photo.
# Other finger/limb gaps remain connected to the outside through white pixels.
HOLE_SEEDS = {
    "cable loop above head": (83, 224),
    "left compartment between head cables": (92, 241),
    "right compartment between head cables": (127, 224),
    "cable loop below head": (80, 363),
    "cable loop beside upper hand": (496, 165),
    "gap between upper fingertips": (482, 122),
    "middle gap between lower fingers": (510, 404),
    "bottom gap between lower fingers": (487, 423),
    "lower finger cable gap left": (502, 394),
    "lower finger cable gap center": (506, 393),
    "lower finger cable gap right": (509, 392),
    "lower finger cable gap tip": (514, 390),
}


def create_cutout() -> dict:
    source_bytes = SOURCE.read_bytes()
    if sha256(source_bytes).hexdigest() != SOURCE_SHA256:
        raise ValueError("The source changed. Review background seeds before preparing a new image.")
    original = Image.open(SOURCE).convert("RGB")
    if original.size != (894, 569):
        raise ValueError("The mannequin must retain its original 894 x 569 dimensions.")
    width, height = original.size
    raw = original.tobytes()
    colors = list(zip(raw[0::3], raw[1::3], raw[2::3]))
    # Distinguish neutral white backdrop from cream tendons, yellow nerves and
    # other light but colored mannequin details. Flooding protects closed detail.
    candidate = bytearray(min(rgb) >= 230 and max(rgb) - min(rgb) <= 20 for rgb in colors)
    removed = bytearray(width * height)
    queue = deque()

    def seed(x: int, y: int) -> None:
        index = y * width + x
        if candidate[index] and not removed[index]:
            removed[index] = 1
            queue.append(index)

    for x in range(width):
        seed(x, 0)
        seed(x, height - 1)
    for y in range(height):
        seed(0, y)
        seed(width - 1, y)
    for name, (x, y) in HOLE_SEEDS.items():
        if not candidate[y * width + x]:
            raise ValueError(f"Known empty {name} seed is not white background.")
        seed(x, y)
    while queue:
        index = queue.popleft()
        x, y = index % width, index // width
        if x:
            seed(x - 1, y)
        if x + 1 < width:
            seed(x + 1, y)
        if y:
            seed(x, y - 1)
        if y + 1 < height:
            seed(x, y + 1)

    background = Image.frombytes("L", original.size, bytes(value * 255 for value in removed))
    near_background = background.filter(ImageFilter.MaxFilter(3))
    adjacent = near_background.tobytes()
    alpha = bytearray(width * height)
    for index, rgb in enumerate(colors):
        if removed[index]:
            continue
        # Soften only the one-pixel silhouette edge; do not smooth anatomy inside.
        alpha[index] = round(255 * min(1, (255 - min(rgb)) / 55)) if adjacent[index] else 255
    result = original.convert("RGBA")
    result.putalpha(Image.frombytes("L", original.size, bytes(alpha)))
    result.save(TARGET, optimize=True)
    if result.convert("RGB").tobytes() != original.tobytes():
        raise AssertionError("Cutout preparation changed source RGB values.")
    if SOURCE.read_bytes() != source_bytes:
        raise AssertionError("Cutout preparation changed the source file.")
    return {
        "source": str(SOURCE.relative_to(ROOT)), "target": str(TARGET.relative_to(ROOT)),
        "dimensions": list(result.size), "mode": result.mode,
        "transparent_pixels": alpha.count(0), "opaque_pixels": alpha.count(255),
        "partial_alpha_pixels": len(alpha) - alpha.count(0) - alpha.count(255),
        "foreground_bbox": result.getchannel("A").getbbox(),
        "source_rgb_unchanged": True, "source_sha256": SOURCE_SHA256,
        "target_sha256": sha256(TARGET.read_bytes()).hexdigest(),
    }


def save_qa() -> None:
    output = ROOT / "runs/mannequin-cutout-qa"
    output.mkdir(parents=True, exist_ok=True)
    cutout = Image.open(TARGET).convert("RGBA")
    dark = Image.new("RGBA", cutout.size, "#0c2028")
    dark.alpha_composite(cutout)
    dark.convert("RGB").save(output / "dark.png")
    checker = Image.new("RGBA", cutout.size, "#364953")
    draw = ImageDraw.Draw(checker)
    for y in range(0, cutout.height, 20):
        for x in range(0, cutout.width, 20):
            if (x // 20 + y // 20) % 2:
                draw.rectangle((x, y, x + 19, y + 19), fill="#71808a")
    checker.alpha_composite(cutout)
    checker.convert("RGB").save(output / "checkerboard.png")
    cutout.getchannel("A").save(output / "alpha.png")
    for name, bounds in {"head-cables": (20, 180, 165, 395), "upper-hand": (410, 90, 540, 190), "lower-hand": (425, 365, 540, 445)}.items():
        crop = dark.crop(bounds)
        crop.resize((crop.width * 3, crop.height * 3)).convert("RGB").save(output / f"{name}.png")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--qa", action="store_true", help="Write dark/checkerboard previews only under ignored runs/.")
    args = parser.parse_args()
    print(json.dumps(create_cutout(), indent=2))
    if args.qa:
        save_qa()
