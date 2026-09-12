"""Record preprocessing geometry so overlays return to the original image exactly.

Coordinates describe pixel edges: the bottom-right image boundary is (width,
height). Never export model-input coordinates as original-image coordinates.
This module transforms geometry only; the model adapter must perform the actual
image resize and padding using the exact dimensions returned here.
"""

from dataclasses import dataclass
import math
from typing import Iterable

Point = tuple[float, float]


@dataclass(frozen=True)
class ImageSize:
    width: int
    height: int

    def __post_init__(self) -> None:
        for dimension in (self.width, self.height):
            if type(dimension) is not int or dimension <= 0:
                raise ValueError("Image dimensions must be positive integers")


@dataclass(frozen=True)
class ResizeTransform:
    """The exact resize/padding operation, including integer rounding.

    Keep one transform per input frame. Letterbox scale factors can differ
    slightly after rounding; using a single nominal factor for the inverse
    causes drift at image edges.
    """

    original: ImageSize
    processed: ImageSize
    resized: ImageSize
    pad_left: int = 0
    pad_top: int = 0

    def __post_init__(self) -> None:
        for padding in (self.pad_left, self.pad_top):
            if type(padding) is not int or padding < 0:
                raise ValueError("Padding must be a nonnegative integer")
        if self.pad_left + self.resized.width > self.processed.width:
            raise ValueError("Resized image exceeds processed width")
        if self.pad_top + self.resized.height > self.processed.height:
            raise ValueError("Resized image exceeds processed height")

    @classmethod
    def stretch(cls, original: ImageSize, processed: ImageSize) -> "ResizeTransform":
        """Resize directly to the target, with independent x/y scales."""
        return cls(original=original, processed=processed, resized=processed)

    @classmethod
    def letterbox(cls, original: ImageSize, processed: ImageSize) -> "ResizeTransform":
        """Fit inside the target with centered integer padding.

        Use ``resized`` for the actual resize, and ``pad_*`` for actual padding.
        An existing model preprocessor with different rounding must instead
        construct this dataclass from its recorded dimensions.
        """
        scale = min(processed.width / original.width, processed.height / original.height)
        resized = ImageSize(
            width=max(1, min(processed.width, round(original.width * scale))),
            height=max(1, min(processed.height, round(original.height * scale))),
        )
        return cls(
            original=original,
            processed=processed,
            resized=resized,
            pad_left=(processed.width - resized.width) // 2,
            pad_top=(processed.height - resized.height) // 2,
        )

    @property
    def scale_x(self) -> float:
        return self.resized.width / self.original.width

    @property
    def scale_y(self) -> float:
        return self.resized.height / self.original.height

    @property
    def pad_right(self) -> int:
        return self.processed.width - self.resized.width - self.pad_left

    @property
    def pad_bottom(self) -> int:
        return self.processed.height - self.resized.height - self.pad_top

    def to_processed(self, point: Point) -> Point:
        x, y = _finite_point(point)
        return (x * self.scale_x + self.pad_left, y * self.scale_y + self.pad_top)

    def to_original(self, point: Point) -> Point:
        """Invert preprocessing without hiding detections in padded regions.

        Out-of-image points stay out of bounds. The adapter must clip masks or
        polygon geometry to the image before export, then reject degenerate
        shapes. Clamping individual vertices is not general polygon clipping.
        """
        x, y = _finite_point(point)
        return ((x - self.pad_left) / self.scale_x, (y - self.pad_top) / self.scale_y)

    def polygon_to_original(self, polygon: Iterable[Point]) -> list[Point]:
        points = [self.to_original(point) for point in polygon]
        if len(points) < 3:
            raise ValueError("A polygon needs at least three vertices")
        return points


def _finite_point(point: Point) -> Point:
    x, y = point
    if isinstance(x, bool) or isinstance(y, bool) or not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("Point coordinates must be finite numbers")
    return (float(x), float(y))
