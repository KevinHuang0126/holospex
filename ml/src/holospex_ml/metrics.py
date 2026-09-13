"""Pixel metrics from an aggregated confusion matrix, with no background inflation."""

from __future__ import annotations

from typing import Any

import numpy as np

IGNORE_INDEX = 255


def confusion_matrix(target: np.ndarray, prediction: np.ndarray, num_classes: int, *, ignore_index: int = IGNORE_INDEX) -> np.ndarray:
    """Rows are truth, columns predictions; ignored target pixels never score."""
    target, prediction = np.asarray(target), np.asarray(prediction)
    if target.shape != prediction.shape:
        raise ValueError("Target and prediction shapes must match.")
    if num_classes < 2:
        raise ValueError("At least background and one foreground class are required.")
    for name, values in [("target", target), ("prediction", prediction)]:
        if not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"{name} must contain integer training indices.")
    # Filter BEFORE range validation and counting. Predictions in an unlabeled
    # region are neither false positives nor evidence of correct background.
    scored = target != ignore_index
    target, prediction = target[scored], prediction[scored]
    for name, values in [("target", target), ("prediction", prediction)]:
        if values.size and (values.min() < 0 or values.max() >= num_classes):
            raise ValueError(f"{name} contains an out-of-range training index.")
    encoded = target.astype(np.int64).ravel() * num_classes + prediction.astype(np.int64).ravel()
    return np.bincount(encoded, minlength=num_classes**2).reshape(num_classes, num_classes)


def metrics_from_confusion(confusion: np.ndarray, classes: list[dict[str, Any]], *, ignored_pixels: int = 0) -> dict[str, Any]:
    """Aggregate pixels across the split before computing per-class IoU/Dice.

    Classes absent from BOTH truth and prediction have no defined score and are
    excluded from the macro mean. False positives for an absent truth class do
    count as zero. Index 0 is background and excluded from foreground averages.
    """
    confusion = np.asarray(confusion)
    if confusion.shape != (len(classes), len(classes)):
        raise ValueError("Confusion matrix shape must match the label map.")
    if [entry.get("index") for entry in classes] != list(range(len(classes))):
        raise ValueError("Classes must be in contiguous training-index order.")
    if not np.issubdtype(confusion.dtype, np.integer) or np.any(confusion < 0):
        raise ValueError("Confusion entries must be nonnegative integer pixel counts.")
    if isinstance(ignored_pixels, bool) or not isinstance(ignored_pixels, int) or ignored_pixels < 0:
        raise ValueError("ignored_pixels must be a nonnegative integer.")
    per_class: list[dict[str, Any]] = []
    for index, entry in enumerate(classes):
        intersection = int(confusion[index, index])
        support = int(confusion[index, :].sum())
        predicted = int(confusion[:, index].sum())
        union = support + predicted - intersection
        per_class.append({
            **entry,
            "iou": intersection / union if union else None,
            "dice": 2 * intersection / (support + predicted) if support + predicted else None,
            "support_pixels": support,
            "predicted_pixels": predicted,
        })
    foreground = [entry for entry in per_class[1:] if entry["iou"] is not None]
    total = int(confusion.sum())
    return {
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
        "foreground_macro_iou": sum(entry["iou"] for entry in foreground) / len(foreground) if foreground else None,
        "foreground_macro_dice": sum(entry["dice"] for entry in foreground) / len(foreground) if foreground else None,
        "foreground_classes_scored": len(foreground),
        "pixel_accuracy": int(np.trace(confusion)) / total if total else None,
        "scored_pixels": total,
        "ignored_pixels": ignored_pixels,
        "total_pixels": total + ignored_pixels,
    }
