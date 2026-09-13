"""Paired segmentation training, validation selection, and explicit evaluation.

No test-set pixels enter fitting or checkpoint selection. A deliberately limited
run is saved as such and verifies plumbing, not the quality of a surgical model.
"""

from __future__ import annotations

import json
import hashlib
import importlib.metadata
import math
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .dataset import encode_mask, read_image, read_semantic_mask
from .metrics import IGNORE_INDEX, confusion_matrix, metrics_from_confusion
from .model import (
    ARCHITECTURE, MODEL_ID, NORMALIZATION_MEAN, NORMALIZATION_STD,
    build_model, freeze_batchnorm, load_checkpoint, resolve_device,
)


class SegmentationDataset(Dataset):
    """Resize the RGB image and encoded mask together, with label-safe sampling."""

    def __init__(self, samples: list[dict[str, Any]], classes: list[dict[str, Any]], width: int, height: int, ignore_source_ids: tuple[int, ...] | list[int] = ()):
        self.samples, self.classes = samples, classes
        self.size = (width, height)
        self.ignore_source_ids = tuple(ignore_source_ids)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        image = read_image(Path(sample["imagePath"]))
        mask = read_semantic_mask(Path(sample["maskPath"]))
        if image.shape[:2] != mask.shape:
            raise ValueError(f"Image/mask dimensions differ for {sample['imagePath']}.")
        encoded = encode_mask(mask, self.classes, ignore_source_ids=self.ignore_source_ids)
        resized_image = np.array(Image.fromarray(image).resize(self.size, Image.Resampling.BILINEAR), dtype=np.float32)
        resized_mask = np.array(Image.fromarray(encoded.astype(np.int32)).resize(self.size, Image.Resampling.NEAREST), dtype=np.int64)
        pixels = torch.from_numpy(resized_image.transpose(2, 0, 1).copy()) / 255.0
        mean = torch.tensor(NORMALIZATION_MEAN, dtype=torch.float32)[:, None, None]
        std = torch.tensor(NORMALIZATION_STD, dtype=torch.float32)[:, None, None]
        return (pixels - mean) / std, torch.from_numpy(resized_mask)


def _validate_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    classes = manifest.get("classes", [])
    if len(classes) < 2 or [entry.get("index") for entry in classes] != list(range(len(classes))):
        raise ValueError("Manifest requires a contiguous label map beginning at background index 0.")
    if classes[0].get("structureId") != "background":
        raise ValueError("Training index 0 must be background.")
    _ignore_settings(manifest)
    # Repeat the leakage check at training time even for a hand-edited manifest.
    video_splits: dict[str, str] = {}
    for sample in manifest.get("samples", []):
        split, video = sample["split"], str(sample["videoId"])
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown sample split {split!r}.")
        if video in video_splits and video_splits[video] != split:
            raise ValueError(f"Video {video} appears in more than one split.")
        video_splits[video] = split
    return classes


def _ignore_settings(manifest: dict[str, Any]) -> list[int]:
    """The dataset's explicit source policy maps only listed IDs to target255."""
    if manifest.get("ignoreIndex", IGNORE_INDEX) != IGNORE_INDEX:
        raise ValueError("This training adapter requires ignored targets to use index 255.")
    source_ids = manifest.get("ignoreSourceIds", [])
    if not isinstance(source_ids, (list, tuple)) or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in source_ids):
        raise ValueError("ignoreSourceIds must contain nonnegative integer source IDs.")
    if set(source_ids) & {entry["sourceId"] for entry in manifest.get("classes", [])}:
        raise ValueError("An ignored source ID cannot also identify an anatomy/background class.")
    return sorted(set(source_ids))


def _check_ignore_policy(manifest: dict[str, Any], checkpoint: dict[str, Any]) -> list[int]:
    source_ids = _ignore_settings(manifest)
    if checkpoint.get("ignore_index", IGNORE_INDEX) != IGNORE_INDEX or sorted(checkpoint.get("ignore_source_ids", [])) != source_ids:
        raise ValueError("Checkpoint and manifest ignored-label policies differ.")
    return source_ids


def _samples(manifest: dict[str, Any], split: str, limit: int | None) -> list[dict[str, Any]]:
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, or test.")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0):
        raise ValueError("Sample limits must be positive integers.")
    samples = [sample for sample in manifest["samples"] if sample["split"] == split]
    if not samples:
        raise ValueError(f"The manifest has no {split} samples.")
    return samples[:limit] if limit is not None else samples


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _save_checkpoint(path: Path, checkpoint: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def _manifest_digest(manifest: dict[str, Any]) -> str:
    """Fingerprint the exact ordered metadata; image/mask bytes are not hashed."""
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _video_ids(samples: list[dict[str, Any]]) -> list[str]:
    # JSON may serialize an ID as an integer or string; compare the same case
    # identity in both forms and preserve exact selected cases for limited runs.
    return sorted({str(sample["videoId"]) for sample in samples})


def _check_evaluation_split(samples: list[dict[str, Any]], split: str, checkpoint: dict[str, Any]) -> None:
    """Check held-out identity against checkpoint history, not just new splits."""
    if split == "train":
        return
    training = checkpoint.get("training", {})
    keys = ["train_video_ids"] + (["val_video_ids"] if split == "test" else [])
    forbidden: set[str] = set()
    for key in keys:
        if not isinstance(training.get(key), list):
            raise ValueError(f"Checkpoint lacks {key}; held-out evaluation cannot be verified. Use a checkpoint with recorded split identities.")
        forbidden.update(str(value) for value in training[key])
    overlap = set(_video_ids(samples)) & forbidden
    if overlap:
        raise ValueError(f"{split} evaluation overlaps checkpoint training/selection cases: videos {sorted(overlap)}.")


def _software_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for package in ("torch", "torchvision", "numpy", "Pillow"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unknown"
    return versions


def _balanced_class_weights(dataset: SegmentationDataset) -> dict[str, Any]:
    """Count the exact resized selected TRAIN targets; never inspect val/test."""
    counts = np.zeros(len(dataset.classes), dtype=np.int64)
    ignored_pixels = 0
    for index in range(len(dataset)):
        _, target = dataset[index]
        values = target.numpy()
        ignored_pixels += int(np.count_nonzero(values == IGNORE_INDEX))
        scored = values[values != IGNORE_INDEX]
        counts += np.bincount(scored, minlength=len(counts))
    missing = [entry["structureId"] for entry, count in zip(dataset.classes, counts) if count == 0]
    if missing:
        raise ValueError(
            f"Balanced weighting has no scored training pixels for {missing} after resizing and ignore handling. "
            "Use more training samples (expand --limit-train), increase training resolution, or use --class-weighting none."
        )
    frequencies = counts.astype(np.float64) / int(counts.sum())
    raw = 1.0 / np.sqrt(frequencies)
    foreground_mean = float(raw[1:].mean())
    # Record the actual float32 values supplied to CrossEntropyLoss, not merely
    # a higher-precision approximation of the configured training objective.
    weights = np.clip(raw / foreground_mean, 0.1, 5.0).astype(np.float32)
    return {
        "scope": "selected_train_samples_after_exact_resize_and_ignore_handling",
        "sample_count": len(dataset), "counts": counts.tolist(),
        "scored_pixels": int(counts.sum()), "ignored_pixels": ignored_pixels,
        "frequencies": frequencies.tolist(), "raw_inverse_sqrt_weights": raw.tolist(),
        "foreground_mean_raw_weight": foreground_mean,
        "clip_min": 0.1, "clip_max": 5.0, "dtype": "float32",
        "weights": weights.tolist(),
        "formula": "clip((1/sqrt(count_i/sum(counts))) / mean_foreground(1/sqrt(count_i/sum(counts))), 0.1, 5.0)",
    }


def _evaluate_model(model: nn.Module, loader: DataLoader, classes: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    model.eval()
    confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)
    loss_sum, sample_count, ignored_pixels, scored_pixels = 0.0, 0, 0, 0
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    with torch.inference_mode():
        for image, target in loader:
            image, target = image.to(device), target.to(device)
            batch_ignored = int((target == IGNORE_INDEX).sum().item())
            batch_scored = target.numel() - batch_ignored
            if batch_scored == 0:
                raise ValueError("Evaluation batch contains only ignored pixels; no loss or score is defined.")
            logits = model(image)["out"]
            loss = criterion(logits, target)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite evaluation loss; refusing to report invalid metrics.")
            count = image.shape[0]
            loss_sum += float(loss.item()) * batch_scored
            sample_count += count
            ignored_pixels += batch_ignored
            scored_pixels += batch_scored
            confusion += confusion_matrix(target.cpu().numpy(), logits.argmax(dim=1).cpu().numpy(), len(classes))
    return {**metrics_from_confusion(confusion, classes, ignored_pixels=ignored_pixels), "loss": loss_sum / scored_pixels, "sample_count": sample_count, "ignore_index": IGNORE_INDEX, "loss_reduction": "mean_over_scored_pixels"}


def train(
    manifest: dict, output_dir: Path, *, epochs: int = 3, batch_size: int = 2,
    lr: float = 0.001, width: int = 448, height: int = 256, device: str = "auto",
    pretrained: bool = True, limit_train: int | None = None,
    limit_val: int | None = None, seed: int = 42, class_weighting: str = "none",
) -> dict:
    classes = _validate_manifest(manifest)
    ignore_source_ids = _ignore_settings(manifest)
    for name, value in [("epochs", epochs), ("batch_size", batch_size), ("width", width), ("height", height)]:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be positive and finite.")
    if class_weighting not in {"none", "balanced"}:
        raise ValueError("class_weighting must be none or balanced.")
    train_samples, val_samples = _samples(manifest, "train", limit_train), _samples(manifest, "val", limit_val)
    selected_device = resolve_device(device)
    output_dir = Path(output_dir)
    if any((output_dir / name).exists() for name in ("best.pt", "last.pt", "config.json")):
        raise ValueError("Output directory already contains a training run; choose a new output directory.")
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    train_dataset = SegmentationDataset(train_samples, classes, width, height, ignore_source_ids)
    weighting_details = None
    if class_weighting == "balanced":
        print(f"Computing balanced weights from {len(train_dataset)} selected training masks only.", flush=True)
        weighting_details = _balanced_class_weights(train_dataset)
        print(f"Training pixel counts={weighting_details['counts']} weights={weighting_details['weights']}", flush=True)
    class_weights = weighting_details["weights"] if weighting_details else None
    version = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    config = {
        "architecture": ARCHITECTURE, "model_id": MODEL_ID, "run_id": version,
        "epochs": epochs, "batch_size": batch_size, "learning_rate": lr,
        "input_size": {"width": width, "height": height}, "device": str(selected_device), "seed": seed,
        "pretrained": pretrained, "initialization": "COCO_WITH_VOC_LABELS_V1" if pretrained else "random",
        "class_weighting": class_weighting, "class_weights": class_weights,
        "class_weighting_details": weighting_details,
        "training_loss_reduction": "mean_over_target_class_weights" if class_weights else "mean_over_scored_pixels",
        "validation_loss_weighting": "none",
        "limited_run": limit_train is not None or limit_val is not None,
        "limit_train": limit_train, "limit_val": limit_val,
        "train_samples": len(train_samples), "val_samples": len(val_samples),
        "train_video_ids": _video_ids(train_samples), "val_video_ids": _video_ids(val_samples),
        "manifest_sha256": _manifest_digest(manifest),
        "manifest_digest_scope": "Canonical ordered JSON metadata; image and mask contents are not hashed",
        "software_versions": _software_versions(),
        "batchnorm": "frozen", "selection_metric": "foreground_macro_iou",
        "metric_resolution": {"width": width, "height": height},
        "classes": classes,
        "ignore_index": IGNORE_INDEX, "ignore_source_ids": ignore_source_ids,
        "ignored_pixel_policy": "Explicit source IDs are excluded from loss and pixel metrics; never relabeled as background.",
        "dataset": {key: manifest.get(key) for key in ("dataset", "root", "report")},
    }
    _write_json(output_dir / "config.json", config)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator, num_workers=0)
    val_loader = DataLoader(SegmentationDataset(val_samples, classes, width, height, ignore_source_ids), batch_size=batch_size, shuffle=False, num_workers=0)
    model = build_model(len(classes), pretrained=pretrained).to(selected_device)
    freeze_batchnorm(model)
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=lr)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=selected_device) if class_weights else None
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, weight=weight_tensor)
    history: list[dict[str, Any]] = []
    best_score = -math.inf
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        started = time.monotonic()
        model.train()
        freeze_batchnorm(model)  # train() resets module modes, so repeat each epoch.
        loss_sum, sample_count, ignored_pixels, scored_pixels = 0.0, 0, 0, 0
        loss_normalizer = 0.0
        for batch_index, (image, target) in enumerate(train_loader, start=1):
            image, target = image.to(selected_device), target.to(selected_device)
            batch_ignored = int((target == IGNORE_INDEX).sum().item())
            batch_scored = target.numel() - batch_ignored
            if batch_scored == 0:
                raise ValueError("Training batch contains only ignored pixels; no supervision remains.")
            optimizer.zero_grad(set_to_none=True)
            outputs = model(image)
            loss = criterion(outputs["out"], target)
            if "aux" in outputs:
                loss = loss + 0.4 * criterion(outputs["aux"], target)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss; stopping without reporting a successful run.")
            loss.backward()
            optimizer.step()
            # Weighted CE divides by target weights, not the number of pixels.
            # Match that denominator when aggregating the reported epoch loss.
            batch_normalizer = float((weight_tensor[target.clamp_max(len(classes) - 1)] * (target != IGNORE_INDEX)).sum().item()) if weight_tensor is not None else batch_scored
            loss_sum += float(loss.item()) * batch_normalizer
            loss_normalizer += batch_normalizer
            sample_count += image.shape[0]
            ignored_pixels += batch_ignored
            scored_pixels += batch_scored
            if batch_index == 1 or batch_index % 25 == 0 or batch_index == len(train_loader):
                print(f"epoch={epoch}/{epochs} batch={batch_index}/{len(train_loader)} train_loss={loss_sum / loss_normalizer:.4f}", flush=True)
        validation = {
            **_evaluate_model(model, val_loader, classes, selected_device),
            "split": "val", "ignore_source_ids": ignore_source_ids,
            "metric_resolution": {"width": width, "height": height},
        }
        score = validation["foreground_macro_iou"]
        if score is None:
            raise ValueError("Validation has no scored foreground classes; cannot select a segmentation checkpoint.")
        row = {"epoch": epoch, "train_loss": loss_sum / loss_normalizer, "train_loss_normalizer": loss_normalizer, "train_scored_pixels": scored_pixels, "train_ignored_pixels": ignored_pixels, "validation": validation, "duration_seconds": time.monotonic() - started}
        history.append(row)
        checkpoint = {
            "format_version": 1, "architecture": ARCHITECTURE, "model_id": MODEL_ID,
            "model_version": f"{version}-epoch-{epoch}", "run_id": version,
            "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "classes": classes, "input_size": {"width": width, "height": height},
            "ignore_index": IGNORE_INDEX, "ignore_source_ids": ignore_source_ids,
            "class_weighting": class_weighting, "class_weights": class_weights,
            "normalization": {"mean": list(NORMALIZATION_MEAN), "std": list(NORMALIZATION_STD)},
            "epochs_trained": epoch, "seed": seed, "dataset": config["dataset"],
            "training": config, "validation": validation,
        }
        _save_checkpoint(output_dir / "last.pt", checkpoint)
        if score > best_score:
            best_score, best_epoch = score, epoch
            _save_checkpoint(output_dir / "best.pt", checkpoint)
            _write_json(output_dir / "metrics-validation.json", validation)
        _write_json(output_dir / "history.json", history)
        print(f"epoch={epoch}/{epochs} train_loss={row['train_loss']:.4f} val_foreground_iou={score:.4f} seconds={row['duration_seconds']:.1f}", flush=True)
    return {
        "checkpoint": str((output_dir / "best.pt").resolve()),
        "last_checkpoint": str((output_dir / "last.pt").resolve()),
        "best_epoch": best_epoch, "best_foreground_macro_iou": best_score,
        "limited_run": config["limited_run"], "epochs_completed": epochs,
        "history": history, "config": config,
    }


def evaluate(checkpoint_path: Path, manifest: dict, *, split: str = "test", device: str = "auto", limit: int | None = None) -> dict:
    classes = _validate_manifest(manifest)
    samples = _samples(manifest, split, limit)
    selected_device = resolve_device(device)
    model, checkpoint = load_checkpoint(checkpoint_path, selected_device)
    if checkpoint["classes"] != classes:
        raise ValueError("Checkpoint and dataset class maps differ; refuse to score mislabeled outputs.")
    _check_evaluation_split(samples, split, checkpoint)
    ignore_source_ids = _check_ignore_policy(manifest, checkpoint)
    size = checkpoint["input_size"]
    loader = DataLoader(SegmentationDataset(samples, classes, size["width"], size["height"], ignore_source_ids), batch_size=2, shuffle=False, num_workers=0)
    return {
        **_evaluate_model(model, loader, classes, selected_device),
        "split": split, "model_id": checkpoint["model_id"], "model_version": checkpoint["model_version"],
        "limited_evaluation": limit is not None, "limited_training": checkpoint["training"]["limited_run"],
        "metric_resolution": size, "checkpoint": str(Path(checkpoint_path).resolve()),
        "evaluation_video_ids": _video_ids(samples),
        "evaluation_manifest_sha256": _manifest_digest(manifest),
        "training_manifest_sha256": checkpoint["training"].get("manifest_sha256"),
        "ignore_index": IGNORE_INDEX, "ignore_source_ids": ignore_source_ids,
    }
