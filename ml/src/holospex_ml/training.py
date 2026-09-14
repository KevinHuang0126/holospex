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
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .dataset import encode_mask, read_image, read_semantic_mask
from .metrics import IGNORE_INDEX, confusion_matrix, metrics_from_confusion
from .losses import foreground_generalized_dice_loss, generalized_dice_policy, foreground_lovasz_softmax_loss, lovasz_policy
from .model import (
    ARCHITECTURE, NORMALIZATION_MEAN, NORMALIZATION_STD,
    build_model, freeze_batchnorm, load_checkpoint, load_backbone_checkpoint, model_id_for_architecture, resolve_device,
)


def _augment_pair(image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
    """Mild training perturbations; never interpolate or recolor anatomy labels.

    Flipping changes only image-plane pose, not anatomy IDs or any world-side
    interpretation. No rotation/crop can erase a tiny target or invent padding.
    """
    if random.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    image = ImageEnhance.Brightness(image).enhance(random.uniform(0.9, 1.1))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(0.9, 1.1))
    return image, mask


class SegmentationDataset(Dataset):
    """Resize the RGB image and encoded mask together, with label-safe sampling."""

    def __init__(self, samples: list[dict[str, Any]], classes: list[dict[str, Any]], width: int, height: int, ignore_source_ids: tuple[int, ...] | list[int] = (), *, augmentation: str = "none"):
        if augmentation not in {"none", "mild"}:
            raise ValueError("augmentation must be none or mild.")
        if augmentation != "none" and any(sample.get("split") != "train" for sample in samples):
            raise ValueError("Augmentation is allowed only on explicitly marked training samples.")
        self.samples, self.classes = samples, classes
        self.size = (width, height)
        self.ignore_source_ids = tuple(ignore_source_ids)
        self.augmentation = augmentation

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        image = read_image(Path(sample["imagePath"]))
        mask = read_semantic_mask(Path(sample["maskPath"]))
        if image.shape[:2] != mask.shape:
            raise ValueError(f"Image/mask dimensions differ for {sample['imagePath']}.")
        encoded = encode_mask(mask, self.classes, ignore_source_ids=self.ignore_source_ids)
        resized_image = Image.fromarray(image).resize(self.size, Image.Resampling.BILINEAR)
        resized_mask = Image.fromarray(encoded.astype(np.int32)).resize(self.size, Image.Resampling.NEAREST)
        if self.augmentation == "mild":
            resized_image, resized_mask = _augment_pair(resized_image, resized_mask)
        resized_image = np.array(resized_image, dtype=np.float32)
        resized_mask = np.array(resized_mask, dtype=np.int64)
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


def _learning_rate_factor(step: int, total_steps: int, schedule: str, warmup_steps: int) -> float:
    """Factor for a zero-based optimizer step; warmup ends at the base rate.

    Cosine reaches zero after the final update. The last actual update therefore
    still has a positive rate, including a one-step run without warmup.
    """
    if warmup_steps and step < warmup_steps:
        return (step + 1) / warmup_steps
    if schedule == "none":
        return 1.0
    progress = min(1.0, max(0.0, (step - warmup_steps) / (total_steps - warmup_steps)))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _optimizer_groups(model: nn.Module, lr: float, backbone_lr_multiplier: float) -> list[dict[str, Any]]:
    """Keep every trainable tensor exactly once; frozen BN stays excluded."""
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("The model has no trainable parameters.")
    if backbone_lr_multiplier == 1.0:
        return [{"params": parameters, "lr": lr, "name": "all"}]
    backbone = getattr(model, "backbone", None)
    if not isinstance(backbone, nn.Module):
        raise ValueError("A backbone_lr_multiplier other than 1 requires model.backbone.")
    backbone_ids = {id(parameter) for parameter in backbone.parameters()}
    groups = [
        {"params": [p for p in parameters if id(p) in backbone_ids], "lr": lr * backbone_lr_multiplier, "name": "backbone"},
        {"params": [p for p in parameters if id(p) not in backbone_ids], "lr": lr, "name": "heads"},
    ]
    if not groups[0]["params"]:
        raise ValueError("model.backbone has no trainable parameters to apply its learning-rate multiplier.")
    return [group for group in groups if group["params"]]


def _check_initial_checkpoint(
    checkpoint: dict[str, Any], manifest: dict[str, Any], architecture: str,
    train_samples: list[dict[str, Any]], val_samples: list[dict[str, Any]],
) -> None:
    """Warm-start only on the same cases, preserving selection provenance."""
    if checkpoint.get("architecture") != architecture:
        raise ValueError("Initial checkpoint and requested architecture differ.")
    if checkpoint.get("classes") != manifest["classes"]:
        raise ValueError("Initial checkpoint and manifest class maps differ.")
    _check_ignore_policy(manifest, checkpoint)
    training = checkpoint.get("training", {})
    for key, samples in (("train_video_ids", train_samples), ("val_video_ids", val_samples)):
        previous = training.get(key)
        if not isinstance(previous, list) or sorted({str(value) for value in previous}) != _video_ids(samples):
            raise ValueError(f"Initial checkpoint requires the same selected {key}; changed or missing case provenance is unsafe.")


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _case_balanced_weights(samples: list[dict[str, Any]]) -> list[float]:
    """Each selected TRAIN video has equal expected probability, not each frame.

    Use replacement and len(samples) draws per epoch. Individual epochs need not
    contain every frame/case; validation always visits every selected frame once.
    """
    if not samples or any(sample.get("split") != "train" for sample in samples):
        raise ValueError("Case-balanced sampling requires nonempty training samples only.")
    counts = Counter(str(sample["videoId"]) for sample in samples)
    return [1.0 / counts[str(sample["videoId"])] for sample in samples]


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
    architecture: str = ARCHITECTURE, augmentation: str = "none", sampling: str = "uniform",
    loss: str = "ce", dice_weight: float = 1.0, lovasz_weight: float = 0.25,
    auxiliary_loss_weight: float = 0.4,
    lr_schedule: str = "none", warmup_epochs: int = 0, weight_decay: float = 0.01,
    backbone_lr_multiplier: float = 1.0, max_duration_seconds: float | None = None,
    initial_checkpoint: Path | None = None, backbone_checkpoint: Path | None = None,
) -> dict:
    run_started = time.monotonic()
    classes = _validate_manifest(manifest)
    ignore_source_ids = _ignore_settings(manifest)
    for name, value in [("epochs", epochs), ("batch_size", batch_size), ("width", width), ("height", height)]:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be positive and finite.")
    if lr_schedule not in {"none", "cosine"}:
        raise ValueError("lr_schedule must be none or cosine.")
    if isinstance(warmup_epochs, bool) or not isinstance(warmup_epochs, int) or not 0 <= warmup_epochs < epochs:
        raise ValueError("warmup_epochs must be an integer in [0, epochs).")
    for name, value, allow_zero in (("weight_decay", weight_decay, True), ("backbone_lr_multiplier", backbone_lr_multiplier, False), ("max_duration_seconds", max_duration_seconds, False)):
        if value is None and name == "max_duration_seconds":
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
            raise ValueError(f"{name} must be {'nonnegative' if allow_zero else 'positive'} and finite.")
    if class_weighting not in {"none", "balanced"}:
        raise ValueError("class_weighting must be none or balanced.")
    model_id = model_id_for_architecture(architecture)
    if backbone_checkpoint is not None:
        if initial_checkpoint is not None:
            raise ValueError("backbone_checkpoint and initial_checkpoint are mutually exclusive.")
        if architecture != "deeplabv3_resnet50" or pretrained is not True:
            raise ValueError("backbone_checkpoint requires deeplabv3_resnet50 and pretrained=True to retain the COCO context and auxiliary features.")
        backbone_checkpoint = Path(backbone_checkpoint)
        if not backbone_checkpoint.is_file():
            raise ValueError("backbone_checkpoint must name an existing checkpoint file.")
    if augmentation not in {"none", "mild"}:
        raise ValueError("augmentation must be none or mild.")
    if sampling not in {"uniform", "case_balanced"}:
        raise ValueError("sampling must be uniform or case_balanced.")
    if loss not in {"ce", "ce_generalized_dice", "ce_lovasz"}:
        raise ValueError("loss must be ce, ce_generalized_dice, or ce_lovasz.")
    if isinstance(dice_weight, bool) or not isinstance(dice_weight, (int, float)) or not math.isfinite(dice_weight) or dice_weight < 0:
        raise ValueError("dice_weight must be nonnegative and finite.")
    if isinstance(lovasz_weight, bool) or not isinstance(lovasz_weight, (int, float)) or not math.isfinite(lovasz_weight) or lovasz_weight < 0:
        raise ValueError("lovasz_weight must be nonnegative and finite.")
    if isinstance(auxiliary_loss_weight, bool) or not isinstance(auxiliary_loss_weight, (int, float)) or not math.isfinite(auxiliary_loss_weight) or not 0 <= auxiliary_loss_weight <= 10:
        raise ValueError("auxiliary_loss_weight must be finite and in [0, 10].")
    loss_mode = loss
    use_dice = loss_mode == "ce_generalized_dice"
    use_lovasz = loss_mode == "ce_lovasz"
    use_overlap = use_dice or use_lovasz
    auxiliary_weight_label = str(float(auxiliary_loss_weight))
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
    initialization_details = None
    backbone_initialization = None
    if initial_checkpoint is not None:
        initial_checkpoint = Path(initial_checkpoint)
        model, previous_checkpoint = load_checkpoint(initial_checkpoint, selected_device)
        _check_initial_checkpoint(previous_checkpoint, manifest, architecture, train_samples, val_samples)
        previous_config = previous_checkpoint["training"]
        initialization_details = {
            "checkpoint": str(initial_checkpoint.resolve()), "checkpoint_sha256": _file_digest(initial_checkpoint),
            "model_version": previous_checkpoint.get("model_version"), "run_id": previous_checkpoint.get("run_id"),
            "epochs_trained": previous_checkpoint.get("epochs_trained"),
            "training_config_sha256": _manifest_digest({key: value for key, value in previous_config.items() if key != "config_sha256"}),
            "manifest_sha256": previous_config.get("manifest_sha256"),
            "train_video_ids": previous_config["train_video_ids"], "val_video_ids": previous_config["val_video_ids"],
            "optimizer_state": "fresh; model weights only", "case_policy": "identical_selected_train_and_val_case_sets",
        }
        del previous_checkpoint
    else:
        model = build_model(len(classes), pretrained=pretrained, architecture=architecture).to(selected_device)
        if backbone_checkpoint is not None:
            backbone_initialization = load_backbone_checkpoint(model, backbone_checkpoint, architecture=architecture)
    freeze_batchnorm(model)
    optimizer_groups = _optimizer_groups(model, lr, backbone_lr_multiplier)
    train_dataset = SegmentationDataset(train_samples, classes, width, height, ignore_source_ids, augmentation=augmentation)
    weighting_details = None
    if class_weighting == "balanced":
        print(f"Computing balanced weights from {len(train_dataset)} selected training masks only.", flush=True)
        # Count each selected training frame once without stochastic transforms
        # or replacement sampling, so loss weights remain comparable across runs.
        weighting_dataset = SegmentationDataset(train_samples, classes, width, height, ignore_source_ids)
        weighting_details = _balanced_class_weights(weighting_dataset)
        print(f"Training pixel counts={weighting_details['counts']} weights={weighting_details['weights']}", flush=True)
    class_weights = weighting_details["weights"] if weighting_details else None
    version = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    config = {
        "architecture": architecture, "model_id": model_id, "run_id": version,
        "epochs": epochs, "batch_size": batch_size, "learning_rate": lr,
        "lr_schedule": lr_schedule, "warmup_epochs": warmup_epochs,
        "weight_decay": weight_decay, "backbone_lr_multiplier": backbone_lr_multiplier,
        "optimizer": "AdamW", "max_duration_seconds": max_duration_seconds,
        "duration_limit_policy": "check_after_completed_epoch_and_checkpoint; includes_setup; always_complete_at_least_one_epoch",
        "optimizer_groups": [{"name": group["name"], "base_learning_rate": group["lr"], "parameter_count": sum(parameter.numel() for parameter in group["params"])} for group in optimizer_groups],
        "lr_schedule_details": {"step_unit": "optimizer_update", "warmup": "linear_from_1/warmup_steps_to_1", "cosine": "0.5*(1+cos(pi*(step-warmup_steps)/(total_steps-warmup_steps)))", "minimum_factor": 0.0},
        "input_size": {"width": width, "height": height}, "device": str(selected_device), "seed": seed,
        "pretrained": pretrained if initial_checkpoint is None else False,
        "initialization": "checkpoint" if initial_checkpoint is not None else ("COCO_WITH_VOC_LABELS_V1+backbone_checkpoint" if backbone_checkpoint is not None else ("COCO_WITH_VOC_LABELS_V1" if pretrained else "random")),
        "initial_checkpoint": initialization_details,
        "backbone_checkpoint": backbone_initialization,
        "initialization_recipe": None if backbone_checkpoint is None else {
            "order": ["load_generic_COCO_WITH_VOC_LABELS_V1_segmentation_weights", "replace_final_classification_convolutions_for_manifest_classes", "replace_backbone_from_explicit_checkpoint", "freeze_batchnorm", "initialize_fresh_optimizer"],
            "retained_weights": "COCO_ASPP_context_and_auxiliary_feature_layers",
            "final_classification_layers": "new_random_manifest_class_convolutions",
            "backbone_source": backbone_initialization,
        },
        "class_weighting": class_weighting, "class_weights": class_weights,
        "class_weighting_details": weighting_details,
        "loss": loss_mode, "dice_weight": float(dice_weight) if use_dice else 0.0,
        "lovasz_weight": float(lovasz_weight) if use_lovasz else 0.0,
        "auxiliary_loss_weight": float(auxiliary_loss_weight),
        "loss_details": {
            "head_objective": "CE + dice_weight * GDL" if use_dice else ("main: CE + lovasz_weight * Lovasz; auxiliary: CE" if use_lovasz else "CE"),
            "batch_objective": f"CE_main + {auxiliary_weight_label}*CE_aux + dice_weight*(GDL_main + {auxiliary_weight_label}*GDL_aux)" if use_dice else (f"CE_main + {auxiliary_weight_label}*CE_aux + lovasz_weight*Lovasz_main" if use_lovasz else f"CE_main + {auxiliary_weight_label}*CE_aux"),
            "main_head_weight": 1.0, "auxiliary_head_weight": float(auxiliary_loss_weight),
            "auxiliary_loss_enabled": auxiliary_loss_weight > 0,
            "zero_auxiliary_weight_policy": "skip_auxiliary_loss_graph; keep_model_and_checkpoint_shapes; unused_auxiliary_parameters_have_no_gradient_or_AdamW_update",
            "ce_reduction": "mean_over_target_class_weights" if class_weights else "mean_over_scored_pixels",
            "dice": generalized_dice_policy() if use_dice else None,
            "lovasz": lovasz_policy() if use_lovasz else None,
            "lovasz_weight": float(lovasz_weight) if use_lovasz else 0.0,
            "epoch_component_means": f"arithmetic_mean_over_optimizer_steps; CE_main_plus_{auxiliary_weight_label}_CE_aux_and_main_only_Lovasz" if use_lovasz else f"arithmetic_mean_over_optimizer_steps; each_head_then_main_plus_{auxiliary_weight_label}_aux",
            "epoch_ce_weighted_diagnostic": "sum(batch_combined_CE * batch_CE_denominator) / sum(batch_CE_denominator)",
            "epoch_total": "arithmetic_mean_of_actual_batch_objectives; Dice_is_not_recomputed_from_epoch_pixel_sums" if use_dice else ("arithmetic_mean_of_actual_batch_objectives; Lovasz_is_not_recomputed_over_the_epoch" if use_lovasz else "legacy_CE_denominator_weighted_mean"),
        },
        "training_loss_reduction": "mean_over_optimizer_steps" if use_overlap else ("mean_over_target_class_weights" if class_weights else "mean_over_scored_pixels"),
        "validation_loss_weighting": "none",
        "augmentation": augmentation,
        "augmentation_details": None if augmentation == "none" else {
            "scope": "train_only_after_paired_resize_before_normalization",
            "horizontal_flip_probability": 0.5,
            "brightness_factor_range": [0.9, 1.1], "contrast_factor_range": [0.9, 1.1],
            "photometric_order": ["brightness", "contrast"], "mask_transform": "horizontal_flip_only",
            "rng": "python_random_seeded_by_run_seed_num_workers_0",
        },
        "sampling": sampling,
        "sampling_details": {
            "scope": "selected_train_samples_only", "draws_per_epoch": len(train_samples),
            "replacement": sampling == "case_balanced",
            "weight_formula": "1 / selected_frame_count_for_video" if sampling == "case_balanced" else "uniform_shuffle_without_replacement",
            "video_frame_counts": dict(sorted(Counter(str(sample["videoId"]) for sample in train_samples).items())),
            "loss_class_counts_use_each_selected_frame_once": True,
        },
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
    config["config_sha256"] = _manifest_digest(config)
    _write_json(output_dir / "config.json", config)
    generator = torch.Generator().manual_seed(seed)
    sampler = None
    if sampling == "case_balanced":
        sampler = WeightedRandomSampler(_case_balanced_weights(train_samples), len(train_samples), replacement=True,
                                        generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=sampler is None, sampler=sampler, generator=generator, num_workers=0)
    val_loader = DataLoader(SegmentationDataset(val_samples, classes, width, height, ignore_source_ids), batch_size=batch_size, shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW(optimizer_groups, lr=lr, weight_decay=weight_decay)
    scheduler = None
    if lr_schedule != "none" or warmup_epochs:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: _learning_rate_factor(step, epochs * len(train_loader), lr_schedule, warmup_epochs * len(train_loader)))
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=selected_device) if class_weights else None
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, weight=weight_tensor)
    history: list[dict[str, Any]] = []
    best_score = -math.inf
    best_epoch = 0
    stop_reason = "epochs_completed"
    for epoch in range(1, epochs + 1):
        started = time.monotonic()
        first_learning_rates = {group["name"]: float(group["lr"]) for group in optimizer.param_groups}
        model.train()
        freeze_batchnorm(model)  # train() resets module modes, so repeat each epoch.
        loss_sum, sample_count, ignored_pixels, scored_pixels = 0.0, 0, 0, 0
        loss_normalizer = 0.0
        ce_weighted_sum, ce_normalizer, objective_sum = 0.0, 0.0, 0.0
        component_sums = {"ce_main": 0.0, "ce_aux": 0.0, "dice_main": 0.0, "dice_aux": 0.0, "lovasz_main": 0.0}
        batch_count = 0
        for batch_index, (image, target) in enumerate(train_loader, start=1):
            image, target = image.to(selected_device), target.to(selected_device)
            batch_ignored = int((target == IGNORE_INDEX).sum().item())
            batch_scored = target.numel() - batch_ignored
            if batch_scored == 0:
                raise ValueError("Training batch contains only ignored pixels; no supervision remains.")
            optimizer.zero_grad(set_to_none=True)
            outputs = model(image)
            ce_main = criterion(outputs["out"], target)
            use_auxiliary = auxiliary_loss_weight > 0 and "aux" in outputs
            ce_aux = criterion(outputs["aux"], target) if use_auxiliary else ce_main.new_zeros(())
            ce_combined = ce_main + auxiliary_loss_weight * ce_aux
            dice_main = foreground_generalized_dice_loss(outputs["out"], target) if use_dice else ce_main.new_zeros(())
            dice_aux = foreground_generalized_dice_loss(outputs["aux"], target) if use_dice and use_auxiliary else ce_main.new_zeros(())
            batch_loss = ce_combined + dice_weight * (dice_main + auxiliary_loss_weight * dice_aux) if use_dice else ce_combined
            lovasz_main = foreground_lovasz_softmax_loss(outputs["out"], target) if use_lovasz else ce_main.new_zeros(())
            if use_lovasz:
                batch_loss = batch_loss + lovasz_weight * lovasz_main
            if not torch.isfinite(batch_loss):
                raise RuntimeError("Nonfinite training loss; stopping without reporting a successful run.")
            batch_loss.backward()
            last_learning_rates = {group["name"]: float(group["lr"]) for group in optimizer.param_groups}
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            # Weighted CE divides by target weights, not the number of pixels.
            # Match that denominator when aggregating the reported epoch loss.
            batch_normalizer = float((weight_tensor[target.clamp_max(len(classes) - 1)] * (target != IGNORE_INDEX)).sum().item()) if weight_tensor is not None else batch_scored
            # Overlap losses are not additive pixel losses. Report the mean
            # actual optimizer-step objective for those combinations; preserve
            # the denominator-weighted epoch mean for CE-only callers.
            ce_weighted_sum += float(ce_combined.item()) * batch_normalizer
            ce_normalizer += batch_normalizer
            objective_sum += float(batch_loss.item())
            batch_count += 1
            for key, value in (("ce_main", ce_main), ("ce_aux", ce_aux), ("dice_main", dice_main), ("dice_aux", dice_aux), ("lovasz_main", lovasz_main)):
                component_sums[key] += float(value.item())
            loss_sum = objective_sum if use_overlap else ce_weighted_sum
            loss_normalizer = batch_count if use_overlap else ce_normalizer
            sample_count += image.shape[0]
            ignored_pixels += batch_ignored
            scored_pixels += batch_scored
            if batch_index == 1 or batch_index % 25 == 0 or batch_index == len(train_loader):
                ce_mean = (component_sums["ce_main"] + auxiliary_loss_weight * component_sums["ce_aux"]) / batch_count
                dice_mean = (component_sums["dice_main"] + auxiliary_loss_weight * component_sums["dice_aux"]) / batch_count
                suffix = f" ce_step_mean={ce_mean:.4f} dice_step_mean={dice_mean:.4f}" if use_dice else ""
                if use_lovasz:
                    suffix = f" ce_step_mean={ce_mean:.4f} lovasz_main_step_mean={component_sums['lovasz_main'] / batch_count:.4f}"
                print(f"epoch={epoch}/{epochs} batch={batch_index}/{len(train_loader)} train_loss={loss_sum / loss_normalizer:.4f}{suffix}", flush=True)
        validation = {
            **_evaluate_model(model, val_loader, classes, selected_device),
            "split": "val", "ignore_source_ids": ignore_source_ids,
            "metric_resolution": {"width": width, "height": height},
        }
        score = validation["foreground_macro_iou"]
        if score is None:
            raise ValueError("Validation has no scored foreground classes; cannot select a segmentation checkpoint.")
        row = {"epoch": epoch, "train_loss": loss_sum / loss_normalizer, "train_loss_normalizer": loss_normalizer, "train_scored_pixels": scored_pixels, "train_ignored_pixels": ignored_pixels, "validation": validation, "duration_seconds": time.monotonic() - started}
        row.update({
            "train_batches": batch_count,
            "learning_rates": {"first_update": first_learning_rates, "last_update": last_learning_rates},
            "train_loss_components": {
                "aggregation": "arithmetic_mean_over_optimizer_steps",
                "ce": (component_sums["ce_main"] + auxiliary_loss_weight * component_sums["ce_aux"]) / batch_count,
                "generalized_dice": (component_sums["dice_main"] + auxiliary_loss_weight * component_sums["dice_aux"]) / batch_count if use_dice else None,
                "lovasz": component_sums["lovasz_main"] / batch_count if use_lovasz else None,
                "per_head": {key: value / batch_count if (key.startswith("ce") or (key.startswith("dice") and use_dice) or (key.startswith("lovasz") and use_lovasz)) and not (key.endswith("_aux") and auxiliary_loss_weight == 0) else None for key, value in component_sums.items()},
                "auxiliary_head_weight": float(auxiliary_loss_weight), "auxiliary_loss_enabled": auxiliary_loss_weight > 0,
                "dice_weight": float(dice_weight) if use_dice else 0.0,
                "lovasz_weight": float(lovasz_weight) if use_lovasz else 0.0,
                "batch_objective": config["loss_details"]["batch_objective"],
            },
            "train_ce_denominator_weighted_mean": ce_weighted_sum / ce_normalizer,
            "train_ce_normalizer": ce_normalizer,
        })
        history.append(row)
        checkpoint = {
            "format_version": 1, "architecture": architecture, "model_id": model_id,
            "model_version": f"{version}-epoch-{epoch}", "run_id": version,
            "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "classes": classes, "input_size": {"width": width, "height": height},
            "ignore_index": IGNORE_INDEX, "ignore_source_ids": ignore_source_ids,
            "class_weighting": class_weighting, "class_weights": class_weights,
            "loss": loss_mode, "dice_weight": float(dice_weight) if use_dice else 0.0,
            "lovasz_weight": float(lovasz_weight) if use_lovasz else 0.0,
            "auxiliary_loss_weight": float(auxiliary_loss_weight),
            "backbone_checkpoint": backbone_initialization,
            "initialization": config["initialization"], "initialization_recipe": config["initialization_recipe"],
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
        if max_duration_seconds is not None and epoch < epochs and time.monotonic() - run_started >= max_duration_seconds:
            stop_reason = "max_duration_seconds"
            print(f"Stopping at epoch {epoch} after the duration limit; best and last checkpoints are saved.", flush=True)
            break
    return {
        "checkpoint": str((output_dir / "best.pt").resolve()),
        "last_checkpoint": str((output_dir / "last.pt").resolve()),
        "best_epoch": best_epoch, "best_foreground_macro_iou": best_score,
        "limited_run": config["limited_run"], "epochs_completed": len(history),
        "stop_reason": stop_reason, "duration_seconds": time.monotonic() - run_started,
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
