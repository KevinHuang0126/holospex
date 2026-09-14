"""A success claim requires intact evidence and the full native validation set."""

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location("audit_autonomous_results", Path(__file__).resolve().parents[1] / "cloud/audit_autonomous_results.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)

# Fixed truth counts from the established native validation report. Predictions
# below are synthetic perfect masks or an all-background failure, never models.
TRUTH_BY_CASE = [
    [858814, 562787, 37102, 11950, 0, 10105, 158922],
    [5505804, 1514131, 208424, 17200, 49066, 2189, 491666],
    [554015, 168730, 9123, 4307, 0, 0, 78274],
    [1009606, 464935, 37910, 9118, 0, 578, 117533],
    [3526400, 840528, 97697, 20973, 0, 809, 428232],
    [467997, 259327, 6367, 7189, 1032, 6068, 71860],
    [594777, 104619, 5923, 0, 7710, 3656, 103155],
    [6654984, 927780, 128454, 0, 36567, 15777, 434838],
    [479852, 231695, 13959, 6963, 7739, 2417, 77215],
    [2345525, 508471, 63639, 37132, 29701, 1111, 242982],
]


def mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def report(counts, ignored):
    """Small independent fixture calculator, including precision and recall."""
    rows = []
    for i, cls in enumerate(audit.ONTOLOGY):
        tp, support, predicted = counts[i][i], sum(counts[i]), sum(row[i] for row in counts)
        union = support + predicted - tp
        rows.append({**cls, "support_pixels": support, "predicted_pixels": predicted,
                     "true_positive_pixels": tp, "false_positive_pixels": predicted - tp,
                     "false_negative_pixels": support - tp, "iou": tp / union if union else None,
                     "dice": 2 * tp / (support + predicted) if support + predicted else None,
                     "precision": tp / predicted if predicted else None, "recall": tp / support if support else None})
    total = sum(map(sum, counts))
    value = {"confusion_matrix": counts, "per_class": rows, "scored_pixels": total,
             "ignored_pixels": ignored, "total_pixels": total + ignored,
             "pixel_accuracy": sum(counts[i][i] for i in range(7)) / total}
    add_aggregates(value)
    return value


def add_aggregates(value):
    for prefix, rows in (("foreground", value["per_class"][1:]), ("small_anatomy", value["per_class"][2:6])):
        for metric in ("iou", "dice"):
            value[f"{prefix}_macro_{metric}"] = mean([row[metric] for row in rows])
        value[f"{prefix}_classes_scored"] = sum(row["iou"] is not None for row in rows)
    value["small_anatomy_class_ids"] = [row["structureId"] for row in audit.ONTOLOGY[2:6]]


def fixture(perfect=True):
    videos, frames = [], []
    for case, support in zip(audit.VALIDATION_CASES, TRUTH_BY_CASE):
        counts = [[0] * 7 for _ in range(7)]
        for index, count in enumerate(support):
            counts[index][index if perfect else 0] = count
        ignored = len(audit.VALIDATION_FRAMES[case]) * 854 * 480 - sum(support)
        videos.append({**report(counts, ignored), "video_id": case, "sample_count": len(audit.VALIDATION_FRAMES[case])})
        for index, number in enumerate(audit.VALIDATION_FRAMES[case]):
            frame_ignore = ignored if index == 0 else 0
            frames.append({"video_id": case, "frame_number": number, "width": 854, "height": 480,
                           "scored_pixels": 854 * 480 - frame_ignore, "ignored_pixels": frame_ignore})
    counts = [[sum(video["confusion_matrix"][i][j] for video in videos) for j in range(7)] for i in range(7)]
    metrics = report(counts, sum(video["ignored_pixels"] for video in videos))
    equal = {"case_count": 10, "per_class": [{**cls, **{key: mean([v["per_class"][i][key] for v in videos])
                                                       for key in ("iou", "dice", "precision", "recall")}}
                                             for i, cls in enumerate(audit.ONTOLOGY)]}
    add_aggregates(equal)
    manifest = {"classes": audit.ONTOLOGY, "ignoreIndex": 255, "ignoreSourceIds": [255],
                "samples": [{"videoId": 4, "frameNumber": 1, "split": "train"},
                            {"videoId": 200, "frameNumber": 1, "split": "test"}] +
                           [{"videoId": case, "frameNumber": number, "split": "val"}
                            for case, number in sorted(audit.VALIDATION_IDENTITIES)]}
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    config = {"classes": audit.ONTOLOGY, "run_id": "fixture", "epochs": 1, "limited_run": False,
              "limit_train": None, "limit_val": None, "ignore_index": 255, "ignore_source_ids": [255],
              "train_video_ids": ["4"], "val_video_ids": audit.VALIDATION_CASES, "train_samples": 1,
              "val_samples": 75, "selection_metric": "foreground_macro_iou", "manifest_sha256": digest,
              "architecture": "deeplabv3_mobilenet_v3_large", "model_id": "fixture", "input_size": {"width": 672, "height": 384}}
    checkpoint = b"Opaque synthetic checkpoint; the auditor must never load it."
    metrics.update(per_video=videos, frames=frames, case_equal=equal, sample_count=75, case_count=10,
                   model_version="fixture-epoch-1", limited_evaluation=False, limited_training=False,
                   split="val", report_type="original_resolution_segmentation_evaluation", ignore_index=255,
                   ignore_source_ids=[255], architecture=config["architecture"], model_id=config["model_id"],
                   input_size=config["input_size"], checkpoint_sha256=hashlib.sha256(checkpoint).hexdigest(),
                   evaluation_video_ids=audit.VALIDATION_CASES, training_manifest_sha256=digest,
                   evaluation_manifest_sha256=digest,
                   metric_resolution={"basis": "original_mask_pixels", "sizes": [{"width": 854, "height": 480}]},
                   metric_policy={"hud_filters_applied": False, "prediction": "bilinear resize logits to original mask shape before argmax; align_corners=False"})
    return {"train/best.pt": checkpoint, "train/config.json": config, "manifest.json": manifest,
            "train/history.json": [{"epoch": 1, "validation": {"foreground_macro_iou": metrics["foreground_macro_iou"]}}],
            "train/metrics-val-original.json": metrics}


def write_run(root, files, completion_updates=None):
    completion = {"state": "completed", "run_name": "fixture", "epochs_completed": 1, "epochs_requested": 1, "artifacts": {}}
    for name, value in files.items():
        payload = value if isinstance(value, bytes) else json.dumps(value).encode()
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        completion["artifacts"][name] = {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
    completion.update(completion_updates or {})
    payload = json.dumps(completion).encode()
    (root / "cloud-completion.json").write_bytes(payload)
    (root / "cloud-collection.json").write_text(json.dumps({"completion_sha256": hashlib.sha256(payload).hexdigest(),
        "completion_bytes": len(payload), "artifact_count": len(files), "metadata_paths_rewritten": False}))


class AutonomousAuditTests(unittest.TestCase):
    def test_verified_duration_stop_is_eligible_but_unexplained_truncation_fails(self):
        updates = {"epochs_requested": 40, "epochs_completed": 1, "budget_exhausted": True,
                   "stop_reason": "max_duration_seconds", "training_duration_seconds": 61.5}
        for invalid in (None, {"budget_exhausted": False}, {"stop_reason": "epochs_completed"},
                        {"training_duration_seconds": 59.0}, {"training_duration_seconds": None},
                        {"epochs_completed": 2}):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as temporary:
                files = fixture()
                files["train/config.json"].update(epochs=40, max_duration_seconds=60.0)
                root = Path(temporary)
                write_run(root, files, {**updates, **(invalid or {})})
                if invalid:
                    with self.assertRaises(ValueError):
                        audit.summarize_run(root)
                else:
                    result = audit.summarize_run(root)
                    self.assertTrue(result["target_reached"])
                    self.assertTrue(result["duration_limited"])
                    self.assertEqual(result["epochs_completed"], 1)
                    self.assertEqual(result["epochs_requested"], 40)

    def test_verified_success_and_background_only_failure(self):
        for perfect in (True, False):
            with self.subTest(perfect=perfect), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                write_run(root, fixture(perfect))
                before = {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}
                result = audit.summarize_run(root)
                self.assertEqual(result["target_reached"], perfect)
                self.assertEqual(result["foreground_macro_iou"], float(perfect))
                self.assertEqual(result["small_pooled_iou"], float(perfect))
                self.assertEqual(result["small_case_equal_iou"], float(perfect))
                self.assertFalse(result["audit"]["model_loaded"])
                self.assertEqual(before, {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()})

    def test_checkpoint_and_metric_corruption_fail_hash_verification(self):
        for name in ("train/best.pt", "train/metrics-val-original.json", "cloud-completion.json"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                write_run(root, fixture())
                with (root / name).open("ab") as stream:
                    stream.write(b" ")
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    audit.summarize_run(root)

    def test_consistently_rehashed_invalid_evidence_is_rejected(self):
        mutations = {
            "limited": lambda f: f["train/metrics-val-original.json"].update(limited_evaluation=True),
            "test_split": lambda f: f["train/metrics-val-original.json"].update(split="test"),
            "resized_grid": lambda f: f["train/metrics-val-original.json"]["metric_resolution"].update(basis="input_pixels"),
            "wrong_frame": lambda f: f["train/metrics-val-original.json"]["frames"][0].update(frame_number=999999),
            "duplicate_frame": lambda f: f["train/metrics-val-original.json"]["frames"].__setitem__(1, deepcopy(f["train/metrics-val-original.json"]["frames"][0])),
            "inflated_iou": lambda f: f["train/metrics-val-original.json"].update(foreground_macro_iou=0.99),
            "wrong_class_iou": lambda f: f["train/metrics-val-original.json"]["per_class"][3].update(iou=0.9),
            "wrong_case_average": lambda f: f["train/metrics-val-original.json"]["case_equal"].update(small_anatomy_macro_iou=0.9),
            "wrong_checkpoint": lambda f: f["train/metrics-val-original.json"].update(checkpoint_sha256="0" * 64),
            "wrong_epoch": lambda f: f["train/metrics-val-original.json"].update(model_version="fixture-epoch-2"),
            "partial_history": lambda f: f["train/config.json"].update(epochs=2),
            "case_leakage": lambda f: f["manifest.json"]["samples"][0].update(videoId=126),
            "nan": lambda f: f["train/metrics-val-original.json"].update(foreground_macro_iou=float("nan")),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                files = fixture()
                mutate(files)
                # Preserve cryptographic consistency so mutations reach the
                # semantic audit instead of stopping at a stale digest.
                digest = hashlib.sha256(json.dumps(files["manifest.json"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                files["train/config.json"]["manifest_sha256"] = digest
                files["train/metrics-val-original.json"]["training_manifest_sha256"] = digest
                files["train/metrics-val-original.json"]["evaluation_manifest_sha256"] = digest
                root = Path(temporary)
                write_run(root, files)
                with self.assertRaises(ValueError):
                    audit.summarize_run(root)


if __name__ == "__main__":
    unittest.main()
