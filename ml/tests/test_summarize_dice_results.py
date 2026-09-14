"""Small synthetic six-run fixtures exercise audit failure and paired summaries."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location("summarize_dice_results", Path(__file__).resolve().parents[1] / "cloud/summarize_dice_results.py")
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


class DiceSummaryTests(unittest.TestCase):
    def fixture(self, root):
        classes = [{"index": i, "sourceId": i, "structureId": value} for i, value in enumerate(["background"] + summary.FOREGROUND)]
        manifest = {"classes": classes, "samples": [{"split": split} for split, count in (("train", 343), ("val", 75), ("test", 74)) for _ in range(count)]}
        bundles = {"source": {"sha256": "a" * 64}, "data": {"sha256": "b" * 64}}
        launch = {"bundles": bundles, "experiments": []}
        for name, seed, objective in summary.RUNS:
            directory = root / name
            (directory / "train").mkdir(parents=True)
            use_dice = objective != "ce"
            value = 0.2 + (seed - 42) * 0.02 + (0.05 if use_dice else 0)
            config = {**deepcopy(summary.FROZEN), "seed": seed, "run_id": f"run-{name}", "loss": objective,
                      "dice_weight": 1.0 if use_dice else 0.0, "classes": classes,
                      "manifest_sha256": summary.canonical_digest(manifest), "software_versions": {"torch": "2.8.0+cu126"},
                      "train_video_ids": [str(i) for i in range(30)], "val_video_ids": [str(i) for i in range(30, 40)],
                      "training_loss_reduction": "mean_over_optimizer_steps" if use_dice else "mean_over_target_class_weights",
                      "loss_details": {"main_head_weight": 1., "auxiliary_head_weight": .4,
                                       "dice": {"epsilon": 1e-6, "variant": "foreground_present_classes_batch_pooled_generalized_soft_dice"} if use_dice else None,
                                       "head_objective": "combo" if use_dice else "CE", "epoch_total": "batch" if use_dice else "weighted"}}
            checkpoint = directory / "train/best.pt"
            checkpoint.write_bytes(f"synthetic-non-model-{name}".encode())
            per_class = [{**entry, "iou": value, "dice": value + .1, "precision": value + .2, "recall": value + .3} for entry in classes]
            metrics = {"split": "val", "sample_count": 75, "case_count": 10, "limited_evaluation": False,
                       "limited_training": False, "input_size": summary.FROZEN["input_size"], "ignore_index": 255, "ignore_source_ids": [255],
                       "device": "cuda", "architecture": summary.FROZEN["architecture"],
                       "metric_resolution": {"basis": "original_mask_pixels", "sizes": [{"width": 854, "height": 480}]},
                       "training_manifest_sha256": config["manifest_sha256"], "evaluation_manifest_sha256": config["manifest_sha256"],
                       "model_version": config["run_id"] + "-epoch-3", "checkpoint_sha256": summary.digest(checkpoint),
                       "frames": [{"frame": i} for i in range(75)], "evaluation_video_ids": config["val_video_ids"], "metric_policy": {"test": "shared"},
                       "per_class": per_class, "small_anatomy_macro_iou": value, "foreground_macro_iou": value, "foreground_macro_dice": value + .1,
                       "case_equal": {"small_anatomy_macro_iou": value - .05,
                                      "per_class": [{"structureId": entry["structureId"], "iou": value - .05} for entry in classes]}}
            files = {"train/config.json": config, "train/metrics-val-original.json": metrics, "manifest.json": manifest,
                     "environment.json": {"torch": "2.8.0+cu126"},
                     "train/history.json": [{"epoch": epoch, "validation": {"foreground_macro_iou": .8 if epoch in (3, 4) else .5}} for epoch in range(1, 41)]}
            for logical, contents in files.items():
                (directory / logical).write_text(json.dumps(contents))
            done = {"state": "completed", "run_name": name, "attempt": name, "epochs_completed": 40, "epochs_requested": 40,
                    "artifacts": {logical: {"bytes": (directory / logical).stat().st_size, "sha256": summary.digest(directory / logical)}
                                  for logical in list(files) + ["train/best.pt"]}}
            (directory / "cloud-completion.json").write_text(json.dumps(done))
            self.receipt(directory)
            env = {"HOLOSPEX_SOURCE_SHA256": "a" * 64, "HOLOSPEX_DATA_SHA256": "b" * 64,
                   "HOLOSPEX_LOSS": objective, "HOLOSPEX_SEED": str(seed), "HOLOSPEX_EPOCHS": "40", "HOLOSPEX_DICE_WEIGHT": "1.0", "HOLOSPEX_RUN_NAME": name}
            job = root / f"{name}.job.json"
            job.write_text(json.dumps({"workerPoolSpecs": [{"machineSpec": {"machineType": "a2-highgpu-1g", "acceleratorType": "NVIDIA_TESLA_A100", "acceleratorCount": 1},
                                                           "containerSpec": {"imageUri": "image@sha256:digest", "env": [{"name": k, "value": v} for k, v in env.items()]}}]}))
            launch["experiments"].append({"name": name, "config": str(job), "job_name": f"jobs/{name}"})
        launch_path = root / "launch.json"
        launch_path.write_text(json.dumps(launch))
        return launch_path

    def receipt(self, directory):
        (directory / "cloud-collection.json").write_text(json.dumps({"completion_sha256": summary.digest(directory / "cloud-completion.json"), "metadata_paths_rewritten": False}))

    def mutate_artifact(self, directory, logical, mutate):
        path = directory / logical
        value = json.loads(path.read_text())
        mutate(value)
        path.write_text(json.dumps(value))
        done = json.loads((directory / "cloud-completion.json").read_text())
        done["artifacts"][logical] = {"bytes": path.stat().st_size, "sha256": summary.digest(path)}
        (directory / "cloud-completion.json").write_text(json.dumps(done))
        self.receipt(directory)

    def test_all_seeds_reported_with_correct_fraction_means_and_percentage_point_deltas(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = summary.summarize(root, self.fixture(root))
            self.assertEqual(len(report["runs"]), 6)
            self.assertEqual([r["selected_epoch"] for r in report["runs"]], [3] * 6)  # First tied best.
            ce = report["group_summary_fraction_units"]["ce"]["small_pooled_iou"]
            self.assertAlmostEqual(ce["mean"], .22)
            self.assertEqual(ce["n"], 3)
            self.assertAlmostEqual(ce["min"], .2)
            self.assertAlmostEqual(ce["max"], .24)
            for pair in report["paired_seeds"]:
                self.assertAlmostEqual(pair["dice_minus_ce_percentage_points"]["small_pooled_iou"], 5.0)
            self.assertEqual(set(report["runs"][0]["per_class"]["cystic_artery"]), {"iou", "dice", "precision", "recall"})

    def test_tampered_artifact_fails_before_reporting_scores(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            launch = self.fixture(root)
            (root / summary.RUNS[0][0] / "train/best.pt").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "hash/size mismatch"):
                summary.summarize(root, launch)

    def test_incomplete_set_or_attempt_cannot_be_averaged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            launch = self.fixture(root)
            path = root / summary.RUNS[-1][0] / "cloud-completion.json"
            done = json.loads(path.read_text()); done["epochs_completed"] = 39
            path.write_text(json.dumps(done))
            with self.assertRaisesRegex(ValueError, "incomplete epoch"):
                summary.summarize(root, launch)
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                summary.summarize(root, launch)

    def test_changed_frozen_setting_or_runtime_is_not_silently_pooled(self):
        for key, value, expected in (("input_size", {"width": 896, "height": 512}, "frozen setting"),
                                     ("software_versions", {"torch": "different"}, "settings, runtime")):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary); launch = self.fixture(root)
                self.mutate_artifact(root / summary.RUNS[1][0], "train/config.json", lambda x: x.update({key: value}))
                with self.assertRaisesRegex(ValueError, expected):
                    summary.summarize(root, launch)

    def test_changed_frame_order_or_test_split_is_rejected(self):
        for mutation, expected in ((lambda x: x.update({"split": "test"}), "evaluation split"),
                                   (lambda x: x["frames"].reverse(), "ordered frames")):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary); launch = self.fixture(root)
                self.mutate_artifact(root / summary.RUNS[1][0], "train/metrics-val-original.json", mutation)
                with self.assertRaisesRegex(ValueError, expected):
                    summary.summarize(root, launch)

    def test_wrong_submitted_bundle_or_duplicate_launch_run_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); launch = self.fixture(root)
            data = json.loads(launch.read_text()); data["bundles"]["source"]["sha256"] = "c" * 64
            launch.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "submitted HOLOSPEX_SOURCE_SHA256"):
                summary.summarize(root, launch)
            data["experiments"][-1] = data["experiments"][0]
            launch.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "exactly the frozen six"):
                summary.summarize(root, launch)


if __name__ == "__main__":
    unittest.main()
