"""Synthetic complete runs test integrity gates and paired experiment reporting."""

from collections import Counter
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location("summarize_reviewed_results", Path(__file__).resolve().parents[1] / "cloud/summarize_reviewed_results.py")
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


class ReviewedSummaryTests(unittest.TestCase):
    def write(self, path, value):
        path.write_text(json.dumps(value, allow_nan=False))

    def recompute_fixture(self, metrics):
        # Fixture construction is separate from summary.confusion_scores so a
        # transpose/denominator error there cannot manufacture passing inputs.
        for report in [*metrics["per_video"], metrics]:
            if report is metrics:
                report["confusion_matrix"] = [[sum(video["confusion_matrix"][i][j] for video in metrics["per_video"])
                                               for j in range(7)] for i in range(7)]
            matrix = report["confusion_matrix"]
            records = []
            for index, entry in enumerate(summary.ONTOLOGY):
                tp = matrix[index][index]
                fn = sum(matrix[index][j] for j in range(7) if j != index)
                fp = sum(matrix[i][index] for i in range(7) if i != index)
                records.append({**entry, "iou": tp / (tp + fp + fn) if tp + fp + fn else None,
                                "dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
                                "precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn) if tp + fn else None,
                                "support_pixels": tp + fn, "predicted_pixels": tp + fp, "true_positive_pixels": tp,
                                "false_positive_pixels": fp, "false_negative_pixels": fn})
            report["per_class"] = records
            report["scored_pixels"] = sum(sum(row) for row in matrix)
            report["ignored_pixels"] = 0
            report["total_pixels"] = report["scored_pixels"]
        equal = {"case_count": 10, "per_class": [
            {**entry, **{key: summary.mean_defined([video["per_class"][index][key] for video in metrics["per_video"]])
                         for key in summary.CLASS_METRICS}} for index, entry in enumerate(summary.ONTOLOGY)]}
        metrics["case_equal"] = equal
        for report in [*metrics["per_video"], metrics, equal]:
            records = report["per_class"]
            for prefix, selected in (("foreground", records[1:]), ("small_anatomy", records[2:6])):
                for key in ("iou", "dice"):
                    report[f"{prefix}_macro_{key}"] = summary.mean_defined([row[key] for row in selected])

    def fixture(self, root):
        base = [{"split": split, "imagePath": f"/base/{split}_seg/{offset + i % cases}_{1000 + i}.jpg",
                 "maskPath": f"/base/semseg/{offset + i % cases}_{1000 + i}.png", "videoId": offset + i % cases,
                 "frameNumber": 1000 + i, "timestampMs": (1000 + i) * 40.0}
                for split, count, cases, offset in (("train", 343, 30, 0), ("val", 75, 10, 100), ("test", 74, 10, 200)) for i in range(count)]
        additions = [{"split": "train", "imagePath": f"/prepared/images/{300 + i}_2000.jpg", "maskPath": f"/prepared/masks/{300 + i}_2000.png",
                      "videoId": 300 + i, "frameNumber": 2000, "timestampMs": 80000.0, "annotationSource": "reviewed_partial_anatomy",
                      "reviewProvenancePath": f"/prepared/provenance/{300 + i}_2000.json"} for i in range(18)]
        bundles = {"source": {"sha256": "a" * 64}, "data": {"sha256": "b" * 64},
                   "prepared": {"sha256": "c" * 64, "uri": "gs://bucket/prepared.tar.gz"}}
        launch = {"bundles": bundles, "experiments": []}
        for name, seed, arm in summary.RUNS:
            directory = root / name
            (directory / "train").mkdir(parents=True)
            count, epochs = summary.ARM[arm]["train_samples"], summary.ARM[arm]["epochs"]
            samples = deepcopy(base + (additions if arm == "added" else []))
            # Different cloud prefixes are permissible; original folder/file identities are retained.
            for sample in samples:
                for key in ("imagePath", "maskPath"):
                    sample[key] = sample[key].replace("/base/", f"/work/{name}/data/")
            manifest = {"classes": summary.ONTOLOGY, "samples": samples, "ignoreIndex": 255, "ignoreSourceIds": [255]}
            cases = {split: sorted({str(x["videoId"]) for x in samples if x["split"] == split}) for split in ("train", "val", "test")}
            value = 0.2 + (seed - 42) * 0.02 + (0.05 if arm == "added" else 0)
            counts = [1000] * 7
            counts[0] += 100 if arm == "added" else 0
            weights = [0.5 if arm == "added" else 0.6] * 7
            config = {**deepcopy(summary.FROZEN), "epochs": epochs, "seed": seed, "run_id": f"run-{name}",
                      "train_samples": count, "classes": summary.ONTOLOGY,
                      "manifest_sha256": summary.canonical_digest(manifest), "software_versions": {"torch": "2.8.0+cu126"},
                      "train_video_ids": cases["train"], "val_video_ids": cases["val"], "class_weights": weights,
                      "dataset": {"root": f"/work/{name}", "report": {"train": count}},
                      "class_weighting_details": {"sample_count": count, "counts": counts, "scored_pixels": sum(counts),
                          "ignored_pixels": count * 672 * 384 - sum(counts), "frequencies": [0.1] * 7,
                          "raw_inverse_sqrt_weights": [1.0] * 7, "foreground_mean_raw_weight": 1.0, "weights": weights,
                          "scope": "selected_train_samples_after_exact_resize_and_ignore_handling", "formula": "frozen_formula", "clip_min": 0.1, "clip_max": 5.0},
                      "sampling_details": {"draws_per_epoch": count, "video_frame_counts": dict(sorted(Counter(str(x["videoId"]) for x in samples if x["split"] == "train").items())),
                                           "replacement": False, "weight_formula": "uniform_shuffle_without_replacement"},
                      "loss_details": {"main_head_weight": 1.0, "auxiliary_head_weight": 0.4, "dice": None,
                                       "head_objective": "CE", "ce_reduction": "mean_over_target_class_weights"}}
            checkpoint = directory / "train/best.pt"
            checkpoint.write_bytes(f"synthetic-non-model-{name}".encode())
            per_class = [{**entry, "iou": value, "dice": value + .1, "precision": value + .2, "recall": value + .3} for entry in summary.ONTOLOGY]
            frames = [{"video_id": str(x["videoId"]), "frame_number": x["frameNumber"], "timestamp_ms": x["timestampMs"],
                       "image_path": x["imagePath"], "mask_path": x["maskPath"], "width": 854, "height": 480,
                       "scored_pixels": 409920, "ignored_pixels": 0} for x in samples if x["split"] == "val"]
            metrics = {"split": "val", "sample_count": 75, "case_count": 10, "limited_evaluation": False,
                       "limited_training": False, "input_size": summary.FROZEN["input_size"], "ignore_index": 255, "ignore_source_ids": [255],
                       "device": "cuda", "architecture": summary.FROZEN["architecture"],
                       "metric_resolution": {"basis": "original_mask_pixels", "sizes": [{"width": 854, "height": 480}]},
                       "training_manifest_sha256": config["manifest_sha256"], "evaluation_manifest_sha256": config["manifest_sha256"],
                       "model_version": config["run_id"] + "-epoch-3", "checkpoint_sha256": summary.digest(checkpoint),
                       "frames": frames, "evaluation_video_ids": cases["val"], "metric_policy": {"comparison": "shared"},
                       "per_class": per_class, "small_anatomy_macro_iou": value, "foreground_macro_iou": value, "foreground_macro_dice": value + .1,
                       "case_equal": {"small_anatomy_macro_iou": value - .05,
                                      "per_class": [{**entry, "iou": value - .05} for entry in per_class]}}
            correct = int(2 * 58560 * value / (1 + value))
            per_frame = [[correct if i == j else 58560 - correct if j == (i + 1) % 7 else 0 for j in range(7)] for i in range(7)]
            metrics["per_video"] = []
            for video_id in cases["val"]:
                frame_count = sum(frame["video_id"] == video_id for frame in frames)
                metrics["per_video"].append({"video_id": video_id, "sample_count": frame_count,
                                             "confusion_matrix": [[n * frame_count for n in row] for row in per_frame]})
            self.recompute_fixture(metrics)
            files = {"train/config.json": config, "train/metrics-val-original.json": metrics, "manifest.json": manifest,
                     "environment.json": {"torch": "2.8.0+cu126"},
                     "train/history.json": [{"epoch": epoch, "validation": {"foreground_macro_iou": .8 if epoch in (3, 4) else .5},
                                             "train_batches": (count + 1) // 2, "train_scored_pixels": sum(counts),
                                             "train_ignored_pixels": count * 672 * 384 - sum(counts), "duration_seconds": 10.0}
                                            for epoch in range(1, epochs + 1)],
                     "commands.json": [["evaluate", ["python", "-m", "holospex_ml", "evaluate-original", "--split", "val"], "evaluate.log"]]}
            for logical, contents in files.items():
                self.write(directory / logical, contents)
            audited = {split: [{**{k: v for k, v in x.items() if k not in summary.PATH_FIELDS},
                               "imageSha256": summary.canonical_digest([x["videoId"], x["frameNumber"], "image"]),
                               "maskSha256": summary.canonical_digest([x["videoId"], x["frameNumber"], "mask"])}
                              for x in samples if x["split"] == split] for split in ("train", "val", "test")}
            self.write(directory / "training-input-audit.json", {
                "artifactType": "training_input_audit", "manifestSha256": summary.digest(directory / "manifest.json"),
                "splitCounts": {"train": count, "val": 75, "test": 74}, "splitCaseIds": cases,
                "splitContentSha256": {split: summary.record_digest(records) for split, records in audited.items()},
                "classes": summary.ONTOLOGY, "ignoreIndex": 255, "ignoreSourceIds": [255], "samples": audited})
            if arm == "added":
                source_manifest = {**manifest, "samples": deepcopy(base + additions)}
                base_manifest = {**manifest, "samples": deepcopy(base)}
                self.write(directory / "manifest-source.json", source_manifest)
                self.write(directory / "base-manifest-source.json", base_manifest)
                package_files = [{"sourcePath": sample[f"{prefix}Path"],
                                  "sha256": summary.canonical_digest([sample["videoId"], sample["frameNumber"], prefix])}
                                 for sample in source_manifest["samples"] for prefix in ("image", "mask")]
                self.write(directory / "prepared-package.json", {
                    "manifestSha256": summary.digest(directory / "manifest-source.json"),
                    "baseManifestSha256": summary.digest(directory / "base-manifest-source.json"), "files": package_files})
                bundles["prepared"]["package_sha256"] = summary.digest(directory / "prepared-package.json")
                bundles["prepared"]["prepared_manifest_sha256"] = summary.digest(directory / "manifest-source.json")
                self.write(directory / "prepared-input-verification.json", {
                    "verified": True, "originalSamplesPreserved": True, "heldOutSamplesPreserved": True, "addedTrainImageCount": 18,
                    "verifiedFileCount": len(package_files),
                    "counts": {"train": 361, "val": 75, "test": 74}, "packageSha256": summary.digest(directory / "prepared-package.json"),
                    "sourceManifestSha256": summary.digest(directory / "manifest-source.json"),
                    "baseManifestSha256": summary.digest(directory / "base-manifest-source.json")})
            output_uri = f"gs://bucket/runs/{name}"
            done = {"state": "completed", "run_name": name, "attempt": name, "attempt_uri": output_uri + "/attempts/one",
                    "epochs_completed": epochs, "epochs_requested": epochs,
                    "artifacts": {str(path.relative_to(directory)): {"bytes": path.stat().st_size, "sha256": summary.digest(path)}
                                  for path in directory.rglob("*") if path.is_file()}}
            self.write(directory / "cloud-completion.json", done)
            self.receipt(directory)
            env = {"HOLOSPEX_SOURCE_SHA256": "a" * 64, "HOLOSPEX_DATA_SHA256": "b" * 64,
                   "HOLOSPEX_LOSS": "ce", "HOLOSPEX_SEED": str(seed), "HOLOSPEX_EPOCHS": str(epochs), "HOLOSPEX_RUN_NAME": name}
            if arm == "added":
                env.update(HOLOSPEX_PREPARED_SHA256="c" * 64, HOLOSPEX_PREPARED_URI=bundles["prepared"]["uri"])
            job = root / f"{name}.job.json"
            self.write(job, {"workerPoolSpecs": [{"machineSpec": {"machineType": "a2-highgpu-1g", "acceleratorType": "NVIDIA_TESLA_A100", "acceleratorCount": 1},
                                                "containerSpec": {"imageUri": "image@sha256:digest", "env": [{"name": k, "value": v} for k, v in env.items()]}}]})
            launch["experiments"].append({"name": name, "arm": arm, "seed": seed, "epochs": epochs, "config": str(job),
                                          "config_sha256": summary.digest(job), "job_name": f"jobs/{name}", "output_uri": output_uri})
        launch_path = root / "launch.json"
        self.write(launch_path, launch)
        return launch_path

    def receipt(self, directory):
        self.write(directory / "cloud-collection.json", {"completion_sha256": summary.digest(directory / "cloud-completion.json"), "metadata_paths_rewritten": False})

    def mutate_artifact(self, directory, logical, mutate):
        path = directory / logical
        value = summary.read_json(path)
        mutate(value)
        self.write(path, value)
        done = summary.read_json(directory / "cloud-completion.json")
        done["artifacts"][logical] = {"bytes": path.stat().st_size, "sha256": summary.digest(path)}
        self.write(directory / "cloud-completion.json", done)
        self.receipt(directory)

    def test_complete_paired_seeds_allow_data_statistics_and_path_prefix_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = summary.summarize(root, self.fixture(root))
            self.assertEqual(len(result["runs"]), 6)
            self.assertEqual([r["selected_epoch"] for r in result["runs"]], [3] * 6)  # First tied best.
            self.assertEqual([r["epochs"] for r in result["runs"]], [42, 40] * 3)
            self.assertEqual([r["planned_optimizer_steps"] for r in result["runs"]], [7224, 7240] * 3)
            self.assertAlmostEqual(result["group_summary_fraction_units"]["base"]["small_pooled_iou"]["mean"], .22, places=4)
            for pair in result["paired_seeds"]:
                self.assertAlmostEqual(pair["added_minus_base_percentage_points"]["small_pooled_iou"], 5.0, places=2)
            self.assertGreater(result["per_class_paired_delta_summary_percentage_points"]["pooled"]["cystic_artery"]["recall"]["mean"], 0)
            self.assertTrue(result["audit"]["original_sample_image_and_mask_hashes_match"])
            self.assertTrue(result["practical_pilot_criterion"]["met"])

    def test_tampered_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); launch = self.fixture(root)
            (root / summary.RUNS[0][0] / "train/best.pt").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "hash/size mismatch"):
                summary.summarize(root, launch)

    def test_incomplete_or_old_epoch_budget_cannot_be_averaged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); launch = self.fixture(root)
            directory = root / summary.RUNS[0][0]
            done = summary.read_json(directory / "cloud-completion.json"); done["epochs_completed"] = 40
            self.write(directory / "cloud-completion.json", done); self.receipt(directory)
            with self.assertRaisesRegex(ValueError, "epoch budget"):
                summary.summarize(root, launch)

    def test_runtime_and_frozen_objective_changes_fail(self):
        for key, value, pattern in (("software_versions", {"torch": "different"}, "shared settings"),
                                    ("input_size", {"width": 896, "height": 512}, "frozen setting"),
                                    ("loss", "ce_generalized_dice", "frozen setting")):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary); launch = self.fixture(root)
                self.mutate_artifact(root / summary.RUNS[1][0], "train/config.json", lambda x: x.update({key: value}))
                with self.assertRaisesRegex(ValueError, pattern):
                    summary.summarize(root, launch)

    def test_test_split_commands_and_wrong_frame_order_fail(self):
        for logical, mutation, pattern in (("train/metrics-val-original.json", lambda x: x.update(split="test"), "evaluation split"),
                                            ("train/metrics-val-original.json", lambda x: x["frames"].reverse(), "ordered validation"),
                                            ("commands.json", lambda x: x[0][1].__setitem__(-1, "test"), "unexpected evaluation command")):
            with self.subTest(logical=logical), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary); launch = self.fixture(root)
                self.mutate_artifact(root / summary.RUNS[1][0], logical, mutation)
                with self.assertRaisesRegex(ValueError, pattern):
                    summary.summarize(root, launch)

    def test_changed_heldout_pixels_fail_even_when_artifact_and_fingerprint_match(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); launch = self.fixture(root)
            def mutation(x):
                x["samples"]["test"][0]["maskSha256"] = "f" * 64
                x["splitContentSha256"]["test"] = summary.record_digest(x["samples"]["test"])
            for name, _, arm in summary.RUNS:
                if arm == "added":
                    self.mutate_artifact(root / name, "training-input-audit.json", mutation)
            with self.assertRaisesRegex(ValueError, "training mask hash differs from frozen prepared package"):
                summary.summarize(root, launch)

    def test_consistent_new_mask_change_cannot_replace_the_reviewed_package(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); launch = self.fixture(root)
            def mutation(x):
                # Repair every collected integrity layer and keep all three
                # added seeds consistent. Only the frozen package detects it.
                self.assertEqual(x["samples"]["train"][-1]["annotationSource"], "reviewed_partial_anatomy")
                x["samples"]["train"][-1]["maskSha256"] = "f" * 64
                x["splitContentSha256"]["train"] = summary.record_digest(x["samples"]["train"])
            for name, _, arm in summary.RUNS:
                if arm == "added":
                    self.mutate_artifact(root / name, "training-input-audit.json", mutation)
            with self.assertRaisesRegex(ValueError, "training mask hash differs from frozen prepared package"):
                summary.summarize(root, launch)

    def test_actual_epoch_pixel_and_optimizer_budget_must_match_weighting_inputs(self):
        for key in ("train_scored_pixels", "train_ignored_pixels", "train_batches"):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary); launch = self.fixture(root)
                self.mutate_artifact(root / summary.RUNS[1][0], "train/history.json", lambda x: x[0].update({key: x[0][key] + 1}))
                with self.assertRaisesRegex(ValueError, "actual training steps/pixel budget differs"):
                    summary.summarize(root, launch)

    def test_class_weighting_policy_changes_are_not_allowed_as_data_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); launch = self.fixture(root)
            self.mutate_artifact(root / summary.RUNS[1][0], "train/config.json", lambda x: x["class_weighting_details"].update(formula="different"))
            with self.assertRaisesRegex(ValueError, "shared settings"):
                summary.summarize(root, launch)

    def test_undefined_precision_does_not_silently_average_two_seeds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); launch = self.fixture(root)
            def remove_artery_predictions(metrics):
                for video in metrics["per_video"]:
                    for row in video["confusion_matrix"]:
                        row[0] += row[3]
                        row[3] = 0
                self.recompute_fixture(metrics)
            self.mutate_artifact(root / summary.RUNS[1][0], "train/metrics-val-original.json", remove_artery_predictions)
            result = summary.summarize(root, launch)
            aggregate = result["per_class_group_summary_fraction_units"]["added"]["pooled"]["cystic_artery"]["precision"]
            self.assertEqual(aggregate, {"mean": None, "min": None, "max": None, "n": 3, "defined_n": 2})

    def test_plausible_reported_precision_and_confusion_mismatches_fail(self):
        for mutation in (lambda x: x["per_class"][3].update(precision=.9),
                         lambda x: x["confusion_matrix"][0].__setitem__(0, x["confusion_matrix"][0][0] + 1),
                         lambda x: x["case_equal"]["per_class"][3].update(recall=.9)):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary); launch = self.fixture(root)
                self.mutate_artifact(root / summary.RUNS[1][0], "train/metrics-val-original.json", mutation)
                with self.assertRaisesRegex(ValueError, "confusion"):
                    summary.summarize(root, launch)

    def test_wrong_bundle_duplicate_run_or_arm_is_rejected(self):
        for mutation, pattern in ((lambda x: x["bundles"]["prepared"].update(sha256="f" * 64), "submitted HOLOSPEX_PREPARED_SHA256"),
                                  (lambda x: x["experiments"].__setitem__(-1, x["experiments"][0]), "exactly the frozen six"),
                                  (lambda x: x["experiments"][0].update(arm="added"), "launch seed/arm")):
            with self.subTest(pattern=pattern), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary); launch = self.fixture(root)
                value = summary.read_json(launch); mutation(value); self.write(launch, value)
                with self.assertRaisesRegex(ValueError, pattern):
                    summary.summarize(root, launch)

    def test_nonfinite_and_duplicate_json_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "invalid.json"
            for contents in ('{"x":NaN}', '{"x":1,"x":2}'):
                path.write_text(contents)
                with self.assertRaises(ValueError):
                    summary.read_json(path)


if __name__ == "__main__":
    unittest.main()
