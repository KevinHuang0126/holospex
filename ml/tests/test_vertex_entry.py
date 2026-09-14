"""Exercise durable cloud artifact handling without cloud or GPU dependencies."""

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

MODULE_PATH = Path(__file__).resolve().parents[1] / "cloud/vertex_entry.py"
SPEC = importlib.util.spec_from_file_location("vertex_entry", MODULE_PATH)
entry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(entry)


class PreconditionFailed(Exception):
    code = 412


class FakeBlob:
    def __init__(self, bucket, name):
        self.bucket, self.name = bucket, name
        self.metadata = {}

    def upload_from_string(self, data, *, if_generation_match, content_type=None):
        assert if_generation_match == 0
        if self.name in self.bucket.objects:
            raise PreconditionFailed()
        self.bucket.objects[self.name] = (bytes(data), dict(self.metadata))

    def upload_from_filename(self, path, *, if_generation_match):
        self.upload_from_string(Path(path).read_bytes(), if_generation_match=if_generation_match)


class FakeBucket:
    name = "test-bucket"

    def __init__(self):
        self.objects = {}

    def blob(self, name):
        return FakeBlob(self, name)

    def get_blob(self, name):
        if name not in self.objects:
            return None
        data, metadata = self.objects[name]
        return argparse.Namespace(size=len(data), metadata=metadata)


def make_epoch(root, epochs=1, best_epoch=1):
    train = root / "train"
    train.mkdir(exist_ok=True)
    config = {"manifest_sha256": "manifest", "seed": 42}
    history = [{"epoch": epoch, "validation": {"foreground_macro_iou": .5 if epoch == best_epoch else .3}} for epoch in range(1, epochs + 1)]
    (train / "config.json").write_text(json.dumps(config))
    (train / "history.json").write_text(json.dumps(history))
    (train / "metrics-validation.json").write_text(json.dumps(history[best_epoch - 1]["validation"]))
    for kind, epoch in (("last", epochs), ("best", best_epoch)):
        (train / f"{kind}.pt").write_text(json.dumps({"epochs_trained": epoch, "training": config,
            "validation": history[epoch - 1]["validation"], "model_version": f"run-epoch-{epoch}", "run_id": "run"}))


def read_fake_checkpoint(path):
    return json.loads(path.read_text())


class VertexEntryTests(unittest.TestCase):
    def test_subprocess_failure_keeps_combined_log_and_nonzero_exit(self):
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "command.log"
            synchronizer = unittest.mock.Mock()
            command = [sys.executable, "-u", "-c", "import sys; print('starting'); print('failure detail', file=sys.stderr); sys.exit(9)"]
            with self.assertRaises(subprocess.CalledProcessError) as caught:
                entry.run_command(command, log, synchronizer, .01)
            self.assertEqual(caught.exception.returncode, 9)
            self.assertIn("starting", log.read_text())
            self.assertIn("failure detail", log.read_text())

    def test_uri_rejects_ambiguous_or_unsafe_destinations(self):
        self.assertEqual(entry.parse_output_uri("gs://test-bucket/runs/control-1"), ("test-bucket", "runs/control-1"))
        for uri in ("https://bucket/x", "gs://bucket", "gs://bucket/", "gs://bucket/a/../b", "gs://bucket/a//b", "gs://bucket/x?key=secret", "gs://bucket/a%2Fb", "gs://user:pass@bucket/x"):
            with self.subTest(uri=uri), self.assertRaises(ValueError):
                entry.parse_output_uri(uri)

    def test_immutable_upload_recovers_uncertain_retry_but_rejects_collision(self):
        bucket = FakeBucket()
        store = entry.ArtifactStore(bucket, "run/attempts/one")
        first = store.record("status/running.json", {"state": "running"})
        retry = entry.ArtifactStore(bucket, "run/attempts/one")
        self.assertEqual(first, retry.record("status/running.json", {"state": "running"}))
        with self.assertRaisesRegex(RuntimeError, "collision"):
            retry.record("status/running.json", {"state": "different"})
        with self.assertRaisesRegex(RuntimeError, "collision"):
            entry.ArtifactStore(bucket, "run/attempts/one").record("status/running.json", {"state": "different"})

    def test_changed_file_is_not_published_as_stable(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, destination = Path(temporary) / "source", Path(temporary) / "copy"
            source.write_text("before")
            real_copy = entry.shutil.copyfile

            def replace_during_copy(src, dst):
                real_copy(src, dst)
                replacement = source.with_suffix(".tmp")
                replacement.write_text("after")
                replacement.replace(source)

            with patch.object(entry.shutil, "copyfile", side_effect=replace_during_copy):
                self.assertIsNone(entry.stable_copy(source, destination))
            self.assertFalse(destination.exists())

    def test_epoch_snapshot_requires_matching_history_and_preserves_old_objects(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bucket = FakeBucket()
            sync = entry.Synchronizer(root, entry.ArtifactStore(bucket, "run/attempts/a"), read_fake_checkpoint)
            make_epoch(root)
            sync.sync()
            self.assertEqual(sync.last_epoch, 1)
            epoch_one = dict(bucket.objects)
            make_epoch(root, epochs=2, best_epoch=2)
            (root / "train/history.json").write_text(json.dumps([{"epoch": 1, "validation": {"foreground_macro_iou": .3}}]))
            sync.sync()
            self.assertEqual(sync.last_epoch, 1, "An in-progress second epoch must not replace coherent epoch one")
            make_epoch(root, epochs=2, best_epoch=2)
            sync.sync(final=True)
            self.assertEqual(sync.last_epoch, 2)
            for name, value in epoch_one.items():
                self.assertEqual(bucket.objects[name], value)
            record = json.loads(bucket.objects["run/attempts/a/epochs/0002.json"][0])
            self.assertEqual(record["best_epoch"], 2)
            self.assertFalse(record["exact_resume_supported"])

    def args(self, root):
        return argparse.Namespace(data_root=root, output_uri="gs://test-bucket/runs/control", run_name="control",
            epochs=2, width=672, height=384, work_dir=root / "outputs", sync_seconds=.01,
            architecture=None, augmentation=None, sampling=None, seed=42)

    def test_commands_preserve_matched_control_and_optional_ablation_flags(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.args(Path(temporary))
            commands = entry.build_commands(args, args.work_dir)
            prepare = next(command for phase, command, _ in commands if phase == "prepare")
            train = next(command for phase, command, _ in commands if phase == "train")
            evaluate = next(command for phase, command, _ in commands if phase == "evaluate")
            self.assertEqual(prepare[prepare.index("--ignore-source-id") + 1], "255")
            self.assertEqual(prepare[prepare.index("--exclude-frame") + 1], "153_32700")
            self.assertEqual(train[train.index("--device") + 1], "cuda")
            self.assertEqual(train[train.index("--width") + 1], "672")
            self.assertEqual(train[train.index("--height") + 1], "384")
            self.assertNotIn("--architecture", train)
            self.assertNotIn("--augmentation", train)
            self.assertNotIn("--sampling", train)
            self.assertNotIn("--loss", train)
            self.assertNotIn("--dice-weight", train)
            self.assertEqual(evaluate[evaluate.index("--split") + 1], "val")
            args.architecture, args.augmentation, args.sampling, args.seed = "larger", "light", "case-balanced", 123
            train = next(command for phase, command, _ in entry.build_commands(args, args.work_dir) if phase == "train")
            for flag, value in (("architecture", "larger"), ("augmentation", "light"), ("sampling", "case-balanced"), ("seed", "123")):
                self.assertEqual(train[train.index(f"--{flag}") + 1], value)

            args.width, args.height = 896, 512
            train = next(command for phase, command, _ in entry.build_commands(args, args.work_dir) if phase == "train")
            self.assertEqual(train[train.index("--width") + 1], "896")
            self.assertEqual(train[train.index("--height") + 1], "512")

    def test_dice_objective_reaches_training_without_changing_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.args(Path(temporary))
            before = entry.build_commands(args, args.work_dir)
            args.loss, args.dice_weight = "ce_generalized_dice", 1.0
            after = entry.build_commands(args, args.work_dir)
            for (phase, old, _), (_, new, _) in zip(before, after):
                if phase == "train":
                    self.assertEqual(new[new.index("--loss") + 1], args.loss)
                    self.assertEqual(new[new.index("--dice-weight") + 1], "1.0")
                else:
                    self.assertEqual(new, old)

    def test_attempt_retry_is_fresh_and_failure_never_claims_completion(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, bucket = self.args(Path(temporary)), FakeBucket()

            def fail(command, log, sync, interval):
                log.write_text("runtime failure\n")
                raise subprocess.CalledProcessError(1, command)

            for _ in range(2):
                with self.assertRaises(subprocess.CalledProcessError):
                    entry.execute(args, bucket, run=fail, checkpoint_reader=read_fake_checkpoint)
            failures = [name for name in bucket.objects if name.endswith("status/failed.json")]
            self.assertEqual(len(failures), 2)
            self.assertFalse(any(name.endswith("status/completed.json") for name in bucket.objects))
            for name in failures:
                self.assertEqual(json.loads(bucket.objects[name][0])["epochs_preserved"], 0)

    def test_completion_manifest_requires_evaluation_and_all_epochs(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, bucket = self.args(Path(temporary)), FakeBucket()

            def successful(command, log, sync, interval):
                log.write_text("phase finished\n")
                if "train" in command:
                    make_epoch(sync.root, epochs=2, best_epoch=1)
                if "evaluate-original" in command:
                    (sync.root / "train/metrics-val-original.json").write_text('{"split":"val"}')
                sync.sync()

            with patch.object(entry, "audit_training_inputs"):
                completed = entry.execute(args, bucket, run=successful, checkpoint_reader=read_fake_checkpoint)
            self.assertEqual(completed["epochs_completed"], 2)
            self.assertIn("train/metrics-val-original.json", completed["artifacts"])
            self.assertTrue(any(name.endswith("status/completed.json") for name in bucket.objects))

            def missing_evaluation(command, log, sync, interval):
                log.write_text("finished\n")
                make_epoch(sync.root, epochs=2)

            with patch.object(entry, "audit_training_inputs"), self.assertRaisesRegex(RuntimeError, "incomplete"):
                entry.execute(args, bucket, run=missing_evaluation, checkpoint_reader=read_fake_checkpoint)


class PreparedManifestTests(unittest.TestCase):
    def setUp(self):
        from PIL import Image
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.prepared, self.data, self.output = [self.root / name for name in ("prepared", "data", "output")]
        for directory in (self.prepared, self.data, self.output):
            directory.mkdir()
        self.base = {"schemaVersion": "1.0.0", "dataset": "endoscapes-seg50", "root": "/original/data",
                     "classes": [{"index": 0, "sourceId": 0, "structureId": "background"},
                                 {"index": 1, "sourceId": 4, "structureId": "cystic_duct"}],
                     "ignoreIndex": 255, "ignoreSourceIds": [255], "samples": []}
        for index, split in enumerate(("train", "val", "test", "train"), 1):
            folder = self.data if index < 4 else self.prepared
            origin = "/original/data" if index < 4 else "/original/prepared"
            Image.new("RGB", (3, 2), (50, 60, 70)).save(folder / f"{index}.jpg")
            mask = Image.new("L", (3, 2), 4)
            mask.putpixel((0, 0), 255)
            mask.save(folder / f"{index}.png")
            sample = {"split": split, "videoId": index, "frameNumber": 100,
                      "timestampMs": 4000.0, "imagePath": f"{origin}/{index}.jpg", "maskPath": f"{origin}/{index}.png"}
            if index < 4:
                self.base["samples"].append(sample)
            else:
                sample.update(annotationSource="reviewed_partial_anatomy", reviewProvenancePath=f"{origin}/4.json")
                (folder / "4.json").write_text('{"reviewScope":"anatomy"}')
                self.added = sample
        self.manifest = {**copy.deepcopy(self.base), "dataset": "endoscapes-seg50+reviewed-partial",
                         "samples": self.base["samples"] + [self.added], "report": {"counts": {"train": 2, "val": 1, "test": 1}},
                         "reviewedPartialProvenance": {"preparationPath": "/original/prepared/summary.json",
                            "overlapPolicy": "ignore_conflicts", "unknownPixelPolicy": "ignore",
                            "reviewer": {"reviewScope": "anatomy"}, "bundleSha256": "a" * 64,
                            "reviewSha256": "b" * 64, "resolutionRecordSha256": "c" * 64}}
        # Real source manifests are independent immutable documents.
        self.manifest = copy.deepcopy(self.manifest)
        self.refresh()

    def refresh(self):
        base_path = self.prepared / "base-manifest.json"
        base_path.write_bytes(entry.json_bytes(self.base))
        base_sha = entry.sha256(base_path)
        self.manifest["reviewedPartialProvenance"]["baseManifestSha256"] = base_sha
        (self.prepared / "manifest.json").write_bytes(entry.json_bytes(self.manifest))
        manifest_sha = entry.sha256(self.prepared / "manifest.json")
        summary = {"artifactType": "reviewed_partial_training_preparation",
                   **{key: self.manifest["reviewedPartialProvenance"][key] for key in
                      ("baseManifestSha256", "reviewSha256", "bundleSha256", "resolutionRecordSha256")},
                   "combinedManifestSha256": manifest_sha,
                   "summary": {"combinedCounts": {"train": 2, "val": 1, "test": 1}, "addedTrainImageCount": 1, "baseSampleCount": 3},
                   "baseFiles": [{"imagePath": row["imagePath"], "maskPath": row["maskPath"],
                                  "imageSha256": entry.sha256(self.data / Path(row["imagePath"]).name),
                                  "maskSha256": entry.sha256(self.data / Path(row["maskPath"]).name)} for row in self.base["samples"]],
                   "artifactInventory": [{"path": name, "sha256": entry.sha256(self.prepared / name)}
                                         for name in ("manifest.json", "4.jpg", "4.png", "4.json")]}
        (self.prepared / "summary.json").write_bytes(entry.json_bytes(summary))
        self.package = {"formatVersion": "1.0.0", "artifactType": "holospex_prepared_cloud_package",
                        "manifestSha256": manifest_sha, "baseManifestSha256": base_sha, "files": []}
        for kind, folder in (("data", self.data), ("prepared", self.prepared)):
            for path in sorted(folder.iterdir()):
                if path.name == "package.json":
                    continue
                self.package["files"].append({"sourcePath": f"/original/{kind}/{path.name}", "root": kind,
                                              "path": path.name, "sha256": entry.sha256(path), "bytes": path.stat().st_size})
        self.save_package()

    def save_package(self):
        (self.prepared / "package.json").write_bytes(entry.json_bytes(self.package))

    def prepare(self):
        return entry.prepare_packaged_manifest(self.prepared / "package.json", self.data, self.output)

    def test_exact_sources_preserved_and_only_runtime_paths_rebased(self):
        before = {path: path.read_bytes() for folder in (self.prepared, self.data) for path in folder.iterdir()}
        report = self.prepare()
        actual = json.loads((self.output / "manifest.json").read_text())
        self.assertEqual(report["addedTrainImageCount"], 1)
        self.assertTrue(report["heldOutSamplesPreserved"])
        self.assertEqual(actual["samples"][1]["split"], "val")
        self.assertEqual(actual["samples"][1]["imagePath"], str(self.data / "2.jpg"))
        self.assertEqual(actual["samples"][3]["maskPath"], str(self.prepared / "4.png"))
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)
        self.assertEqual((self.output / "manifest-source.json").read_bytes(), (self.prepared / "manifest.json").read_bytes())

    def test_corrupt_bytes_fail_before_any_run_manifest_is_written(self):
        (self.data / "2.png").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "integrity mismatch"):
            self.prepare()
        self.assertFalse((self.output / "manifest.json").exists())

    def test_held_out_metadata_cannot_change_even_with_updated_package_hashes(self):
        self.manifest["samples"][1]["frameNumber"] += 1
        self.refresh()
        with self.assertRaisesRegex(ValueError, "ordered prefix"):
            self.prepare()

    def test_added_case_must_not_match_any_original_case(self):
        self.manifest["samples"][-1]["videoId"] = 2
        self.manifest["samples"][-1]["frameNumber"] += 1
        self.refresh()
        with self.assertRaisesRegex(ValueError, "leakage|new training cases"):
            self.prepare()

    def test_missing_review_provenance_inventory_entry_fails(self):
        self.package["files"] = [row for row in self.package["files"] if row["path"] != "4.json"]
        self.save_package()
        with self.assertRaisesRegex(ValueError, "artifact missing"):
            self.prepare()

    def test_unknown_labels_and_partial_background_are_rejected(self):
        from PIL import Image
        for value, message in ((9, "Unknown semantic"), (0, "fabricated background"), (255, "only ignored")):
            with self.subTest(value=value):
                Image.new("L", (3, 2), value).save(self.prepared / "4.png")
                self.refresh()
                with self.assertRaisesRegex(ValueError, message):
                    self.prepare()

    def test_escaping_symlink_and_duplicate_paths_are_rejected(self):
        for path in ("../data/1.jpg", str(self.data / "1.jpg"), "./1.jpg", "a//b"):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "Unsafe package"):
                entry.contained_file(self.prepared, path)
        (self.prepared / "link").symlink_to(self.data / "1.jpg")
        with self.assertRaisesRegex(ValueError, "Symlink"):
            entry.contained_file(self.prepared, "link")
        self.package["files"].append(self.package["files"][0])
        self.save_package()
        with self.assertRaisesRegex(ValueError, "unique absolute"):
            self.prepare()

    def test_audit_fingerprints_ignore_machine_paths_but_detect_mask_changes(self):
        self.prepare()
        first = entry.audit_training_inputs(self.output)
        alternate = self.root / "alternate"
        alternate.mkdir()
        manifest = json.loads((self.output / "manifest.json").read_text())
        for index, sample in enumerate(manifest["samples"]):
            for field in ("imagePath", "maskPath"):
                source = Path(sample[field])
                destination = alternate / f"{index}-{field}{source.suffix}"
                destination.write_bytes(source.read_bytes())
                sample[field] = str(destination)
        (alternate / "manifest.json").write_bytes(entry.json_bytes(manifest))
        second = entry.audit_training_inputs(alternate)
        self.assertEqual(first["splitContentSha256"], second["splitContentSha256"])
        Path(manifest["samples"][1]["maskPath"]).write_bytes(b"changed")
        third = entry.audit_training_inputs(alternate)
        self.assertNotEqual(first["splitContentSha256"]["val"], third["splitContentSha256"]["val"])
        self.assertEqual(first["splitContentSha256"]["test"], third["splitContentSha256"]["test"])

    def test_prepared_mode_skips_regeneration_and_validation_failure_never_trains(self):
        args = VertexEntryTests().args(self.root)
        args.data_root, args.prepared_package = self.data, self.prepared / "package.json"
        self.assertNotIn("prepare", [phase for phase, _, _ in entry.build_commands(args, self.output)])
        (self.prepared / "4.png").write_bytes(b"corrupted")
        runner = unittest.mock.Mock()
        bucket = FakeBucket()
        with self.assertRaisesRegex(ValueError, "integrity mismatch"):
            entry.execute(args, bucket, run=runner)
        runner.assert_not_called()
        self.assertTrue(any(name.endswith("status/failed.json") for name in bucket.objects))

    def test_strict_json_rejects_duplicate_keys_and_nonfinite_numbers(self):
        for invalid in ('{"key": 1, "key": 2}', '{"key": NaN}'):
            (self.prepared / "package.json").write_text(invalid)
            with self.assertRaises(ValueError):
                self.prepare()


if __name__ == "__main__":
    unittest.main()
