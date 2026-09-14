"""Duration limits and warm starts remain auditable without cloud access."""

import argparse
import contextlib
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from test_vertex_bootstrap import bootstrap
from test_vertex_entry import FakeBucket, entry, make_epoch, read_fake_checkpoint


class CloudOptimizationTests(unittest.TestCase):
    def args(self, root):
        return argparse.Namespace(data_root=root, output_uri="gs://test-bucket/runs/bounded", run_name="bounded",
            epochs=4, width=672, height=384, work_dir=root / "outputs", sync_seconds=.01,
            architecture=None, augmentation=None, sampling=None, seed=42, max_duration_seconds=10.0)

    def runner(self, args, *, actual=2, report=None, config_changes=None, fail=False, evaluation=True):
        if report is None:
            report = {"epochs_completed": actual, "stop_reason": "max_duration_seconds", "duration_seconds": 12.0}

        def run(command, log, sync, interval):
            log.write_text("phase finished\n")
            if "train" in command:
                make_epoch(sync.root, epochs=actual)
                config_path = sync.root / "train/config.json"
                config = {**json.loads(config_path.read_text()), "epochs": args.epochs,
                          "max_duration_seconds": float(command[command.index("--max-duration-seconds") + 1]), **(config_changes or {})}
                config_path.write_text(json.dumps(config))
                for name in ("best.pt", "last.pt"):
                    path = sync.root / "train" / name
                    checkpoint = json.loads(path.read_text())
                    checkpoint["training"] = config
                    path.write_text(json.dumps(checkpoint))
                log.write_text("epoch progress before final report\n" + (json.dumps(report, indent=2) if report else ""))
                if fail:
                    raise subprocess.CalledProcessError(7, command)
            if "evaluate-original" in command and evaluation:
                (sync.root / "train/metrics-val-original.json").write_text('{"split":"val"}')
            sync.sync()
        return run

    def test_graceful_duration_stop_preserves_requested_and_actual_epochs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, bucket = Path(temporary), FakeBucket()
            args = self.args(root)
            with patch.object(entry, "audit_training_inputs"), contextlib.redirect_stdout(io.StringIO()):
                completed = entry.execute(args, bucket, run=self.runner(args), checkpoint_reader=read_fake_checkpoint)
            self.assertEqual(completed["epochs_requested"], 4)
            self.assertEqual(completed["epochs_completed"], 2)
            self.assertTrue(completed["budget_exhausted"])
            self.assertEqual(completed["stop_reason"], "max_duration_seconds")
            self.assertEqual(completed["training_duration_seconds"], 12.0)
            self.assertIn("train/metrics-val-original.json", completed["artifacts"])
            self.assertTrue(any(name.endswith("status/completed.json") for name in bucket.objects))

    def test_finishing_all_epochs_before_duration_cap_is_not_budget_exhaustion(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, bucket = self.args(Path(temporary)), FakeBucket()
            report = {"epochs_completed": 4, "stop_reason": "epochs_completed", "duration_seconds": 3.0}
            with patch.object(entry, "audit_training_inputs"), contextlib.redirect_stdout(io.StringIO()):
                completed = entry.execute(args, bucket, run=self.runner(args, actual=4, report=report), checkpoint_reader=read_fake_checkpoint)
            self.assertFalse(completed["budget_exhausted"])
            self.assertEqual(completed["stop_reason"], "epochs_completed")

    def test_duration_option_cannot_relabel_incomplete_or_failed_runs_completed(self):
        valid = {"epochs_completed": 2, "stop_reason": "max_duration_seconds", "duration_seconds": 12.0}
        variants = [
            {"report": {}},
            {"report": {**valid, "stop_reason": "epochs_completed"}},
            {"report": {**valid, "epochs_completed": 1}},
            {"report": {**valid, "duration_seconds": 9.0}},
            {"report": {**valid, "duration_seconds": float("nan")}},
            {"config_changes": {"epochs": 2}},
            {"config_changes": {"max_duration_seconds": None}},
            {"evaluation": False},
            {"fail": True},
            {"actual": 5, "report": {**valid, "epochs_completed": 5}},
        ]
        for variant in variants:
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as temporary:
                args, bucket = self.args(Path(temporary)), FakeBucket()
                with patch.object(entry, "audit_training_inputs"), contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises((RuntimeError, subprocess.CalledProcessError)):
                        entry.execute(args, bucket, run=self.runner(args, **variant), checkpoint_reader=read_fake_checkpoint)
                self.assertFalse(any(name.endswith("status/completed.json") for name in bucket.objects))
                self.assertTrue(any(name.endswith("status/failed.json") for name in bucket.objects))

    def test_invalid_optimization_inputs_fail_before_cloud_attempt_creation(self):
        for option, value in (("lr", 0), ("weight_decay", -1), ("max_duration_seconds", float("nan")),
                              ("backbone_lr_multiplier", float("inf")), ("warmup_epochs", 4),
                              ("lr_schedule", "unknown"), ("sync_seconds", float("nan")), ("lovasz_weight", -1)):
            with self.subTest(option=option), tempfile.TemporaryDirectory() as temporary:
                args, bucket = self.args(Path(temporary)), FakeBucket()
                setattr(args, option, value)
                with self.assertRaises(ValueError):
                    entry.execute(args, bucket)
                self.assertFalse(bucket.objects)
                self.assertFalse(args.work_dir.exists())

    def test_bootstrap_rejects_bad_checkpoint_or_optimization_before_runtime(self):
        environment = {"HOLOSPEX_SOURCE_URI": "gs://bucket/source.tar.gz", "HOLOSPEX_SOURCE_SHA256": "0" * 64,
                       "HOLOSPEX_DATA_URI": "gs://bucket/data.tar.gz", "HOLOSPEX_DATA_SHA256": "1" * 64,
                       "HOLOSPEX_OUTPUT_URI": "gs://bucket/runs/test", "HOLOSPEX_RUN_NAME": "test", "HOLOSPEX_EPOCHS": "4"}
        variants = [{"HOLOSPEX_INITIAL_CHECKPOINT_URI": "gs://bucket/initial.pt"},
                    {"HOLOSPEX_INITIAL_CHECKPOINT_SHA256": "2" * 64},
                    {"HOLOSPEX_INITIAL_CHECKPOINT_URI": "https://bucket/initial.pt", "HOLOSPEX_INITIAL_CHECKPOINT_SHA256": "2" * 64},
                    {"HOLOSPEX_INITIAL_CHECKPOINT_URI": "gs://bucket/initial.pt", "HOLOSPEX_INITIAL_CHECKPOINT_SHA256": "invalid"},
                    {"HOLOSPEX_MAX_DURATION_SECONDS": "nan"}, {"HOLOSPEX_LR": "0"},
                    {"HOLOSPEX_WARMUP_EPOCHS": "4"}, {"HOLOSPEX_LR_SCHEDULE": "unknown"},
                    {"HOLOSPEX_BACKBONE_LR_MULTIPLIER": "infinity"}, {"HOLOSPEX_WEIGHT_DECAY": "-1"},
                    {"HOLOSPEX_LOVASZ_WEIGHT": "nan"}, {"HOLOSPEX_DEADLINE_UTC": "2020-01-01T00:00:00Z"},
                    {"HOLOSPEX_DEADLINE_UTC": "2100-01-01T00:00:00"}]
        for variant in variants:
            with self.subTest(variant=variant), patch.dict(bootstrap.os.environ, {**environment, **variant}, clear=True), \
                 patch.object(bootstrap, "verify_runtime") as runtime, patch.object(bootstrap.subprocess, "run") as process:
                with self.assertRaises(ValueError):
                    bootstrap.main()
                runtime.assert_not_called()
                process.assert_not_called()

    def test_checkpoint_and_optimization_flags_reach_cuda_training_only(self):
        environment = {"HOLOSPEX_OUTPUT_URI": "gs://bucket/runs/test", "HOLOSPEX_RUN_NAME": "test", "HOLOSPEX_EPOCHS": "4",
                       "HOLOSPEX_INITIAL_CHECKPOINT_URI": "gs://bucket/initial.pt", "HOLOSPEX_INITIAL_CHECKPOINT_SHA256": "2" * 64,
                       "HOLOSPEX_LR": "0.0001", "HOLOSPEX_LR_SCHEDULE": "cosine", "HOLOSPEX_WARMUP_EPOCHS": "1",
                       "HOLOSPEX_WEIGHT_DECAY": "0.02", "HOLOSPEX_BACKBONE_LR_MULTIPLIER": "0.1", "HOLOSPEX_MAX_DURATION_SECONDS": "600"}
        bootstrap.validate_optimization_environment(environment)
        command = bootstrap.entry_command(environment, Path("/work"))
        for flag, value in (("lr", "0.0001"), ("lr-schedule", "cosine"), ("warmup-epochs", "1"),
                            ("weight-decay", "0.02"), ("backbone-lr-multiplier", "0.1"),
                            ("max-duration-seconds", "600"), ("initial-checkpoint", "/work/initial.pt")):
            self.assertEqual(command[command.index("--" + flag) + 1], value)
        with tempfile.TemporaryDirectory() as temporary:
            args = self.args(Path(temporary))
            args.lr, args.lr_schedule, args.warmup_epochs = 0.0001, "cosine", 1
            args.weight_decay, args.backbone_lr_multiplier = 0.02, 0.1
            args.initial_checkpoint = Path(temporary) / "initial.pt"
            train = next(command for phase, command, _ in entry.build_commands(args, args.work_dir) if phase == "train")
            evaluation = next(command for phase, command, _ in entry.build_commands(args, args.work_dir) if phase == "evaluate")
            self.assertEqual(train[train.index("--initial-checkpoint") + 1], str(args.initial_checkpoint))
            for flag in ("--lr", "--lr-schedule", "--warmup-epochs", "--weight-decay", "--backbone-lr-multiplier", "--max-duration-seconds"):
                self.assertIn(flag, train)
                self.assertNotIn(flag, evaluation)

    def test_lovasz_objective_and_scalar_propagate_without_changing_evaluation(self):
        environment = {"HOLOSPEX_OUTPUT_URI": "gs://bucket/runs/test", "HOLOSPEX_RUN_NAME": "test", "HOLOSPEX_EPOCHS": "4",
                       "HOLOSPEX_LOSS": "ce_lovasz", "HOLOSPEX_LOVASZ_WEIGHT": "0.25"}
        bootstrap.validate_optimization_environment(environment)
        command = bootstrap.entry_command(environment, Path("/work"))
        self.assertEqual(command[command.index("--loss") + 1], "ce_lovasz")
        self.assertEqual(command[command.index("--lovasz-weight") + 1], "0.25")
        with tempfile.TemporaryDirectory() as temporary:
            args = self.args(Path(temporary))
            before = entry.build_commands(args, args.work_dir)
            args.loss, args.lovasz_weight = "ce_lovasz", .25
            after = entry.build_commands(args, args.work_dir)
            for (phase, old, _), (_, new, _) in zip(before, after):
                if phase == "train":
                    self.assertEqual(new[new.index("--loss") + 1], "ce_lovasz")
                    self.assertEqual(new[new.index("--lovasz-weight") + 1], "0.25")
                else:
                    self.assertEqual(new, old)

    def test_aware_deadlines_normalize_to_utc_and_reserve_evaluation_time(self):
        current = datetime(2030, 1, 1, 12, tzinfo=timezone.utc)
        expected = current + timedelta(minutes=10)
        for validate in (bootstrap.validate_deadline, entry.parse_deadline):
            self.assertEqual(validate("2030-01-01T14:10:00+02:00", now=current), expected)
            for invalid in ("2030-01-01T12:10:00", "invalid", "2030-01-01T12:00:00Z"):
                with self.subTest(validate=validate, invalid=invalid), self.assertRaises(ValueError):
                    validate(invalid, now=current)
        for requested, effective in ((None, 480), (900, 480), (300, 300)):
            args = argparse.Namespace(deadline_utc=expected.isoformat(), max_duration_seconds=requested)
            budget = entry.apply_deadline_budget(args, now=current)
            self.assertEqual(args.max_duration_seconds, effective)
            self.assertEqual(budget["requested_max_duration_seconds"], requested)
            self.assertEqual(budget["evaluation_and_upload_reserve_seconds"], 120)
        args = argparse.Namespace(deadline_utc=(current + timedelta(seconds=120)).isoformat(), max_duration_seconds=None)
        with self.assertRaisesRegex(ValueError, "120 seconds"):
            entry.apply_deadline_budget(args, now=current)

    def test_worker_recomputes_deadline_after_audit_and_records_actual_command_cap(self):
        current = [datetime(2030, 1, 1, 12, tzinfo=timezone.utc)]
        with tempfile.TemporaryDirectory() as temporary:
            args, bucket = self.args(Path(temporary)), FakeBucket()
            args.deadline_utc = (current[0] + timedelta(minutes=10)).isoformat()
            args.max_duration_seconds = 900.0
            def audit(_):
                current[0] += timedelta(minutes=3)
            report = {"epochs_completed": 2, "stop_reason": "max_duration_seconds", "duration_seconds": 302.0}
            with patch.object(entry, "utc_now", side_effect=lambda: current[0].isoformat()), \
                 patch.object(entry, "audit_training_inputs", side_effect=audit), contextlib.redirect_stdout(io.StringIO()):
                completed = entry.execute(args, bucket, run=self.runner(args, report=report), checkpoint_reader=read_fake_checkpoint)
            self.assertEqual(args.max_duration_seconds, 900, "Execution must preserve the caller's requested cap")
            self.assertEqual(completed["max_duration_seconds_requested"], 900)
            self.assertEqual(completed["max_duration_seconds_effective"], 300)
            self.assertEqual(completed["training_duration_seconds"], 302)
            output = next(args.work_dir.iterdir())
            budget = entry.read_strict_json(output / "deadline-budget.json")
            self.assertEqual(budget["remaining_seconds"], 420)
            self.assertEqual(budget["effective_max_duration_seconds"], 300)
            self.assertEqual(entry.read_strict_json(output / "train/config.json")["max_duration_seconds"], 300)
            command = next(command for phase, command, _ in entry.read_strict_json(output / "commands.json") if phase == "train")
            self.assertEqual(float(command[command.index("--max-duration-seconds") + 1]), 300)
            self.assertIn("deadline-budget.json", completed["artifacts"])

    def test_worker_refuses_training_if_preparation_uses_remaining_window(self):
        current = [datetime(2030, 1, 1, 12, tzinfo=timezone.utc)]
        with tempfile.TemporaryDirectory() as temporary:
            args, bucket = self.args(Path(temporary)), FakeBucket()
            args.deadline_utc = (current[0] + timedelta(minutes=5)).isoformat()
            phases = []
            def run(command, log, sync, interval):
                phases.append(command)
                log.write_text("finished")
            def audit(_):
                current[0] += timedelta(minutes=3)
            with patch.object(entry, "utc_now", side_effect=lambda: current[0].isoformat()), \
                 patch.object(entry, "audit_training_inputs", side_effect=audit), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "120 seconds"):
                    entry.execute(args, bucket, run=run, checkpoint_reader=read_fake_checkpoint)
            self.assertFalse(any("train" in command for command in phases))
            self.assertFalse(any(name.endswith("status/completed.json") for name in bucket.objects))

    def test_bootstrap_forwards_absolute_deadline(self):
        environment = {"HOLOSPEX_OUTPUT_URI": "gs://bucket/runs/test", "HOLOSPEX_RUN_NAME": "test", "HOLOSPEX_EPOCHS": "4",
                       "HOLOSPEX_DEADLINE_UTC": "2030-01-01T12:10:00Z"}
        command = bootstrap.entry_command(environment, Path("/work"))
        self.assertEqual(command[command.index("--deadline-utc") + 1], environment["HOLOSPEX_DEADLINE_UTC"])


if __name__ == "__main__":
    unittest.main()
