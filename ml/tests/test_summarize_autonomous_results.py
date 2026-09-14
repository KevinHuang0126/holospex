"""Report fresh audits, failed evidence, and timing without cloud/model access."""

import contextlib
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

from test_autonomous_audit import fixture, write_run

CLOUD = Path(__file__).resolve().parents[1] / "cloud"
sys.path.insert(0, str(CLOUD))
try:
    SPEC = importlib.util.spec_from_file_location("summarize_autonomous_results", CLOUD / "summarize_autonomous_results.py")
    report = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(report)
finally:
    sys.path.pop(0)

SNAPSHOT = datetime(2030, 1, 1, 1, tzinfo=timezone.utc)


class AutonomousResultsReportTests(unittest.TestCase):
    def completed(self, root, name, *, perfect=True, partial=False):
        directory = root / "results" / name
        files = fixture(perfect)
        files["train/config.json"].update(initialization="COCO_WITH_VOC_LABELS_V1", seed=42,
            learning_rate=.0003, lr_schedule="cosine", batch_size=2, loss="ce", auxiliary_loss_weight=.4,
            dataset={"dataset": "synthetic-reviewed-reference"})
        files["train/history.json"][0]["duration_seconds"] = 1.25
        completion = {"training_duration_seconds": 4.5}
        if partial:
            files["train/config.json"].update(epochs=40, max_duration_seconds=60)
            completion.update(epochs_requested=40, epochs_completed=1, budget_exhausted=True,
                              stop_reason="max_duration_seconds", training_duration_seconds=61.5)
        write_run(directory, files, completion)
        return {"name": name, "state": "JOB_STATE_SUCCEEDED", "result_dir": str(directory),
                "recipe": {"label": "requested-name", "lr": .9},
                "audit": {"verified": True, "target_reached": True, "foreground_macro_iou": 1.0}}

    def state(self, root, runs, references=None, **overrides):
        value = {"status": "running", "started_at": "2030-01-01T00:00:00Z", "deadline": "2030-01-01T04:00:00Z",
                 "updated_at": "2030-01-01T00:59:00Z", "runs": runs, "reference_runs": references or [],
                 "queue": [{"label": "queued"}], "target": {"metric": "foreground_macro_iou", "threshold": .75, "split": "val"},
                 **overrides}
        path = root / "state.json"
        path.write_text(json.dumps(value))
        return path

    def metadata(self, root, run):
        jobs = root / "jobs"
        jobs.mkdir(exist_ok=True)
        source_hash = "a" * 64
        configuration = jobs / (run["name"] + ".json")
        configuration.write_text(json.dumps({"workerPoolSpecs": [{"containerSpec": {"env": [
            {"name": "HOLOSPEX_SOURCE_SHA256", "value": source_hash}]}}]}))
        run.update(config=str(configuration), config_sha256=hashlib.sha256(configuration.read_bytes()).hexdigest(),
                   bundle_references={"source": {"sha256": source_hash}})
        (jobs / (run["name"] + ".status.json")).write_text(json.dumps({"state": "JOB_STATE_SUCCEEDED",
            "startTime": "2030-01-01T00:00:00Z", "endTime": "2030-01-01T00:15:00Z", "updateTime": "2030-01-01T00:15:02Z"}))

    def test_mixed_snapshot_reaudits_metrics_and_preserves_all_states(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            good = self.completed(root, "good")
            self.metadata(root, good)
            reference = self.completed(root, "reviewed-reference", perfect=False)
            runs = [good, {"name": "active", "state": "JOB_STATE_RUNNING", "recipe": {}},
                    {"name": "failed", "state": "JOB_STATE_FAILED", "error": {"code": 7, "message": "worker failed"}},
                    {"name": "uncollected", "state": "JOB_STATE_SUCCEEDED", "audit": {"verified": True, "target_reached": True}}]
            state = self.state(root, runs, [reference])
            inputs = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
            result = report.create_report(state, root / "report", snapshot_time=SNAPSHOT)
            self.assertEqual(result["counts"]["runs"], 4)
            self.assertEqual(result["counts"]["verified"], 1)
            self.assertEqual(result["counts"]["active"], 1)
            self.assertEqual(result["counts"]["failed_or_cancelled_or_expired"], 1)
            self.assertEqual(result["counts"]["awaiting_local_result"], 1)
            self.assertTrue(result["search_active"])
            self.assertFalse(result["deadline_reached"])
            self.assertTrue(result["target"]["met_by_verified_search_run"])
            self.assertEqual(result["best_search_run"], "good")
            self.assertEqual(result["comparison_reference"], "reviewed-reference")
            audited = result["runs"][0]
            self.assertEqual([audited[key] for key in report.SCORE_KEYS], [1.0, 1.0, 1.0])
            self.assertEqual(audited["recipe"]["learning_rate"], .0003, "Actual verified config wins over requested recipe")
            self.assertEqual(audited["source_bundle"]["basis"], "submitted_job_config_sha256_checked")
            self.assertEqual(audited["timing"]["training_epoch_seconds"], 1.25)
            self.assertEqual(audited["timing"]["training_call_seconds"], 4.5)
            self.assertEqual(audited["timing"]["vertex_worker_runtime_seconds"], 900)
            self.assertEqual(len(result["best_search_per_class_tradeoffs"]), 6)
            self.assertTrue(all(row["delta_percentage_points"] == 100 for row in result["best_search_per_class_tradeoffs"]))
            self.assertEqual(inputs, {path: path.read_bytes() for path in inputs})
            saved = json.loads((root / "report/summary.json").read_text())
            self.assertEqual(saved, result)
            markdown = (root / "report/RESULTS.md").read_text()
            for expected in ("worker failed", "uncollected", "100.00%", "1.25 s", "900.0 s", "SHA-256", "No test-set evaluation"):
                self.assertIn(expected, markdown)

    def test_cached_success_never_survives_corrupt_or_failing_evidence(self):
        for corruption in ("checkpoint", "test_split", "actual_zero_score"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                run = self.completed(root, "candidate", perfect=corruption != "actual_zero_score")
                if corruption == "checkpoint":
                    (Path(run["result_dir"]) / "train/best.pt").write_bytes(b"corrupted")
                elif corruption == "test_split":
                    files = fixture()
                    files["train/metrics-val-original.json"]["split"] = "test"
                    write_run(Path(run["result_dir"]), files)
                state = self.state(root, [run], status="target_reached", deadline="2030-01-01T00:30:00Z")
                result = report.summarize_state(state, snapshot_time=SNAPSHOT)
                self.assertFalse(result["target"]["met_by_verified_search_run"])
                self.assertTrue(result["deadline_reached"])
                self.assertFalse(result["search_active"])
                row = result["runs"][0]
                if corruption == "actual_zero_score":
                    self.assertTrue(row["verified"])
                    self.assertEqual(row["foreground_macro_iou"], 0)
                else:
                    self.assertFalse(row["verified"])
                    self.assertEqual(row["verification_status"], "audit_failed")
                    self.assertNotIn("foreground_macro_iou", row)

    def test_reference_cannot_be_claimed_as_a_new_search_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = self.completed(root, "reviewed-reference")
            state = self.state(root, [{"name": "cancelled", "state": "JOB_STATE_CANCELLED"}], [reference], status="deadline_reached")
            result = report.summarize_state(state, snapshot_time=SNAPSHOT)
            self.assertEqual(result["best_available_run"], "reviewed-reference")
            self.assertIsNone(result["best_search_run"])
            self.assertFalse(result["target"]["met_by_verified_search_run"])
            self.assertEqual(result["best_search_per_class_tradeoffs"], [])

    def test_duration_limited_result_keeps_selected_completed_and_requested_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = self.state(root, [self.completed(root, "bounded", partial=True)])
            result = report.summarize_state(state, snapshot_time=SNAPSHOT)["runs"][0]
            self.assertTrue(result["verified"])
            self.assertTrue(result["duration_limited"])
            self.assertEqual((result["selected_epoch"], result["epochs_completed"], result["epochs_requested"]), (1, 1, 40))
            self.assertEqual(result["timing"]["training_call_seconds"], 61.5)
            self.assertIsNone(result["timing"]["vertex_worker_runtime_seconds"])

    def test_report_paths_cannot_overwrite_or_be_nested_in_experiment_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = self.completed(root, "candidate")
            failed = {"name": "failed", "state": "JOB_STATE_FAILED", "result_dir": str(root / "failed-evidence")}
            state = self.state(root, [candidate, failed])
            protected = [root, state, Path(candidate["result_dir"]) / "report", root / "jobs/new-report",
                         root / "failed-evidence/report"]
            for output in protected:
                with self.subTest(output=output), self.assertRaises(ValueError):
                    report.create_report(state, output, snapshot_time=SNAPSHOT)
            output = root / "report"
            report.create_report(state, output, snapshot_time=SNAPSHOT)
            before = (output / "summary.json").read_bytes()
            with self.assertRaises(ValueError):
                report.create_report(state, output, snapshot_time=SNAPSHOT)
            self.assertEqual((output / "summary.json").read_bytes(), before)

    def test_unverified_source_hash_is_marked_separately_from_audited_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = self.completed(root, "candidate")
            self.metadata(root, candidate)
            Path(candidate["config"]).write_text("{}")
            result = report.summarize_state(self.state(root, [candidate]), snapshot_time=SNAPSHOT)["runs"][0]
            self.assertTrue(result["verified"])
            self.assertEqual(result["source_bundle"]["basis"], "controller_record_only")
            self.assertIn("SHA-256", result["source_bundle"]["warning"])

    def test_empty_ongoing_snapshot_and_cli_create_valid_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = self.state(root, [])
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(report.main(["--state", str(state), "--output-dir", str(root / "report")]), 0)
            result = json.loads((root / "report/summary.json").read_text())
            self.assertTrue(result["search_active"])
            self.assertFalse(result["target"]["met_by_verified_search_run"])
            self.assertEqual(result["counts"]["runs"], 0)


if __name__ == "__main__":
    unittest.main()
