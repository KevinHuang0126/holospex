"""Bounded spending, ambiguous submissions, stopping, and candidate fidelity."""

import copy
from datetime import datetime, timedelta, timezone
import fcntl
import importlib.util
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch


CLOUD = Path(__file__).resolve().parents[1] / "cloud"
SPEC = importlib.util.spec_from_file_location("autonomous_loop", CLOUD / "autonomous_loop.py")
loop = importlib.util.module_from_spec(SPEC)
with patch.object(sys, "path", [str(CLOUD), *sys.path]):
    SPEC.loader.exec_module(loop)
START = datetime(2026, 9, 13, 20, tzinfo=timezone.utc)


def fixture_state():
    bundles = {kind: {"uri": f"{loop.BUCKET}/bundles/{digit * 64}/{kind}.tar.gz", "sha256": digit * 64}
               for kind, digit in (("source", "1"), ("data", "2"), ("prepared", "3"))}
    env = {"HOLOSPEX_PROJECT": loop.PROJECT,
           **{f"HOLOSPEX_{kind.upper()}_{field.upper()}": value
              for kind, bundle in bundles.items() for field, value in bundle.items()}}
    return {"status": "ready", "deadline": (START + timedelta(hours=4)).isoformat(),
            "prefix": "test", "max_parallel": 2, "max_runs": 16, "poll_seconds": 1,
            "queue": [loop.recipe("one", "test"), loop.recipe("two", "test")], "runs": [],
            **bundles,
            "template": {"workerPoolSpecs": [{"replicaCount": 1,
                         "machineSpec": {"machineType": "a2-highgpu-1g", "acceleratorType": "NVIDIA_TESLA_A100", "acceleratorCount": 1},
                         "containerSpec": {"imageUri": "docker.io/pytorch/pytorch@sha256:" + "a" * 64,
                                           "command": ["python", "-u", "-c"], "args": ["# original bootstrap"],
                                           "env": [{"name": key, "value": value} for key, value in env.items()]}}],
                         "scheduling": {"strategy": "SPOT", "disableRetries": True, "timeout": "14400s"}}}


def completed(label, *, score=0.5, **recipe_changes):
    return {"name": label, "state": "JOB_STATE_SUCCEEDED", "recipe": loop.recipe(label, "test", **recipe_changes),
            "audit": {"verified": True, "foreground_macro_iou": score, "checkpoint_sha256": "abc"},
            "checkpoint_reference": {"uri": "gs://bucket/leader.pt", "sha256": "abc"}}


def backbone_reference():
    return {"uri": f"{loop.BUCKET}/bundles/{'b' * 64}/surgical-backbone.pt", "sha256": "b" * 64}


def full_completed(label, *, score=0.5, **changes):
    result = completed(label, score=score, **changes)
    result["audit"].update(duration_limited=False, epochs_completed=result["recipe"]["epochs"],
                           epochs_requested=result["recipe"]["epochs"], checkpoint_sha256="a" * 64)
    return result


def replication_marker(seed=43, **changes):
    return {"replicate_best_full": "leader-seeds-43-44", "label": f"best-full-seed{seed}",
            "reason": "Measure seed variation.", "seed": seed, **changes}


class CandidateTests(unittest.TestCase):
    def test_new_replication_group_keeps_queue_until_every_candidate_is_terminal_and_audited(self):
        unready = []
        for status in ("SUBMIT_INTENT", "JOB_STATE_PENDING", "JOB_STATE_RUNNING", "JOB_STATE_CANCELLING"):
            run = full_completed("unresolved", score=0.6)
            run["state"] = status
            unready.append(run)
        for audit in (None, {}, {"verified": False, "foreground_macro_iou": 0.6},
                      {"verified": True, "foreground_macro_iou": float("nan")}):
            run = full_completed("unresolved", score=0.6)
            if audit is None:
                run.pop("audit")
                run["inspection_error"] = "Collection failed"
            else:
                run["audit"] = audit
            unready.append(run)
        for run in unready:
            with self.subTest(run=run):
                state = fixture_state()
                state.update(queue=[replication_marker(), replication_marker(44)],
                             runs=[full_completed("early", score=0.51), run])
                queued = copy.deepcopy(state["queue"])
                self.assertIsNone(loop.choose_next(state, 4000))
                self.assertEqual(state["queue"], queued)
                self.assertNotIn("replication_groups", state)

    def test_replication_marker_rejects_recipe_overrides_and_invalid_groups_or_seeds(self):
        loop.validate_recipe(replication_marker())
        for changes in ({"lr": 0.1}, {"epochs": 3}, {"replicate_best_full": "../other"},
                        {"replicate_best_full": True}, {"replicate_best_full": ""},
                        {"seed": 42}, {"seed": True}, {"seed": "43"}, {"reason": ""}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                loop.validate_recipe({**replication_marker(), **changes})

    def test_best_full_excludes_partial_unverified_cached_and_inconsistent_candidates(self):
        valid = full_completed("valid", score=0.51)
        sources = [valid]
        for changes in ({"verified": False}, {"duration_limited": True, "epochs_completed": 20},
                        {"epochs_completed": 20}, {"epochs_requested": True}, {"duration_limited": None},
                        {"checkpoint_sha256": "bad"}):
            invalid = full_completed("invalid", score=0.99)
            invalid["audit"].update(changes)
            sources.append(invalid)
        historical = full_completed("cached", score=0.99)
        historical["historical"] = True
        sources.append(historical)
        self.assertIs(loop.best_full_run(sources), valid)
        self.assertIsNone(loop.best_full_run(sources[1:]))

    def test_auxiliary_loss_weight_is_optional_and_accepts_only_bounded_finite_numbers(self):
        original = loop.recipe("legacy", "test")
        self.assertNotIn("auxiliary_loss_weight", original)
        loop.validate_recipe(original)
        for value in (0, 0.1, 0.4, 10):
            loop.validate_recipe({**original, "auxiliary_loss_weight": value})
        for value in (-0.01, 10.01, float("nan"), float("inf"), True, "0.4", None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "auxiliary_loss_weight"):
                loop.validate_recipe({**original, "auxiliary_loss_weight": value})

    def test_stronger_historical_reference_drives_refinement_without_using_launch_budget(self):
        historical = completed("reviewed-002", score=0.4587, epochs=40, lr=0.0003, lr_schedule="none")
        weaker = completed("new-regression", score=0.43)
        state = fixture_state()
        state.update(queue=[], runs=[weaker], reference_runs=[historical])
        candidate = loop.choose_next(state, 4000)
        self.assertEqual(candidate["initial_checkpoint"], historical["checkpoint_reference"])
        self.assertIn(historical["name"], candidate["reason"])
        self.assertEqual(candidate["lr_schedule"], "cosine")
        self.assertIsNone(candidate["backbone_checkpoint"])
        self.assertEqual(state["runs"], [weaker])
        self.assertEqual(state["reference_runs"], [historical])

    def test_surgical_backbone_validation_rejects_conflicting_or_unpinned_initialization(self):
        candidate = loop.recipe("surgical", "test", architecture="deeplabv3_resnet50", backbone_checkpoint=backbone_reference())
        loop.validate_recipe(candidate)
        for change in ({"architecture": loop.MOBILE}, {"architecture": loop.DETAIL},
                       {"initial_checkpoint": {"uri": f"{loop.BUCKET}/objects/{'c' * 64}/best.pt", "sha256": "c" * 64}},
                       {"backbone_checkpoint": {"uri": "gs://other-bucket/bundles/" + "b" * 64 + "/weights.pt", "sha256": "b" * 64}},
                       {"backbone_checkpoint": {"uri": f"{loop.BUCKET}/weights/latest.pt", "sha256": "b" * 64}},
                       {"backbone_checkpoint": {"uri": backbone_reference()["uri"], "sha256": "c" * 64}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                loop.validate_recipe({**candidate, **change})

    def test_surgical_leader_initialization_is_retained_only_for_same_architecture_recipes(self):
        leader = completed("surgical", score=0.6, architecture="deeplabv3_resnet50", backbone_checkpoint=backbone_reference())
        state = fixture_state()
        state["queue"] = []
        state["runs"] = [leader]
        refined = loop.choose_next(state, 4000)
        self.assertEqual(refined["initial_checkpoint"], leader["checkpoint_reference"])
        self.assertIsNone(refined["backbone_checkpoint"])
        state["runs"].append(completed("best-refine"))
        detail = loop.choose_next(state, 4000)
        self.assertEqual(detail["architecture"], loop.DETAIL)
        self.assertIsNone(detail["initial_checkpoint"])
        self.assertIsNone(detail["backbone_checkpoint"])
        state["runs"].append(completed("detail-896"))
        for label in ("best-mild", "best-decay", "best-native", "replicate-43"):
            candidate = loop.choose_next(state, 4000)
            self.assertEqual(candidate["label"], label)
            self.assertEqual(candidate["backbone_checkpoint"], backbone_reference())
            self.assertEqual(candidate["architecture"], "deeplabv3_resnet50")
            self.assertIsNone(candidate.get("initial_checkpoint"))
            state["runs"].append(completed(label))

    def test_only_verified_finite_successful_scores_can_win_or_reach_target(self):
        valid = completed("valid", score=0.74)
        for changes in ({"verified": False, "foreground_macro_iou": 0.99, "target_reached": True},
                        {"verified": True, "foreground_macro_iou": float("nan"), "target_reached": True},
                        {"verified": True, "foreground_macro_iou": True, "target_reached": True}):
            invalid = completed("invalid")
            invalid["audit"].update(changes)
            self.assertIs(loop.best_run([invalid, valid]), valid)
            self.assertFalse(loop.target_reached([invalid, valid]))
        reached = completed("reached", score=0.75)
        self.assertTrue(loop.target_reached([reached]))
        reached["state"] = "JOB_STATE_FAILED"
        self.assertFalse(loop.target_reached([reached]))

    def test_sixteen_failed_launches_exhaust_budget_without_consuming_queue(self):
        state = fixture_state()
        state["runs"] = [{"state": "JOB_STATE_FAILED"} for _ in range(16)]
        before = copy.deepcopy(state["queue"])
        self.assertIsNone(loop.choose_next(state, 14400))
        self.assertEqual(state["queue"], before)

    def test_replica_keeps_original_warm_start_but_new_architecture_clears_it(self):
        reference = {"uri": "gs://bucket/original.pt", "sha256": "original"}
        leader = completed("leader", score=0.6, lr=0.00003, initial_checkpoint=reference)
        state = fixture_state()
        state["queue"] = []
        labels = ["best-refine", "detail-896", "best-mild", "best-decay", "best-native"]
        state["runs"] = [leader, *[completed(label) for label in labels]]
        candidate = loop.choose_next(state, 4000)
        self.assertEqual(candidate["label"], "replicate-43")
        self.assertEqual(candidate["initial_checkpoint"], reference)
        self.assertEqual(candidate["lr"], leader["recipe"]["lr"])
        state["runs"] = [leader, completed("best-refine")]
        candidate = loop.choose_next(state, 4000)
        self.assertEqual(candidate["architecture"], loop.DETAIL)
        self.assertIsNone(candidate["initial_checkpoint"])


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "state.json"
        self.clock = [START]
        self.time_patch = patch.object(loop, "now", side_effect=lambda: self.clock[0])
        self.time_patch.start()
        self.addCleanup(self.time_patch.stop)
        self.events = patch.object(loop.Controller, "event")
        self.events.start()
        self.addCleanup(self.events.stop)

    def controller(self, state=None):
        loop.write_json(self.path, state or fixture_state())
        return loop.Controller(self.path)

    def test_replication_barrier_waits_for_later_winner_then_launches_both_seeds_together(self):
        state = fixture_state()
        late = full_completed("late", score=0.6, lr=0.0001)
        late.update(state="JOB_STATE_RUNNING")
        late.pop("audit")
        state.update(max_runs=4, queue=[replication_marker(), replication_marker(44)],
                     runs=[full_completed("early", score=0.51), late])
        controller = self.controller(state)
        polls = [0]

        def reconcile(run):
            if polls[0]:
                run.update(full_completed("late", score=0.6, lr=0.0001))

        def sleep(seconds):
            if polls[0] == 0:
                self.assertEqual(len(controller.state["runs"]), 2)
                self.assertEqual(controller.state["queue"], state["queue"])
                self.assertNotIn("replication_groups", controller.state)
                polls[0] += 1
            else:
                raise InterruptedError("fixture stop")

        replies = [json.dumps({"name": f"job/seed{seed}", "state": "JOB_STATE_PENDING"}) for seed in (43, 44)]
        with patch.object(controller, "reconcile", side_effect=reconcile), \
                patch.object(loop, "command", side_effect=replies) as cloud, \
                patch.object(loop.time, "sleep", side_effect=sleep):
            with self.assertRaisesRegex(InterruptedError, "fixture stop"):
                controller.run()
        self.assertEqual(cloud.call_count, 2)
        self.assertEqual(controller.state["queue"], [])
        group = controller.state["replication_groups"]["leader-seeds-43-44"]
        self.assertEqual(group["source_run_name"], "late")
        repeats = controller.state["runs"][2:]
        self.assertEqual([run["recipe"]["seed"] for run in repeats], [43, 44])
        self.assertTrue(all(run["recipe"]["lr"] == 0.0001 and run["state"] == "JOB_STATE_PENDING" for run in repeats))
        self.assertEqual({run["replication"]["source_recipe_sha256"] for run in repeats}, {group["recipe_sha256"]})

    def test_direct_replication_launch_cannot_bypass_pending_candidate_barrier(self):
        controller = self.controller()
        pending = full_completed("late", score=0.6)
        pending["state"] = "JOB_STATE_RUNNING"
        controller.state["runs"] = [full_completed("early", score=0.51), pending]
        before = copy.deepcopy(controller.state)
        with patch.object(loop, "command") as cloud, self.assertRaisesRegex(ValueError, "terminal candidates"):
            controller.launch(replication_marker())
        cloud.assert_not_called()
        self.assertEqual(controller.state, before)
        self.assertFalse((controller.root / "jobs").exists())

    def test_replication_records_failed_candidates_without_choosing_them_or_waiting_on_history(self):
        controller = self.controller()
        failed = full_completed("failed", score=0.99)
        failed["state"] = "JOB_STATE_FAILED"
        historical = full_completed("history", score=0.99)
        historical.update(historical=True, state="JOB_STATE_RUNNING")
        controller.state.update(runs=[full_completed("valid", score=0.51), failed, historical],
                                queue=[replication_marker()])
        marker = loop.choose_next(controller.state, 4000)
        resolved, metadata = controller.resolve_replication(marker)
        self.assertEqual(metadata["source_run_name"], "valid")
        outcomes = controller.state["replication_groups"][marker["replicate_best_full"]]["candidate_outcomes"]
        self.assertEqual(outcomes, [
            {"name": "valid", "state": "JOB_STATE_SUCCEEDED", "foreground_macro_iou": 0.51, "full_recipe_eligible": True},
            {"name": "failed", "state": "JOB_STATE_FAILED", "foreground_macro_iou": None, "full_recipe_eligible": False},
        ])

    def test_replication_resolves_at_launch_and_persists_frozen_group_before_submission(self):
        controller = self.controller()
        controller.state["runs"] = [full_completed("older", score=0.51)]
        controller.state["queue"] = [replication_marker()]
        marker = loop.choose_next(controller.state, 4000)
        # A newly audited leader after queue selection must still win at launch.
        leader = full_completed("new-lovasz", score=0.54, architecture="deeplabv3_resnet50",
                                epochs=60, warmup_epochs=3, loss="ce_lovasz", lovasz_weight=0.25,
                                backbone_checkpoint=backbone_reference(), auxiliary_loss_weight=0.4)
        controller.state["runs"].append(leader)
        controller.state["reference_runs"] = [full_completed("external-cached", score=0.99)]

        def cloud(args, **kwargs):
            saved = json.loads(self.path.read_text())
            group = saved["replication_groups"]["leader-seeds-43-44"]
            run = saved["runs"][-1]
            self.assertEqual(group["source_run_name"], "new-lovasz")
            self.assertEqual(run["state"], "SUBMIT_INTENT")
            self.assertNotIn("replicate_best_full", run["recipe"])
            self.assertEqual(run["replication"]["source_recipe_sha256"], group["recipe_sha256"])
            for key, value in leader["recipe"].items():
                if key not in {"label", "reason", "seed"}:
                    self.assertEqual(run["recipe"][key], value)
            return json.dumps({"name": "job/seed43", "state": "JOB_STATE_PENDING"})

        with patch.object(loop, "command", side_effect=cloud):
            controller.launch(marker)
        frozen = copy.deepcopy(controller.state["replication_groups"])
        # Restart from persisted state, then a still stronger completed run arrives.
        restarted = loop.Controller(self.path)
        restarted.state["runs"].append(full_completed("later-winner", score=0.60, lr=0.00001))
        with patch.object(loop, "command", return_value=json.dumps({"name": "job/seed44", "state": "JOB_STATE_PENDING"})):
            restarted.launch(replication_marker(44))
        self.assertEqual(restarted.state["replication_groups"], frozen)
        first, second = restarted.state["runs"][-3], restarted.state["runs"][-1]
        self.assertEqual(second["replication"]["source_run_name"], "new-lovasz")
        self.assertEqual({k:v for k,v in first["recipe"].items() if k not in {"label","reason","seed"}},
                         {k:v for k,v in second["recipe"].items() if k not in {"label","reason","seed"}})

    def test_replication_without_full_source_or_with_corrupt_group_cannot_submit(self):
        controller = self.controller()
        controller.state["runs"] = [completed("cached", score=0.9)]
        with patch.object(loop, "command") as cloud, self.assertRaisesRegex(ValueError, "fully completed"):
            controller.launch(replication_marker())
        cloud.assert_not_called()
        self.assertNotIn("replication_groups", controller.state)
        controller.state["replication_groups"] = {"leader-seeds-43-44": {"recipe": loop.recipe("bad", "bad")}}
        with patch.object(loop, "command") as cloud, self.assertRaisesRegex(ValueError, "provenance"):
            controller.launch(replication_marker())
        cloud.assert_not_called()

    def test_late_replication_keeps_full_recipe_and_explicitly_labels_runtime_limit(self):
        state = fixture_state()
        state.update(deadline=(START + timedelta(seconds=900)).isoformat(), queue=[replication_marker()])
        state["runs"] = [full_completed("full", epochs=80, warmup_epochs=3)]
        controller = self.controller(state)
        marker = loop.choose_next(controller.state, 900)
        self.assertNotIn("epochs", marker)
        with patch.object(loop, "command", return_value=json.dumps({"name": "job/seed43", "state": "JOB_STATE_PENDING"})):
            controller.launch(marker)
        run = controller.state["runs"][-1]
        self.assertEqual((run["recipe"]["epochs"], run["recipe"]["warmup_epochs"]), (80, 3))
        self.assertTrue(run["replication"]["late_budget_warning"])
        self.assertEqual(run["replication"]["max_duration_seconds"], 540)
        self.assertIn("count as a full replication only if all epochs complete", run["recipe"]["reason"])
        with patch.object(loop, "command") as cloud, self.assertRaisesRegex(ValueError, "persisted launch intent"):
            controller.launch(replication_marker())
        cloud.assert_not_called()

    def test_duplicate_replication_members_reject_queue_proposal_atomically(self):
        controller = self.controller()
        before = copy.deepcopy(controller.state["queue"])
        loop.write_json(controller.root / "next-plan.json", {"proposal_id": "duplicate-repeat", "queue": [replication_marker(), replication_marker()]})
        with patch.object(loop, "command") as cloud:
            self.assertFalse(controller.apply_next_plan())
        cloud.assert_not_called()
        self.assertEqual(controller.state["queue"], before)
        self.assertIn("group/seed", controller.state["proposal_error"])

    def source_proposal(self, controller):
        bootstrap = b"# reviewed new bootstrap\n"
        member_path = "holospex/ml/cloud/vertex_bootstrap.py"
        archive = controller.root / "source-v2.tar.gz"
        with tarfile.open(archive, "w:gz") as packed:
            member = tarfile.TarInfo(member_path)
            member.size = len(bootstrap)
            packed.addfile(member, io.BytesIO(bootstrap))
        digest = loop.sha(archive)
        receipt = {"path": str(archive), "sha256": digest, "bytes": archive.stat().st_size,
                   "uri": f"{loop.BUCKET}/bundles/{digest}/{archive.name}",
                   "members": [{"path": member_path, "sha256": hashlib.sha256(bootstrap).hexdigest(), "bytes": len(bootstrap)}]}
        loop.write_json(controller.root / "source-bundle.json", receipt)
        template = copy.deepcopy(controller.state["template"])
        container = template["workerPoolSpecs"][0]["containerSpec"]
        container["args"] = [bootstrap.decode()]
        for entry in container["env"]:
            if entry["name"] == "HOLOSPEX_SOURCE_URI":
                entry["value"] = receipt["uri"]
            if entry["name"] == "HOLOSPEX_SOURCE_SHA256":
                entry["value"] = receipt["sha256"]
        return {"proposal_id": "source-v2", "template": template,
                "queue": [loop.recipe("lovasz", "Test the foreground IoU surrogate", loss="ce_lovasz", lovasz_weight=0.25)]}

    def test_plan_is_applied_once_across_restart_without_mutating_running_jobs(self):
        state = fixture_state()
        state["runs"] = [{"name": "active", "state": "JOB_STATE_RUNNING", "config": "original.json",
                          "bundle_references": {"source": copy.deepcopy(state["source"])}}]
        controller = self.controller(state)
        proposal = self.source_proposal(controller)
        old_runs = copy.deepcopy(controller.state["runs"])
        loop.write_json(controller.root / "next-plan.json", proposal)
        with patch.object(loop, "command") as cloud:
            self.assertTrue(controller.apply_next_plan())
            cloud.assert_not_called()
        self.assertEqual(controller.state["runs"], old_runs)
        self.assertEqual(controller.state["deadline"], state["deadline"])
        self.assertEqual(controller.state["source"]["sha256"], loop.sha(controller.root / "source-v2.tar.gz"))
        self.assertEqual(controller.state["queue"][0]["loss"], "ce_lovasz")
        controller.state["queue"].clear()
        controller.save()
        restarted = loop.Controller(self.path)
        self.assertFalse(restarted.apply_next_plan())
        self.assertEqual(restarted.state["queue"], [])

    def test_invalid_proposals_reject_atomically_and_do_not_touch_cloud(self):
        for mutation in ("deadline", "data", "gpu", "bootstrap", "archive", "recipe"):
            with self.subTest(mutation=mutation):
                controller = self.controller()
                proposal = self.source_proposal(controller)
                container = proposal["template"]["workerPoolSpecs"][0]["containerSpec"]
                if mutation == "deadline":
                    proposal["deadline"] = (START + timedelta(days=1)).isoformat()
                elif mutation == "data":
                    next(entry for entry in container["env"] if entry["name"] == "HOLOSPEX_DATA_SHA256")["value"] = "9" * 64
                elif mutation == "gpu":
                    proposal["template"]["workerPoolSpecs"][0]["replicaCount"] = 2
                elif mutation == "bootstrap":
                    container["args"] = ["print('unbound source')"]
                elif mutation == "archive":
                    with (controller.root / "source-v2.tar.gz").open("ab") as stream:
                        stream.write(b"changed")
                else:
                    proposal["queue"][0]["split"] = "test"
                before = copy.deepcopy(controller.state)
                loop.write_json(controller.root / "next-plan.json", proposal)
                with patch.object(loop, "command") as cloud:
                    self.assertFalse(controller.apply_next_plan())
                    cloud.assert_not_called()
                for key in ("template", "source", "queue", "deadline", "runs"):
                    self.assertEqual(controller.state[key], before[key])
                self.assertIn("proposal_error", controller.state)
                self.assertFalse(controller.apply_next_plan())

    def test_queue_only_proposal_validates_bounds_and_cannot_reuse_consumed_id(self):
        controller = self.controller()
        proposal = {"proposal_id": "queue-only", "queue": [loop.recipe("one", "test", loss="ce_lovasz", lovasz_weight=0.5)]}
        path = controller.root / "next-plan.json"
        loop.write_json(path, proposal)
        self.assertTrue(controller.apply_next_plan())
        proposal["queue"][0]["lovasz_weight"] = 1.0
        loop.write_json(path, proposal)
        self.assertFalse(controller.apply_next_plan())
        self.assertEqual(controller.state["queue"][0]["lovasz_weight"], 0.5)
        proposal["proposal_id"] = "queue-invalid"
        proposal["queue"][0]["epochs"] = 241
        loop.write_json(path, proposal)
        self.assertFalse(controller.apply_next_plan())
        proposal["proposal_id"] = "queue-valid"
        proposal["queue"][0]["epochs"] = 40
        loop.write_json(path, proposal)
        self.assertTrue(controller.apply_next_plan())
        self.assertNotIn("proposal_error", controller.state)

    def test_launch_propagates_loss_weight_and_immutable_deadline_and_bundle_references(self):
        controller = self.controller()
        controller.state["template"]["workerPoolSpecs"][0]["containerSpec"]["env"].append(
            {"name": "HOLOSPEX_DEADLINE_UTC", "value": "2099-01-01T00:00:00+00:00"})
        with patch.object(loop, "command", return_value=json.dumps({"name": "job/1", "state": "JOB_STATE_PENDING"})):
            controller.launch(loop.recipe("lovasz", "test", loss="ce_lovasz", lovasz_weight=0.5))
            controller.launch(loop.recipe("ce", "test"))
        for index, weight in ((0, "0.5"), (1, "0.25")):
            run = controller.state["runs"][index]
            config = json.loads(Path(run["config"]).read_text())
            env = {entry["name"]: entry["value"] for entry in config["workerPoolSpecs"][0]["containerSpec"]["env"]}
            self.assertEqual(env["HOLOSPEX_DEADLINE_UTC"], controller.state["deadline"])
            self.assertEqual(env["HOLOSPEX_LOVASZ_WEIGHT"], weight)
            self.assertEqual(run["bundle_references"]["source"], controller.state["source"])

    def test_auxiliary_weight_override_and_legacy_default_replace_stale_template_value(self):
        controller = self.controller()
        controller.state["template"]["workerPoolSpecs"][0]["containerSpec"]["env"].append(
            {"name": "HOLOSPEX_AUXILIARY_LOSS_WEIGHT", "value": "7"})
        with patch.object(loop, "command", return_value=json.dumps({"name": "job/1", "state": "JOB_STATE_PENDING"})):
            controller.launch(loop.recipe("no-auxiliary", "test", auxiliary_loss_weight=0))
            controller.launch(loop.recipe("legacy", "test"))
        for index, expected in ((0, "0"), (1, "0.4")):
            run = controller.state["runs"][index]
            config = json.loads(Path(run["config"]).read_text())
            env = {entry["name"]: entry["value"] for entry in config["workerPoolSpecs"][0]["containerSpec"]["env"]}
            self.assertEqual(env["HOLOSPEX_AUXILIARY_LOSS_WEIGHT"], expected)
        self.assertNotIn("auxiliary_loss_weight", controller.state["runs"][1]["recipe"])

    def test_invalid_auxiliary_weight_cannot_persist_intent_or_spend(self):
        controller = self.controller()
        with patch.object(loop, "command") as cloud, self.assertRaisesRegex(ValueError, "auxiliary_loss_weight"):
            controller.launch(loop.recipe("invalid-auxiliary", "test", auxiliary_loss_weight=float("nan")))
        cloud.assert_not_called()
        self.assertEqual(controller.state["runs"], [])
        self.assertFalse((controller.root / "jobs").exists())

    def test_backbone_launch_binds_reference_and_clears_stale_initialization(self):
        controller = self.controller()
        env = controller.state["template"]["workerPoolSpecs"][0]["containerSpec"]["env"]
        env.extend({"name": key, "value": "stale"} for key in (
            "HOLOSPEX_INITIAL_CHECKPOINT_URI", "HOLOSPEX_INITIAL_CHECKPOINT_SHA256",
            "HOLOSPEX_BACKBONE_CHECKPOINT_URI", "HOLOSPEX_BACKBONE_CHECKPOINT_SHA256"))
        with patch.object(loop, "command", return_value=json.dumps({"name": "job/1", "state": "JOB_STATE_PENDING"})):
            controller.launch(loop.recipe("surgical", "test", architecture="deeplabv3_resnet50", backbone_checkpoint=backbone_reference()))
            controller.launch(loop.recipe("plain", "test"))
        for index in (0, 1):
            config = json.loads(Path(controller.state["runs"][index]["config"]).read_text())
            actual = {entry["name"]: entry["value"] for entry in config["workerPoolSpecs"][0]["containerSpec"]["env"]}
            self.assertFalse(any(key.startswith("HOLOSPEX_INITIAL_") for key in actual))
            if index == 0:
                self.assertEqual(actual["HOLOSPEX_BACKBONE_CHECKPOINT_URI"], backbone_reference()["uri"])
                self.assertEqual(actual["HOLOSPEX_BACKBONE_CHECKPOINT_SHA256"], backbone_reference()["sha256"])
            else:
                self.assertFalse(any(key.startswith("HOLOSPEX_BACKBONE_CHECKPOINT_") for key in actual))

    def test_direct_launch_rejects_invalid_backbone_before_writing_intent_or_spending(self):
        controller = self.controller()
        invalid = loop.recipe("wrong-backbone", "test", backbone_checkpoint=backbone_reference())
        with patch.object(loop, "command") as cloud, self.assertRaisesRegex(ValueError, "resnet50"):
            controller.launch(invalid)
        cloud.assert_not_called()
        self.assertEqual(controller.state["runs"], [])
        self.assertFalse((controller.root / "jobs").exists())

    def test_controller_lifecycle_records_interruption_without_cancelling_training(self):
        controller = self.controller()
        def interrupted():
            saved = json.loads(self.path.read_text())
            self.assertEqual(saved["controller_pid"], loop.os.getpid())
            self.assertEqual(saved["controller_status"], "running")
            raise KeyboardInterrupt()
        with patch.object(loop.Controller, "_run", side_effect=interrupted), patch.object(loop, "command") as cloud:
            with self.assertRaises(KeyboardInterrupt):
                controller.run()
            cloud.assert_not_called()
        saved = json.loads(self.path.read_text())
        self.assertIsNone(saved["controller_pid"])
        self.assertEqual(saved["controller_status"], "interrupted")
        self.assertIn("controller_stopped_at", saved)

    def test_opt_in_batch_closes_after_terminal_results_without_launching_or_cancelling(self):
        state = fixture_state()
        state.update(stop_when_exhausted=True, max_runs=2,
                     runs=[completed("one"), {"name": "two", "state": "JOB_STATE_FAILED",
                                              "recipe": loop.recipe("two", "failed trial")}])
        controller = self.controller(state)
        with patch.object(loop, "command") as cloud, patch.object(loop.time, "sleep") as sleep:
            controller.run()
        cloud.assert_not_called()
        sleep.assert_not_called()
        saved = json.loads(self.path.read_text())
        self.assertEqual(saved["stop_reason"], "experiment_batch_complete")
        self.assertEqual(saved["status"], "experiment_batch_complete")
        self.assertEqual(saved["controller_status"], "stopped")
        self.assertIsNone(saved["controller_pid"])
        self.assertEqual(saved["deadline"], state["deadline"])
        self.assertEqual(saved["queue"], state["queue"])
        self.assertFalse(loop.target_reached(saved["runs"]))

    def test_batch_waits_for_success_audit_then_closes_without_an_extra_poll(self):
        state = fixture_state()
        pending = completed("pending")
        pending.pop("audit")
        state.update(stop_when_exhausted=True, max_runs=1, runs=[pending])
        controller = self.controller(state)
        self.assertFalse(controller.batch_exhausted())

        def collect(run, *, timeout_seconds):
            run["audit"] = completed("audited")["audit"]

        with patch.object(loop, "command", return_value=json.dumps({"state": "JOB_STATE_SUCCEEDED"})) as cloud, \
                patch.object(controller, "collect_successful_run", side_effect=collect) as audit, \
                patch.object(loop.time, "sleep") as sleep:
            controller.state["runs"][0]["job_name"] = "job/one"
            controller.save()
            controller.run()
        self.assertEqual(cloud.call_count, 1)
        self.assertEqual(audit.call_count, 1)
        sleep.assert_not_called()
        self.assertEqual(controller.state["status"], "experiment_batch_complete")

    def test_batch_collection_failure_gets_bounded_final_retry_and_remains_unverified(self):
        state = fixture_state()
        pending = completed("pending")
        pending.pop("audit")
        pending["job_name"] = "job/one"
        state.update(stop_when_exhausted=True, max_runs=1, runs=[pending])
        controller = self.controller(state)
        with patch.object(loop, "command", return_value=json.dumps({"state": "JOB_STATE_SUCCEEDED"})) as cloud, \
                patch.object(controller, "collect_successful_run", side_effect=ValueError("artifact checksum mismatch")) as audit, \
                patch.object(loop.time, "sleep") as sleep:
            controller.run()
        self.assertEqual(cloud.call_count, 1)
        self.assertEqual(audit.call_count, 2)
        self.assertLessEqual(audit.call_args.kwargs["timeout_seconds"], loop.FINAL_COLLECTION_TIMEOUT_SECONDS)
        sleep.assert_not_called()
        self.assertEqual(controller.state["status"], "experiment_batch_complete")
        self.assertEqual(controller.state["runs"][0]["inspection_error"], "artifact checksum mismatch")
        self.assertNotIn("audit", controller.state["runs"][0])
        self.assertFalse(loop.target_reached(controller.state["runs"]))

    def test_exhausted_batch_keeps_waiting_when_flag_is_omitted_or_false(self):
        for value in (None, False):
            with self.subTest(flag=value):
                state = fixture_state()
                state.update(max_runs=1, runs=[completed("one")])
                if value is not None:
                    state["stop_when_exhausted"] = value
                controller = self.controller(state)
                with patch.object(loop, "command") as cloud, \
                        patch.object(loop.time, "sleep", side_effect=InterruptedError("fixture stop")):
                    with self.assertRaises(InterruptedError):
                        controller.run()
                cloud.assert_not_called()
                self.assertEqual(controller.state["status"], "running")
                self.assertNotIn("stop_reason", controller.state)

    def test_batch_requires_all_owned_jobs_terminal_and_audits_valid_before_early_stop(self):
        for status in ("SUBMIT_INTENT", "JOB_STATE_PENDING", "JOB_STATE_RUNNING"):
            with self.subTest(state=status):
                state = fixture_state()
                run = completed("one")
                run.update(state=status, inspection_error="unresolved")
                state.update(stop_when_exhausted=True, max_runs=1, runs=[run])
                self.assertFalse(self.controller(state).batch_exhausted())
        state = fixture_state()
        run = completed("one")
        run["audit"]["verified"] = False
        state.update(stop_when_exhausted=True, max_runs=1, runs=[run])
        self.assertFalse(self.controller(state).batch_exhausted())
        state.update(max_runs=2, runs=[completed("one")])
        self.assertFalse(self.controller(state).batch_exhausted())

    def test_batch_completion_does_not_override_deadline_or_target(self):
        for reason in ("deadline_reached", "target_reached"):
            with self.subTest(reason=reason):
                state = fixture_state()
                state.update(stop_when_exhausted=True, max_runs=1,
                             runs=[completed("one", score=0.8 if reason == "target_reached" else 0.5)])
                if reason == "deadline_reached":
                    state["deadline"] = START.isoformat()
                controller = self.controller(state)
                with patch.object(loop, "command") as cloud:
                    controller.run()
                cloud.assert_not_called()
                self.assertEqual(controller.state["stop_reason"], reason)

    def test_reference_is_visible_in_status_but_cannot_trigger_target_or_cancellation(self):
        state = fixture_state()
        state.update(reference_runs=[completed("historical-reference", score=0.8)], queue=[],
                     deadline=(START + timedelta(seconds=400)).isoformat())
        controller = self.controller(state)
        with patch.object(loop, "command") as cloud, patch.object(loop.time, "sleep", side_effect=InterruptedError("fixture stop")):
            with self.assertRaises(InterruptedError):
                controller.run()
            cloud.assert_not_called()
        self.assertEqual(controller.state["status"], "running")
        self.assertNotIn("stop_reason", controller.state)
        self.assertIn("Current verified leader: historical-reference.", (controller.root / "STATUS.md").read_text())
        self.assertEqual(controller.state["runs"], [])
        # Even an incorrectly nonterminal external reference is not owned.
        controller.state["reference_runs"][0]["state"] = "JOB_STATE_RUNNING"
        with patch.object(loop, "command") as cloud:
            self.assertTrue(controller.stop_owned_jobs())
            cloud.assert_not_called()

    def test_ambiguous_create_is_persisted_and_restart_never_repeats_it(self):
        controller = self.controller()
        with patch.object(loop, "command", side_effect=subprocess.TimeoutExpired("create", 120)) as cloud:
            with self.assertRaises(subprocess.TimeoutExpired):
                controller.launch(loop.recipe("timeout", "test"))
        self.assertEqual(cloud.call_count, 1)
        self.assertEqual(json.loads(self.path.read_text())["runs"][0]["state"], "SUBMIT_INTENT")
        with patch.object(loop, "command", return_value="[]") as cloud, \
                patch.object(loop.time, "sleep", side_effect=InterruptedError("stop fixture")):
            with self.assertRaises(InterruptedError):
                loop.Controller(self.path).run()
        self.assertEqual(cloud.call_count, 1)
        self.assertEqual(cloud.call_args.args[0][:3], ["ai", "custom-jobs", "list"])
        self.assertIn("inspection_error", json.loads(self.path.read_text())["runs"][0])

    def test_deadline_reconciles_unknown_intent_and_waits_for_cancellation_confirmation(self):
        state = fixture_state()
        state["deadline"] = START.isoformat()
        state["runs"] = [{"name": "owned", "state": "SUBMIT_INTENT", "recipe": loop.recipe("owned", "test")}]
        controller = self.controller(state)
        calls = []
        cancelled = [False]

        def cloud(args, **kwargs):
            calls.append(args)
            if args[2] == "list":
                return json.dumps([{"name": "projects/p/locations/r/customJobs/123"}])
            if args[2] == "describe":
                return json.dumps({"state": "JOB_STATE_CANCELLED" if cancelled[0] else "JOB_STATE_RUNNING"})
            if args[2] == "cancel":
                cancelled[0] = True
                return ""
            self.fail(f"Unexpected cloud mutation: {args}")

        def sleep(_):
            saved = json.loads(self.path.read_text())
            self.assertEqual(saved["status"], "stopping")
            self.assertIn("cancellation_requested_at", saved["runs"][0])

        with patch.object(loop, "command", side_effect=cloud), patch.object(loop.time, "sleep", side_effect=sleep):
            controller.run()
        self.assertEqual(controller.state["status"], "deadline_reached")
        self.assertEqual(controller.state["runs"][0]["state"], "JOB_STATE_CANCELLED")
        self.assertEqual(sum(args[2] == "cancel" for args in calls), 1)
        self.assertFalse(any(args[2] == "create" for args in calls))

    def test_stalled_collection_is_bounded_then_owned_job_is_cancelled(self):
        state = fixture_state()
        state.update(deadline=(START + timedelta(seconds=90)).isoformat(), queue=[])
        state["runs"] = [
            {"name": "finished", "job_name": "job/finished", "state": "JOB_STATE_RUNNING", "recipe": loop.recipe("finished", "test"), "output_uri": loop.BUCKET + "/runs/finished"},
            {"name": "active", "job_name": "job/active", "state": "JOB_STATE_RUNNING", "recipe": loop.recipe("active", "test")},
        ]
        controller = self.controller(state)
        monotonic, cancelled, calls = [0.0], [False], []

        def cloud(args, **kwargs):
            calls.append(args)
            if args[:2] == ["storage", "ls"]:
                self.clock[0] += timedelta(seconds=10)
                monotonic[0] += 10
                return loop.BUCKET + "/runs/finished/attempts/a/status/completed.json\n"
            if args[2] == "describe":
                value = "JOB_STATE_SUCCEEDED" if args[3] == "job/finished" else ("JOB_STATE_CANCELLED" if cancelled[0] else "JOB_STATE_RUNNING")
                return json.dumps({"state": value})
            if args[2] == "cancel":
                self.assertEqual(args[3], "job/active")
                cancelled[0] = True
                return ""
            self.fail(f"Unexpected cloud call: {args}")

        def stalled(uri, destination, *, timeout_seconds):
            self.assertEqual(timeout_seconds, 50)  # 90 - 30 reserve - 10 discovery.
            self.clock[0] += timedelta(seconds=timeout_seconds)
            monotonic[0] += timeout_seconds
            raise subprocess.TimeoutExpired("download", timeout_seconds)

        def sleep(seconds):
            self.clock[0] += timedelta(seconds=seconds)
            monotonic[0] += seconds

        with patch.object(loop, "command", side_effect=cloud), patch.object(loop, "collect", side_effect=stalled) as download, \
                patch.object(loop.time, "monotonic", side_effect=lambda: monotonic[0]), patch.object(loop.time, "sleep", side_effect=sleep), \
                patch.object(controller, "audit_after_stopping"):
            controller.run()
        self.assertEqual(download.call_count, 1)
        self.assertTrue(cancelled[0])
        self.assertEqual(controller.state["status"], "deadline_reached")
        self.assertFalse(any(args[:3] == ["ai", "custom-jobs", "create"] for args in calls))

    def test_late_success_is_audited_only_after_all_owned_compute_is_terminal(self):
        state = fixture_state()
        state.update(deadline=START.isoformat(), queue=[])
        state["runs"] = [{"name": "late", "job_name": "job/late", "state": "JOB_STATE_RUNNING", "recipe": loop.recipe("late", "test")}]
        controller = self.controller(state)

        def collected(run, *, timeout_seconds):
            self.assertTrue(all(item["state"] in loop.TERMINAL for item in controller.state["runs"]))
            self.assertGreater(timeout_seconds, 0)
            self.assertLessEqual(timeout_seconds, loop.FINAL_COLLECTION_TIMEOUT_SECONDS)
            run["audit"] = {"verified": True, "foreground_macro_iou": 0.76, "checkpoint_sha256": "test"}

        with patch.object(loop, "command", return_value=json.dumps({"state": "JOB_STATE_SUCCEEDED"})) as cloud, \
                patch.object(controller, "collect_successful_run", side_effect=collected) as audit:
            controller.run()
        self.assertEqual(audit.call_count, 1)
        self.assertTrue(loop.target_reached(controller.state["runs"]))
        self.assertEqual(controller.state["status"], "deadline_reached")
        self.assertTrue(all(call.args[0][2] == "describe" for call in cloud.call_args_list))

    def test_final_collection_has_one_total_budget_and_records_uncollected_successes(self):
        controller = self.controller()
        controller.state["runs"] = [{"name": name, "state": "JOB_STATE_SUCCEEDED", "recipe": loop.recipe(name, "test")}
                                     for name in ("one", "two")]
        elapsed = [0.0]

        def stalled(run, *, timeout_seconds):
            elapsed[0] += timeout_seconds
            raise subprocess.TimeoutExpired("download", timeout_seconds)

        with patch.object(loop.time, "monotonic", side_effect=lambda: elapsed[0]), \
                patch.object(controller, "collect_successful_run", side_effect=stalled) as audit:
            controller.audit_after_stopping()
        self.assertEqual(audit.call_count, 1)
        self.assertTrue(all("inspection_error" in run and "audit" not in run for run in controller.state["runs"]))
        self.assertIn("collect manually", controller.state["runs"][1]["inspection_error"])

    def test_submit_crossing_deadline_cannot_launch_second_parallel_job(self):
        state = fixture_state()
        state["deadline"] = (START + timedelta(seconds=900)).isoformat()
        controller = self.controller(state)
        calls = []

        def cloud(args, **kwargs):
            calls.append(args)
            if args[2] == "create":
                self.clock[0] += timedelta(seconds=901)
                return json.dumps({"name": "projects/p/locations/r/customJobs/123", "state": "JOB_STATE_PENDING"})
            if args[2] == "describe":
                return json.dumps({"state": "JOB_STATE_EXPIRED"})
            self.fail(f"Unexpected cloud mutation: {args}")

        with patch.object(loop, "command", side_effect=cloud), patch.object(loop.time, "sleep"):
            controller.run()
        self.assertEqual(sum(args[2] == "create" for args in calls), 1)
        self.assertEqual(controller.state["status"], "deadline_reached")

    def test_inactive_session_never_resumes_or_calls_cloud(self):
        for status in ("target_reached", "deadline_reached", "experiment_batch_complete", "paused", "max_runs_reached"):
            state = fixture_state()
            state["status"] = status
            controller = self.controller(state)
            with patch.object(loop, "command") as cloud, self.assertRaisesRegex(ValueError, "inactive"):
                controller.run()
            cloud.assert_not_called()
            self.assertEqual(json.loads(self.path.read_text())["status"], status)

    def test_invalid_parallel_and_launch_caps_are_rejected(self):
        for key, value in (("max_parallel", 3), ("max_parallel", True), ("max_runs", 17), ("max_runs", 0)):
            state = fixture_state()
            state[key] = value
            with self.assertRaisesRegex(ValueError, key):
                self.controller(state)

    def test_stop_when_exhausted_requires_explicit_boolean(self):
        for value in (0, 1, "true", None, [], {}):
            state = fixture_state()
            state["stop_when_exhausted"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "stop_when_exhausted"):
                self.controller(state)

    def test_direct_launch_cannot_exceed_parallel_or_total_cap(self):
        for runs in ([{"state": "JOB_STATE_RUNNING"} for _ in range(2)],
                     [{"state": "JOB_STATE_FAILED"} for _ in range(16)]):
            state = fixture_state()
            state["runs"] = runs
            controller = self.controller(state)
            with patch.object(loop, "command") as cloud, self.assertRaisesRegex(RuntimeError, "guard"):
                controller.launch(loop.recipe("excess", "test"))
            cloud.assert_not_called()

    def test_concurrent_controller_cannot_acquire_same_state(self):
        controller = self.controller()
        with (controller.root / "controller.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with patch.object(loop, "command") as cloud, self.assertRaisesRegex(RuntimeError, "already owns"):
                controller.run()
            cloud.assert_not_called()


if __name__ == "__main__":
    unittest.main()
