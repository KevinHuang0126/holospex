"""Reviewed positives must never turn unknown tissue into background targets.

All images, reviewers and decisions here are synthetic test fixtures. No model
weights, real reviews, cloud jobs or dataset downloads are needed.
"""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

HAS_DATA = importlib.util.find_spec("numpy") is not None and importlib.util.find_spec("PIL") is not None
if HAS_DATA:
    import numpy as np


REVIEW_DIRECTORY = Path(__file__).resolve().parents[1] / "review"
sys.path.insert(0, str(REVIEW_DIRECTORY))
try:
    import review_io
    if HAS_DATA:
        from prepare_training import prepare_training
finally:
    sys.path.pop(0)

HAS_TRAINING = HAS_DATA and importlib.util.find_spec("torch") and importlib.util.find_spec("torchvision")
WHEN = "2026-09-13T18:00:00Z"
CLASSES = [
    {"index": index, "sourceId": source, "structureId": structure}
    for index, (source, structure) in enumerate([
        (0, "background"), (5, "gallbladder"), (4, "cystic_duct"),
        (3, "cystic_artery"), (1, "cystic_plate"),
        (2, "hepatocystic_triangle_dissection"), (6, "tool"),
    ])
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2) + "\n")


def fixture(root, *, width=32, height=24, review_case=9, classes=None, mode="standard"):
    from PIL import Image

    classes = copy.deepcopy(classes or CLASSES)
    source_ids = {item["structureId"]: item["sourceId"] for item in classes}
    samples = []
    for video, split in ((1, "train"), (2, "val"), (3, "test")):
        image_path, mask_path = root / f"base-{video}.jpg", root / f"base-{video}.png"
        Image.new("RGB", (8, 8), (80, 100, 120)).save(image_path)
        mask = np.tile(np.array([item["sourceId"] for item in classes] + [255], dtype=np.uint8), (8, 1))
        Image.fromarray(mask).save(mask_path)
        samples.append({"split": split, "imagePath": str(image_path), "maskPath": str(mask_path),
                        "videoId": video, "frameNumber": video * 10, "timestampMs": video * 400.0})
    base = {"schemaVersion": "1.0.0", "dataset": "synthetic-review-test", "root": str(root),
            "classes": classes, "samples": samples, "ignoreIndex": 255, "ignoreSourceIds": [255],
            "report": {"fps": 25.0, "annotationSource": "Synthetic fixture; not real anatomy"}}
    base_path = root / "base.json"
    write_json(base_path, base)

    def rectangle(y1, y2, x1, x2):
        value = np.zeros((height, width), dtype=np.uint8)
        value[y1:y2, x1:x2] = 1
        return value

    duct = rectangle(2, 8, 2, 10)
    duct[4, 4] = 0  # A genuine hole must remain unknown.
    masks = [duct, rectangle(2, 8, 8, 14), rectangle(5, 11, 11, 17),
             rectangle(12, 16, 2, 7), rectangle(12, 16, 12, 18),
             rectangle(18, 20, 2, 5), rectangle(18, 20, 7, 10),
             rectangle(18, 20, 12, 15), rectangle(18, 20, 17, 20)]
    names = ["cystic_duct", "cystic_duct", "cystic_artery", "cystic_plate",
             "hepatocystic_triangle_dissection", "cystic_plate", "cystic_artery",
             "cystic_duct", "hepatocystic_triangle_dissection"]
    states = ["accepted", "accepted", "accepted", "edited", "accepted", "pending", "rejected", "needs_expert"]
    edited = masks[3].copy()
    edited[12:16, 7] = 1
    if mode == "all_conflict":
        masks = [rectangle(2, 8, 2, 10) for _ in masks]
        states[3] = "accepted"
    elif mode == "single_pixel":
        masks = [np.zeros((height, width), dtype=np.uint8)]
        masks[0][0, 2] = 1  # Native pixel omitted by 854 -> 672 nearest sampling.
        names, states = ["cystic_duct"], ["accepted"]

    image_id = f"{review_case}_90"
    image_path = root / "images" / f"{image_id}.jpg"
    image_path.parent.mkdir()
    Image.new("RGB", (width, height), (120, 70, 50)).save(image_path)
    candidates = []
    for number, (mask, structure) in enumerate(zip(masks, names), start=1):
        rle = review_io.encode_rle(mask)
        candidates.append({"id": f"endoscapes-box-{number}", "sourceAnnotationId": number,
                           "structureId": structure, "bboxXYWH": [0, 0, width, height],
                           "source": "model_generated", "mask": rle, "maskSha256": review_io.mask_sha256(rle),
                           "proposalScore": 0.7, "scoreMeaning": "Synthetic model quality, not anatomy confidence", "warnings": []})
    bundle = {"formatVersion": "1.0.0", "artifactType": "candidate_mask_bundle", "bundleId": "synthetic-training-review",
              "createdAt": WHEN, "sourceManifestSha256": digest(base_path),
              "selection": {"description": "Synthetic new TRAIN case fixture"},
              "generator": {"model": "synthetic-test-double", "realModelOutput": False},
              "classes": [{"structureId": name, "label": name, "color": "#33aabb"} for name in sorted(set(names))],
              "images": [{"id": image_id, "videoId": str(review_case), "frameNumber": 90, "split": "train",
                          "width": width, "height": height, "imagePath": image_path.relative_to(root).as_posix(),
                          "imageSha256": digest(image_path), "candidates": candidates}]}
    bundle_path = root / "bundle.json"
    write_json(bundle_path, bundle)
    decisions = []
    for candidate, state in zip(candidates, states):
        mask = review_io.encode_rle(edited) if state == "edited" else copy.deepcopy(candidate["mask"])
        decisions.append({"candidateId": candidate["id"], "imageId": image_id, "imageSha256": digest(image_path),
                          "proposalMaskSha256": candidate["maskSha256"], "decision": state, "mask": mask,
                          "notes": "Original boundary error was corrected" if state == "edited" else
                                   "Uncertain synthetic boundary" if state in {"rejected", "needs_expert"} else "",
                          "reviewedAt": None if state == "pending" else WHEN, "reviewMilliseconds": 2000.0})
    review = {"formatVersion": "1.0.0", "artifactType": "candidate_mask_review", "bundleId": bundle["bundleId"],
              "bundleSha256": digest(bundle_path), "reviewer": {"name": "Synthetic Test Reviewer", "reviewScope": "anatomy"},
              "exportedAt": WHEN, "decisions": decisions}
    review_path = root / "review.json"
    write_json(review_path, review)
    resolution = {"formatVersion": "1.0.0", "artifactType": "review_training_resolution", "bundleId": bundle["bundleId"],
                  "bundleSha256": digest(bundle_path), "reviewSha256": digest(review_path), "baseManifestSha256": digest(base_path),
                  "resolvedAt": WHEN, "resolvedBy": "project_lead", "overlapPolicy": "ignore_conflicts", "unknownPixelPolicy": "ignore",
                  "noteClarifications": [{"candidateId": item["candidateId"], "interpretation": "described_original_proposal",
                                           "reviewedMaskConfirmed": True} for item in decisions if item["decision"] == "edited"],
                  "authorization": "prepare_partial_training_targets",
                  "evidence": {"source": "user_message", "text": "Synthetic fixture authorization; ignore overlaps and prepare partial labels."}}
    resolution_path = root / "resolution.json"
    write_json(resolution_path, resolution)

    expected = np.full((height, width), 255, dtype=np.uint8)
    if mode == "standard":
        expected[(masks[0] | masks[1]).astype(bool)] = source_ids["cystic_duct"]
        expected[masks[2].astype(bool)] = source_ids["cystic_artery"]
        expected[edited.astype(bool)] = source_ids["cystic_plate"]
        expected[masks[4].astype(bool)] = source_ids["hepatocystic_triangle_dissection"]
        expected[5:8, 11:14] = 255  # Different-class overlap is explicitly unknown.
    elif mode == "single_pixel":
        expected[0, 2] = source_ids["cystic_duct"]
    return {"root": root, "base": base, "bundle": bundle, "review": review, "resolution": resolution,
            "base_path": base_path, "bundle_path": bundle_path, "review_path": review_path, "resolution_path": resolution_path,
            "image_path": image_path, "image_id": image_id, "expected": expected}


def prepare(value, output=None):
    output = output or value["root"] / "prepared"
    result = prepare_training(value["bundle_path"], value["review_path"], value["base_path"], value["resolution_path"], output)
    manifest = json.loads(Path(result["manifestPath"]).read_text())
    return result, manifest


@unittest.skipUnless(HAS_DATA, "Install ml[data] for reviewed target preparation")
class ReviewedTrainingPreparationTests(unittest.TestCase):
    def test_binary_foreground_becomes_source_ids_with_unknown_holes_and_conflicts(self):
        from PIL import Image
        from holospex_ml.dataset import encode_mask

        with tempfile.TemporaryDirectory() as temporary:
            value = fixture(Path(temporary))
            imported = value["root"] / "binary-import"
            receipt = review_io.import_review(value["bundle_path"], value["review_path"], imported)
            first = receipt["eligibleMasks"][0]
            binary = np.asarray(Image.open(imported / first["maskPath"]))
            self.assertEqual(int(binary[3, 3]), 255)  # Import255 means reviewed foreground.
            self.assertEqual(int(binary[0, 0]), 0)  # Import0 means UNKNOWN, not background.
            _, manifest = prepare(value)
            added = manifest["samples"][len(value["base"]["samples"]):]
            self.assertEqual(len(added), 1)
            target = np.asarray(Image.open(added[0]["maskPath"]))
            np.testing.assert_array_equal(target, value["expected"])
            self.assertEqual(int(target[3, 3]), 4)  # Duct sourceId, not model index2 or import255.
            self.assertEqual(int(target[3, 9]), 4)  # Same-class overlap stays a union.
            self.assertEqual(int(target[4, 4]), 255)  # A hole stays unknown.
            self.assertEqual(int(target[6, 12]), 255)  # Duct/artery conflict.
            self.assertEqual(int(target[13, 7]), 1)  # Explicit edited plate boundary is retained.
            self.assertNotIn(0, np.unique(target))
            encoded = encode_mask(target, manifest["classes"], manifest["ignoreSourceIds"])
            self.assertEqual([int(encoded[y, x]) for y, x in [(3, 3), (9, 15), (13, 7), (13, 14)]], [2, 3, 4, 5])
            self.assertEqual(set(np.unique(encoded)), {2, 3, 4, 5, 255})

    def test_mapping_comes_from_manifest_not_hardcoded_endoscapes_or_model_indices(self):
        from PIL import Image

        classes = copy.deepcopy(CLASSES)
        for entry, source_id in zip(classes[1:], [11, 17, 23, 31, 43, 59]):
            entry["sourceId"] = source_id
        with tempfile.TemporaryDirectory() as temporary:
            value = fixture(Path(temporary), classes=classes)
            _, manifest = prepare(value)
            actual = np.asarray(Image.open(manifest["samples"][-1]["maskPath"]))
            np.testing.assert_array_equal(actual, value["expected"])
            self.assertEqual(set(np.unique(actual)), {17, 23, 31, 43, 255})

    def test_excluded_and_missing_decisions_never_supply_pixels(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as temporary:
            value = fixture(Path(temporary))
            _, manifest = prepare(value)
            target = np.asarray(Image.open(manifest["samples"][-1]["maskPath"]))
            # Pending, rejected, expert-referral, and absent decisions all own
            # different pixels here, so each exclusion is independently checked.
            for x in (3, 8, 13, 18):
                with self.subTest(x=x):
                    self.assertEqual(int(target[18, x]), 255)
            self.assertEqual(int(target[0, 0]), 255)

    def test_training_exclusions_preserve_review_and_imported_approvals(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as temporary:
            value = fixture(Path(temporary))
            original_review = value["review_path"].read_bytes()
            exclusions = [
                {"candidateId": "endoscapes-box-1", "reason": "Accepted note contradicts anatomy identity."},
                {"candidateId": "endoscapes-box-4", "reason": "Edited boundary needs further review."},
            ]
            value["resolution"]["excludedCandidates"] = exclusions
            write_json(value["resolution_path"], value["resolution"])
            receipt = review_io.import_review(value["bundle_path"], value["review_path"], value["root"] / "import")
            self.assertEqual(len(receipt["eligibleMasks"]), 5)
            result, manifest = prepare(value)
            target = np.asarray(Image.open(manifest["samples"][-1]["maskPath"]))
            expected = value["expected"].copy()
            expected[2:8, 2:8] = 255  # Only the excluded first duct covered these pixels.
            expected[12:16, 2:8] = 255  # Excluding an edited mask also removes its corrections.
            np.testing.assert_array_equal(target, expected)
            self.assertEqual(int(target[3, 9]), 4)  # The second approved duct still supplies its overlap.
            self.assertEqual(int(target[6, 12]), 255)  # Remaining duct/artery conflict is still ignored.
            summary = result["summary"]
            self.assertEqual(summary["eligibleCandidateCount"], 5)
            self.assertEqual(summary["trainingEligibleCandidateCount"], 3)
            self.assertEqual(summary["excludedCandidateCount"], 2)
            self.assertEqual(summary["excludedCandidates"], exclusions)
            preparation = json.loads(Path(result["preparationPath"]).read_text())
            record = preparation["images"][0]
            self.assertEqual(record["excludedCandidateCount"], 2)
            by_id = {candidate["candidateId"]: candidate for candidate in record["candidates"]}
            for number, original_decision in ((1, "accepted"), (4, "edited")):
                candidate = by_id[f"endoscapes-box-{number}"]
                self.assertEqual(candidate["decision"], original_decision)
                self.assertTrue(candidate["reviewEligible"])
                self.assertTrue(candidate["trainingExcluded"])
                self.assertFalse(candidate["eligible"])
                self.assertTrue(candidate["exclusionReason"])
            self.assertEqual(manifest["reviewedPartialProvenance"]["excludedCandidates"], exclusions)
            self.assertFalse(preparation["humanDecisionsChanged"])
            self.assertEqual(value["review_path"].read_bytes(), original_review)
            snapshot = Path(result["manifestPath"]).parent / "provenance/source-review.json"
            self.assertEqual(snapshot.read_bytes(), original_review)

    def test_excluding_other_class_preserves_remaining_positive_overlap(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as temporary:
            value = fixture(Path(temporary))
            value["resolution"]["excludedCandidates"] = [
                {"candidateId": "endoscapes-box-3", "reason": "Artery approval is disputed."},
            ]
            write_json(value["resolution_path"], value["resolution"])
            _, manifest = prepare(value)
            target = np.asarray(Image.open(manifest["samples"][-1]["maskPath"]))
            expected = value["expected"].copy()
            expected[5:11, 11:17] = 255
            expected[5:8, 11:14] = 4  # A retained duct approval supplies these formerly conflicting pixels.
            np.testing.assert_array_equal(target, expected)
            self.assertEqual(int(target[9, 15]), 255)  # Excluded-only pixels are unknown, never background.

    def test_invalid_training_exclusions_fail_before_creating_output(self):
        invalid_exclusions = [
            None,
            {},
            [{"candidateId": "endoscapes-box-999", "reason": "Unknown candidate."}],
            [{"candidateId": "endoscapes-box-1", "reason": " "}],
            [{"candidateId": "endoscapes-box-1", "reason": 5}],
            [{"candidateId": "endoscapes-box-1", "reason": "First reason."},
             {"candidateId": "endoscapes-box-1", "reason": "Duplicate reason."}],
            [{"candidateId": "endoscapes-box-1", "reason": "Reason.", "decision": "rejected"}],
        ] + [[{"candidateId": f"endoscapes-box-{number}", "reason": "Not an eligible approval."}]
             for number in (6, 7, 8, 9)]
        for exclusions in invalid_exclusions:
            with self.subTest(exclusions=exclusions), tempfile.TemporaryDirectory() as temporary:
                value = fixture(Path(temporary))
                value["resolution"]["excludedCandidates"] = exclusions
                write_json(value["resolution_path"], value["resolution"])
                output = value["root"] / "not-created" / "prepared"
                with self.assertRaises(ValueError):
                    prepare(value, output)
                self.assertFalse(output.parent.exists())

    def test_excluding_every_approved_candidate_cannot_publish_empty_supervision(self):
        with tempfile.TemporaryDirectory() as temporary:
            value = fixture(Path(temporary))
            value["resolution"]["excludedCandidates"] = [
                {"candidateId": decision["candidateId"], "reason": "Awaiting annotation clarification."}
                for decision in value["review"]["decisions"] if decision["decision"] in {"accepted", "edited"}
            ]
            write_json(value["resolution_path"], value["resolution"])
            with self.assertRaisesRegex(ValueError, "after training exclusions"):
                prepare(value)
            self.assertFalse((value["root"] / "prepared").exists())

    def test_base_samples_files_and_heldout_splits_are_preserved_exactly(self):
        with tempfile.TemporaryDirectory() as temporary:
            value = fixture(Path(temporary))
            inputs = [value["base_path"], value["bundle_path"], value["review_path"], value["resolution_path"]]
            files = inputs + [Path(row[key]) for row in value["base"]["samples"] for key in ("imagePath", "maskPath")]
            before = {str(path): path.read_bytes() for path in files}
            result, manifest = prepare(value)
            self.assertEqual(manifest["samples"][:3], value["base"]["samples"])
            for split in ("val", "test"):
                self.assertEqual([row for row in manifest["samples"] if row["split"] == split],
                                 [row for row in value["base"]["samples"] if row["split"] == split])
            self.assertEqual(manifest["classes"], value["base"]["classes"])
            self.assertEqual(manifest["samples"][-1]["split"], "train")
            self.assertEqual(manifest["samples"][-1]["videoId"], 9)
            self.assertEqual(manifest["samples"][-1]["timestampMs"], 3600.0)
            for path in files:
                self.assertEqual(path.read_bytes(), before[str(path)])
            output = Path(result["manifestPath"]).parent
            for name, original in [("source-bundle", value["bundle_path"]), ("source-review", value["review_path"]),
                                   ("base-manifest", value["base_path"]), ("resolution", value["resolution_path"])]:
                self.assertEqual((output / "provenance" / f"{name}.json").read_bytes(), original.read_bytes())

    def test_existing_train_val_or_test_case_cannot_be_added_under_another_frame(self):
        for case in (1, 2, 3, "01"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                value = fixture(Path(temporary), review_case=case)
                output = value["root"] / "not-created" / "prepared"
                with self.assertRaises(ValueError):
                    prepare(value, output)
                self.assertFalse(output.exists())

    def test_resolution_hashes_and_source_identity_fail_before_output(self):
        for field in ("bundleId", "bundleSha256", "reviewSha256", "baseManifestSha256"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                value = fixture(Path(temporary))
                value["resolution"][field] = "different" if field == "bundleId" else "f" * 64
                write_json(value["resolution_path"], value["resolution"])
                output = value["root"] / "not-created" / "prepared"
                with self.assertRaises(ValueError):
                    prepare(value, output)
                self.assertFalse(output.parent.exists())
        with tempfile.TemporaryDirectory() as temporary:
            value = fixture(Path(temporary))
            value["image_path"].write_bytes(b"stale source image")
            with self.assertRaises(ValueError):
                prepare(value)
            self.assertFalse((value["root"] / "prepared").exists())

    def test_policy_and_clarification_are_explicit_not_silently_inferred(self):
        for mutate in (
            lambda record: record.update(overlapPolicy="last_mask_wins"),
            lambda record: record.update(unknownPixelPolicy="background"),
            lambda record: record["noteClarifications"].append(copy.deepcopy(record["noteClarifications"][0])),
            lambda record: record["noteClarifications"][0].update(candidateId="endoscapes-box-6"),
            lambda record: record["noteClarifications"][0].update(reviewedMaskConfirmed=False),
        ):
            with self.subTest(mutate=mutate), tempfile.TemporaryDirectory() as temporary:
                value = fixture(Path(temporary))
                mutate(value["resolution"])
                write_json(value["resolution_path"], value["resolution"])
                with self.assertRaises(ValueError):
                    prepare(value)
                self.assertFalse((value["root"] / "prepared").exists())

    def test_all_conflicting_or_noneligible_reviews_cannot_publish_empty_supervision(self):
        with tempfile.TemporaryDirectory() as temporary:
            value = fixture(Path(temporary), mode="all_conflict")
            with self.assertRaises(ValueError):
                prepare(value)
            self.assertFalse((value["root"] / "prepared").exists())
        for mode in ("technical", "pending"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                value = fixture(Path(temporary))
                if mode == "technical":
                    value["review"]["reviewer"]["reviewScope"] = "technical"
                else:
                    for decision in value["review"]["decisions"]:
                        decision.update(decision="pending", reviewedAt=None)
                write_json(value["review_path"], value["review"])
                value["resolution"].update(reviewSha256=digest(value["review_path"]), noteClarifications=[])
                write_json(value["resolution_path"], value["resolution"])
                with self.assertRaises(ValueError):
                    prepare(value)
                self.assertFalse((value["root"] / "prepared").exists())

    def test_all_ignore_image_is_omitted_and_audited_while_positive_image_is_retained(self):
        with tempfile.TemporaryDirectory() as temporary:
            value = fixture(Path(temporary))
            second = copy.deepcopy(value["bundle"]["images"][0])
            second.update(id="10_90", videoId="10", imagePath="images/10_90.jpg")
            (value["root"] / second["imagePath"]).write_bytes(value["image_path"].read_bytes())
            common = np.zeros((24, 32), dtype=np.uint8)
            common[2:8, 2:10] = 1
            common_rle = review_io.encode_rle(common)
            for candidate in second["candidates"]:
                candidate["sourceAnnotationId"] += 100
                candidate.update(id=f"endoscapes-box-{candidate['sourceAnnotationId']}",
                                 mask=copy.deepcopy(common_rle), maskSha256=review_io.mask_sha256(common_rle))
            value["bundle"]["images"].append(second)
            for candidate in second["candidates"]:
                value["review"]["decisions"].append({
                    "candidateId": candidate["id"], "imageId": second["id"], "imageSha256": second["imageSha256"],
                    "proposalMaskSha256": candidate["maskSha256"], "decision": "accepted",
                    "mask": copy.deepcopy(common_rle), "notes": "", "reviewedAt": WHEN, "reviewMilliseconds": 1000.0,
                })
            write_json(value["bundle_path"], value["bundle"])
            value["review"]["bundleSha256"] = digest(value["bundle_path"])
            write_json(value["review_path"], value["review"])
            value["resolution"].update(bundleSha256=digest(value["bundle_path"]), reviewSha256=digest(value["review_path"]))
            write_json(value["resolution_path"], value["resolution"])
            result, manifest = prepare(value)
            self.assertEqual([row["videoId"] for row in manifest["samples"][3:]], [9])
            self.assertEqual(result["summary"]["omittedImageCount"], 1)
            self.assertFalse((Path(result["manifestPath"]).parent / "masks/10_90.png").exists())
            preparation = json.loads(Path(result["preparationPath"]).read_text())
            omitted = next(row for row in preparation["images"] if row["imageId"] == "10_90")
            self.assertFalse(omitted["included"])
            self.assertEqual(omitted["omissionReason"], "all_eligible_pixels_conflict")
            self.assertEqual(omitted["retainedPixels"], 0)
            self.assertEqual(omitted["conflictingPixels"], 48)

    def test_existing_output_is_never_overwritten_and_failed_write_is_not_published(self):
        with tempfile.TemporaryDirectory() as temporary:
            value = fixture(Path(temporary))
            output = value["root"] / "prepared"
            output.mkdir()
            sentinel = output / "keep.txt"
            sentinel.write_text("Existing teammate work")
            with self.assertRaises((ValueError, FileExistsError)):
                prepare(value, output)
            self.assertEqual(sentinel.read_text(), "Existing teammate work")
            self.assertEqual(list(output.iterdir()), [sentinel])
            failed = value["root"] / "write-failure"
            with patch("PIL.Image.Image.save", side_effect=OSError("simulated disk full")):
                with self.assertRaises(OSError):
                    prepare(value, failed)
            self.assertFalse(failed.exists())
            self.assertEqual(list(value["root"].glob(".write-failure*")), [])


@unittest.skipUnless(HAS_TRAINING, "Training dependencies not installed")
class ReviewedTrainingTensorTests(unittest.TestCase):
    def test_native_partial_targets_resize_to_both_training_sizes_and_ignore_gradients(self):
        from PIL import Image
        import torch
        from holospex_ml.losses import foreground_generalized_dice_loss
        from holospex_ml.training import SegmentationDataset

        with tempfile.TemporaryDirectory() as temporary:
            value = fixture(Path(temporary), width=854, height=480)
            _, manifest = prepare(value)
            added = manifest["samples"][-1]
            model_targets = np.full(value["expected"].shape, 255, dtype=np.int32)
            for source_id, model_index in ((4, 2), (3, 3), (1, 4), (2, 5)):
                model_targets[value["expected"] == source_id] = model_index
            for width, height in ((672, 384), (1120, 640)):
                with self.subTest(width=width, height=height):
                    dataset = SegmentationDataset([added], manifest["classes"], width, height, manifest["ignoreSourceIds"])
                    image, target = dataset[0]
                    self.assertEqual(tuple(image.shape), (3, height, width))
                    self.assertEqual(tuple(target.shape), (height, width))
                    self.assertTrue(torch.isfinite(image).all())
                    expected = np.asarray(Image.fromarray(model_targets).resize((width, height), Image.Resampling.NEAREST))
                    np.testing.assert_array_equal(target.numpy(), expected)
                    self.assertEqual(set(target.unique().tolist()), {2, 3, 4, 5, 255})
                    logits = torch.zeros(1, 7, height, width, requires_grad=True)
                    truth = target.unsqueeze(0)
                    ce = torch.nn.functional.cross_entropy(logits, truth, ignore_index=255)
                    dice = foreground_generalized_dice_loss(logits, truth)
                    loss = ce + 0.5 * dice
                    loss.backward()
                    ignored = (truth == 255).unsqueeze(1).expand_as(logits)
                    self.assertTrue(torch.isfinite(loss))
                    self.assertEqual(int(torch.count_nonzero(logits.grad[ignored])), 0)
                    self.assertGreater(float(logits.grad[~ignored].abs().sum()), 0)
                    with torch.no_grad():
                        changed = logits.detach().clone()
                        changed[ignored] = 100.0
                        changed_loss = torch.nn.functional.cross_entropy(changed, truth, ignore_index=255) + 0.5 * foreground_generalized_dice_loss(changed, truth)
                    torch.testing.assert_close(changed_loss, loss.detach())

    def test_a_tiny_native_positive_lost_on_resize_triggers_training_guard_before_optimizer_step(self):
        import torch
        from holospex_ml.losses import foreground_generalized_dice_loss
        from holospex_ml.training import SegmentationDataset, train

        class NoForwardExpected(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.value = torch.nn.Parameter(torch.zeros(1))

            def forward(self, image):
                raise AssertionError("All-ignore targets must be rejected before model execution")

        with tempfile.TemporaryDirectory() as temporary:
            value = fixture(Path(temporary), width=854, height=480, mode="single_pixel")
            _, manifest = prepare(value)
            added = manifest["samples"][-1]
            _, large = SegmentationDataset([added], manifest["classes"], 1120, 640, [255])[0]
            self.assertGreater(int((large == 2).sum()), 0)
            _, small = SegmentationDataset([added], manifest["classes"], 672, 384, [255])[0]
            self.assertEqual(set(small.unique().tolist()), {255})
            with self.assertRaisesRegex(ValueError, "only ignored"):
                foreground_generalized_dice_loss(torch.zeros(1, 7, 384, 672), small.unsqueeze(0))
            manifest["samples"] = [added] + [row for row in manifest["samples"] if row["split"] != "train"]
            optimizer = MagicMock()
            with patch("holospex_ml.training.build_model", return_value=NoForwardExpected()), \
                 patch("holospex_ml.training.torch.optim.AdamW", return_value=optimizer):
                with self.assertRaisesRegex(ValueError, "only ignored"):
                    train(manifest, value["root"] / "guard-test", width=672, height=384,
                          epochs=1, batch_size=1, device="cpu", pretrained=False, class_weighting="none")
            optimizer.zero_grad.assert_not_called()
            optimizer.step.assert_not_called()
            self.assertFalse((value["root"] / "guard-test" / "best.pt").exists())


if __name__ == "__main__":
    unittest.main()
