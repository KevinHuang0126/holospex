"""Review is an explicit partial-label decision, never automatic supervision."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


MODULE = Path(__file__).resolve().parents[1] / "review" / "review_io.py"
SPEC = importlib.util.spec_from_file_location("holospex_review_io", MODULE)
review_io = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review_io)
PIL_AVAILABLE = importlib.util.find_spec("PIL") is not None
WHEN = "2026-09-13T14:00:00Z"
DIGEST = "a" * 64


def example_bundle():
    classes = [
        {"structureId": name, "label": name.replace("_", " "), "color": "#123456"}
        for name in sorted(review_io.STRUCTURES)
    ]
    mask = review_io.encode_rle([[0, 1, 1], [0, 0, 0]])
    candidates = [{
        "id": f"endoscapes-box-{index + 1}", "sourceAnnotationId": index + 1,
        "structureId": classes[index % len(classes)]["structureId"],
        "bboxXYWH": [0, 0, 3, 2], "source": "model_generated",
        "mask": copy.deepcopy(mask), "maskSha256": review_io.mask_sha256(mask),
        "proposalScore": 0.8, "scoreMeaning": "SAM predicted overlap quality, not anatomy confidence",
        "warnings": [],
    } for index in range(6)]
    return {
        "formatVersion": "1.0.0", "artifactType": "candidate_mask_bundle",
        "bundleId": "review-pilot-test", "createdAt": WHEN, "sourceManifestSha256": DIGEST,
        "selection": {"description": "Small train-only test fixture"},
        "generator": {"model": "test-double", "notARealProposal": True},
        "classes": classes,
        "images": [{"id": "1_2", "videoId": "1", "frameNumber": 2, "split": "train",
                    "width": 3, "height": 2, "imagePath": "images/1_2.png",
                    "imageSha256": "b" * 64, "candidates": candidates}],
    }


def example_review(bundle, digest=DIGEST):
    image = bundle["images"][0]
    decisions = []
    states = ["accepted", "edited", "pending", "needs_expert", "rejected"]
    for candidate, state in zip(image["candidates"], states):
        mask = copy.deepcopy(candidate["mask"])
        if state in {"edited", "pending"}:
            mask = review_io.encode_rle([[1, 1, 0], [0, 0, 0]])
        decisions.append({
            "candidateId": candidate["id"], "imageId": image["id"], "imageSha256": image["imageSha256"],
            "proposalMaskSha256": candidate["maskSha256"], "decision": state, "mask": mask,
            "notes": "Please confirm boundary" if state in {"needs_expert", "rejected"} else "",
            "reviewedAt": None if state == "pending" else WHEN,
            "reviewMilliseconds": 3500.5,
        })
    return {
        "formatVersion": "1.0.0", "artifactType": "candidate_mask_review", "bundleId": bundle["bundleId"],
        "bundleSha256": digest, "reviewer": {"name": "Named Test Reviewer", "reviewScope": "anatomy"},
        "exportedAt": WHEN, "decisions": decisions,
    }


class ReviewValidationTests(unittest.TestCase):
    def test_row_major_rle_preserves_holes_disconnected_pixels_and_empty_full_masks(self):
        for rows in ([[1, 1, 1], [1, 0, 1], [1, 1, 1]], [[1, 0, 1], [0, 0, 0]],
                     [[0, 0], [0, 0]], [[1, 1], [1, 1]]):
            with self.subTest(rows=rows):
                expected = bytes(pixel for row in rows for pixel in row)
                rle = review_io.encode_rle(rows)
                self.assertEqual(review_io.decode_rle(rle), expected)
                self.assertEqual(review_io.encode_rle(expected, len(rows[0]), len(rows)), rle)
                self.assertEqual(review_io.mask_sha256(rle), hashlib.sha256(expected).hexdigest())
        self.assertEqual(review_io.encode_rle([[1, 1]])["counts"], [0, 2])
        self.assertEqual(review_io.encode_rle([[0, 0]])["counts"], [2])

    def test_malformed_and_noncanonical_rle_are_rejected_before_allocation(self):
        base = review_io.encode_rle([[0, 1, 1], [0, 0, 0]])
        for change in ({"counts": []}, {"counts": [6, 0]}, {"counts": [0, 0, 6]},
                       {"counts": [3, 4]}, {"counts": [-1, 7]}, {"counts": [3.0, 3]},
                       {"counts": [True, 5]}, {"counts": [10**100]}, {"width": True},
                       {"height": 0}, {"width": 50_000_001}, {"encoding": "coco"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                review_io.decode_rle(dict(base, **change))
        with self.assertRaises(ValueError):
            review_io.encode_rle([[255, 0]])

    def test_proposals_require_immutable_digest_dimensions_source_and_train_split(self):
        base = example_bundle()
        self.assertIsNone(review_io.validate_bundle(base))
        mutations = [
            lambda b: b["images"][0].update(split="test"),
            lambda b: b["images"][0].update(id="9_2"),
            lambda b: b["images"][0].update(imagePath="../escape.png"),
            lambda b: b["images"][0].update(imagePath="C:\\image.png"),
            lambda b: b["images"][0]["candidates"][0].update(maskSha256="c" * 64),
            lambda b: b["images"][0]["candidates"][0].update(source="reviewed_annotation"),
            lambda b: b["images"][0]["candidates"][0].update(id="endoscapes-box-999"),
            lambda b: b["images"][0]["candidates"].append(copy.deepcopy(b["images"][0]["candidates"][0])),
            lambda b: b["images"][0].update(width=4),
            lambda b: b["generator"].update(score=float("nan")),
        ]
        for mutation in mutations:
            bundle = copy.deepcopy(base)
            mutation(bundle)
            with self.subTest(bundle=bundle), self.assertRaises(ValueError):
                review_io.validate_bundle(bundle)

    def test_source_boxes_and_scores_must_be_finite_in_bounds(self):
        for field, value in [("bboxXYWH", [0, 0, 4, 2]), ("bboxXYWH", [0, 0, 0, 2]),
                             ("bboxXYWH", [0, 0, True, 2]), ("bboxXYWH", [float("nan"), 0, 1, 1]),
                             ("proposalScore", float("inf")), ("proposalScore", True)]:
            bundle = example_bundle()
            bundle["images"][0]["candidates"][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                review_io.validate_bundle(bundle)

    def test_partial_and_pending_decisions_are_never_promoted(self):
        bundle = example_bundle()
        review = example_review(bundle)
        summary = review_io.validate_review(review, bundle, DIGEST)
        self.assertEqual(summary["eligibleCandidateIds"], ["endoscapes-box-1", "endoscapes-box-2"])
        self.assertEqual(summary["missingDecisionCount"], 1)
        self.assertEqual(summary["unreviewedCandidateCount"], 2)
        self.assertEqual(summary["decisionCounts"], {"accepted": 1, "edited": 1, "needs_expert": 1, "pending": 1, "rejected": 1})
        review["decisions"] = []
        summary = review_io.validate_review(review, bundle, DIGEST)
        self.assertEqual(summary["eligibleCandidateIds"], [])
        self.assertEqual(summary["unreviewedCandidateCount"], 6)

    def test_technical_scope_never_creates_anatomy_supervision(self):
        bundle = example_bundle()
        review = example_review(bundle)
        review["reviewer"]["reviewScope"] = "technical"
        self.assertEqual(review_io.validate_review(review, bundle, DIGEST)["eligibleCandidateIds"], [])

    def test_stale_bundle_image_and_proposal_review_identities_fail(self):
        bundle = example_bundle()
        for field, value, top in [("bundleId", "another-bundle", True), ("bundleSha256", "c" * 64, True),
                                  ("imageId", "2_2", False), ("imageSha256", "c" * 64, False),
                                  ("proposalMaskSha256", "c" * 64, False), ("candidateId", "unknown", False)]:
            review = example_review(bundle)
            (review if top else review["decisions"][0])[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                review_io.validate_review(review, bundle, DIGEST)
        review = example_review(bundle)
        review["decisions"].append(copy.deepcopy(review["decisions"][0]))
        with self.assertRaises(ValueError):
            review_io.validate_review(review, bundle, DIGEST)

    def test_accept_edit_semantics_and_nonempty_masks(self):
        bundle = example_bundle()
        for index, mask in [(0, [[1, 1, 0], [0, 0, 0]]), (1, [[0, 1, 1], [0, 0, 0]]),
                            (0, [[0, 0, 0], [0, 0, 0]]), (1, [[0, 0, 0], [0, 0, 0]])]:
            review = example_review(bundle)
            review["decisions"][index]["mask"] = review_io.encode_rle(mask)
            with self.subTest(index=index, mask=mask), self.assertRaises(ValueError):
                review_io.validate_review(review, bundle, DIGEST)

    def test_named_reviewer_scope_notes_and_times_are_required(self):
        bundle = example_bundle()
        changes = [
            lambda r: r["reviewer"].update(name=" \t "),
            lambda r: r["reviewer"].update(name="Someone\nElse"),
            lambda r: r["reviewer"].update(reviewScope="expert"),
            lambda r: r["decisions"][3].update(notes="  "),
            lambda r: r["decisions"][4].update(notes=""),
            lambda r: r["decisions"][0].update(reviewedAt=None),
            lambda r: r["decisions"][0].update(reviewedAt="2026-09-13T14:00:00"),
            lambda r: r["decisions"][0].update(reviewedAt="2026-99-13T14:00:00Z"),
            lambda r: r["decisions"][2].update(reviewedAt=WHEN),
            lambda r: r["decisions"][0].update(reviewMilliseconds=float("nan")),
            lambda r: r["decisions"][0].update(reviewMilliseconds=-1),
            lambda r: r["decisions"][0].update(reviewMilliseconds=True),
        ]
        for change in changes:
            review = example_review(bundle)
            change(review)
            with self.subTest(review=review), self.assertRaises(ValueError):
                review_io.validate_review(review, bundle, DIGEST)

    def test_strict_json_rejects_duplicate_keys_nonfinite_and_unknown_fields(self):
        for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                review_io._decode_json(raw)
        bundle = example_bundle()
        review = example_review(bundle)
        review["decisions"][0]["approved"] = True
        with self.assertRaises(ValueError):
            review_io.validate_review(review, bundle, DIGEST)


@unittest.skipUnless(PIL_AVAILABLE, "Install ml[data] for review file import tests")
class ReviewImportTests(unittest.TestCase):
    def fixture(self, root):
        from PIL import Image
        bundle = example_bundle()
        image_path = root / bundle["images"][0]["imagePath"]
        image_path.parent.mkdir(parents=True)
        Image.new("RGB", (3, 2), (10, 20, 30)).save(image_path)
        bundle["images"][0]["imageSha256"] = hashlib.sha256(image_path.read_bytes()).hexdigest()
        bundle_path = root / "bundle.json"
        bundle_path.write_text(json.dumps(bundle))
        digest = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
        review = example_review(bundle, digest)
        review_path = root / "review.json"
        review_path.write_text(json.dumps(review))
        return bundle, review, bundle_path, review_path

    def test_import_preserves_overlapping_candidates_as_partial_binary_masks(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, review, bundle_path, review_path = self.fixture(root)
            output = root / "imported"
            receipt = review_io.import_review(bundle_path, review_path, output)
            self.assertEqual(receipt["trainingEnrollment"], "none")
            self.assertEqual(receipt["reviewer"], review["reviewer"])
            self.assertEqual((output / "review.json").read_bytes(), review_path.read_bytes())
            self.assertEqual(len(receipt["eligibleMasks"]), 2)
            self.assertEqual(len(list((output / "masks").glob("*.png"))), 2)
            actual_masks = []
            for metadata in receipt["eligibleMasks"]:
                with Image.open(output / metadata["maskPath"]) as image:
                    self.assertEqual(image.size, (3, 2))
                    actual_masks.append(list(image.tobytes()))
                self.assertEqual(metadata["coverage"], "partial_positive_only")
                self.assertIn("unknown", metadata["maskSemantics"]["0"])
                self.assertEqual(metadata["split"], "train")
            self.assertEqual(actual_masks, [[0, 255, 255, 0, 0, 0], [255, 255, 0, 0, 0, 0]])
            # Both retain the shared foreground pixel even though classes differ.
            self.assertEqual(actual_masks[0][1], actual_masks[1][1])
            self.assertNotEqual(receipt["eligibleMasks"][0]["structureId"], receipt["eligibleMasks"][1]["structureId"])

    def test_invalid_late_decision_has_no_partial_import_side_effects(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, review, bundle_path, review_path = self.fixture(root)
            review["decisions"][-1]["notes"] = ""
            review_path.write_text(json.dumps(review))
            before = set(root.iterdir())
            output = root / "new-parent" / "imported"
            with self.assertRaises(ValueError):
                review_io.import_review(bundle_path, review_path, output)
            self.assertEqual(set(root.iterdir()), before)

    def test_stale_source_file_and_bundle_bytes_prevent_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, _, bundle_path, review_path = self.fixture(root)
            output = root / "imported"
            original = (root / bundle["images"][0]["imagePath"]).read_bytes()
            (root / bundle["images"][0]["imagePath"]).write_bytes(b"different image")
            with self.assertRaisesRegex(ValueError, "Image SHA"):
                review_io.import_review(bundle_path, review_path, output)
            self.assertFalse(output.exists())
            (root / bundle["images"][0]["imagePath"]).write_bytes(original)
            bundle_path.write_bytes(bundle_path.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "bundle identity"):
                review_io.import_review(bundle_path, review_path, output)
            self.assertFalse(output.exists())

    def test_symlink_escape_and_wrong_image_dimensions_are_rejected(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as elsewhere:
            root = Path(directory)
            bundle, _, bundle_path, _ = self.fixture(root)
            target = root / bundle["images"][0]["imagePath"]
            outside = Path(elsewhere) / "outside.png"
            outside.write_bytes(target.read_bytes())
            target.unlink()
            target.symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "escapes"):
                review_io.validate_bundle_files(bundle, bundle_path)
            target.unlink()
            Image.new("RGB", (4, 2)).save(target)
            bundle["images"][0]["imageSha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "dimensions mismatch"):
                review_io.validate_bundle_files(bundle, bundle_path)

    def test_technical_import_writes_receipt_without_eligible_pngs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, review, bundle_path, review_path = self.fixture(root)
            review["reviewer"]["reviewScope"] = "technical"
            review_path.write_text(json.dumps(review))
            output = root / "imported"
            receipt = review_io.import_review(bundle_path, review_path, output)
            self.assertEqual(receipt["eligibleMasks"], [])
            self.assertTrue((output / "receipt.json").is_file())
            self.assertEqual(list((output / "masks").iterdir()), [])

    def test_existing_destination_is_not_overwritten_and_failed_write_is_cleaned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, bundle_path, review_path = self.fixture(root)
            output = root / "imported"
            output.mkdir()
            marker = output / "owned.txt"
            marker.write_text("keep")
            with self.assertRaises(FileExistsError):
                review_io.import_review(bundle_path, review_path, output)
            self.assertEqual(marker.read_text(), "keep")
            output2 = root / "imported2"
            before = set(root.iterdir())
            with patch("PIL.Image.Image.save", side_effect=OSError("disk write failed")), self.assertRaises(OSError):
                review_io.import_review(bundle_path, review_path, output2)
            self.assertEqual(set(root.iterdir()), before)


if __name__ == "__main__":
    unittest.main()
