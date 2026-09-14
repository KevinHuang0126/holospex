"""Selection leakage guards and lossless proposal serialization, without SAM."""
import copy
import hashlib
import importlib.util
import contextlib
import io
import json
import tempfile
from pathlib import Path
from collections import Counter
import unittest

import numpy as np

SPEC = importlib.util.spec_from_file_location("pilot_generator", Path(__file__).parents[1] / "review/generate_pilot.py")
GEN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GEN)


def manifest():
    images = []
    for case in range(1, 61):
        boxes = [{"annotationId": case*10+i, "structureId": name,
                  "bboxXYWH": [10+i*20, 20, 5+(case%12)*2, 6+case%7]}
                 for i, name in enumerate(GEN.SMALL_CLASSES)]
        images.append({"fileName": f"{case}_100.jpg", "videoId": case, "split": "train",
                       "acquisitionStatus": "usable", "quarantineReasons": [], "frameNumber": 100,
                       "width": 160, "height": 90, "boxes": boxes})
    return {"split": "train", "pixelMasksAvailable": False, "images": images,
            "leakageAudit": {"heldOutCases": [121], "existingSeg50TrainCases": [120],
                             "officialTrainCases": list(range(1, 121))}}


def multi_frame_manifest():
    data = manifest()
    images = []
    for source in data["images"]:
        for offset, frame in enumerate((100, 500, 1600)):
            image = copy.deepcopy(source)
            image["fileName"] = f"{source['videoId']}_{frame}.jpg"
            image["frameNumber"] = frame
            for box in image["boxes"]:
                box["annotationId"] = box["annotationId"] * 10 + offset
            images.append(image)
    data["images"] = images
    return data


def write_preparation_fixture(root):
    data = multi_frame_manifest()
    (root / "images").mkdir()
    (root / "source").mkdir()
    for image in data["images"]:
        image["imagePath"] = "images/" + image["fileName"]
        path = root / image["imagePath"]
        path.write_bytes(b"image fixture " + image["fileName"].encode())
        image["sha256"] = GEN.sha256(path)
    (root / "source/LICENSE").write_text("fixture license")
    (root / "source/README.md").write_text("fixture provenance")
    (root / "manifest.json").write_text(json.dumps(data))
    (root / "anatomy.json").write_text(json.dumps({c: {"label": c, "color": "#ffffff"} for c in GEN.SMALL_CLASSES}))
    prior = {"bundleId": "review-pilot-001", "artifactType": "candidate_mask_bundle",
             "images": [{"videoId": str(case), "split": "train"} for case in range(1, 21)]}
    (root / "prior.json").write_text(json.dumps(prior))
    with contextlib.redirect_stdout(io.StringIO()):
        GEN.prepare(root / "manifest.json", root / "prepared", root / "anatomy.json",
                    image_count=50, case_count=25, exclude_bundles=[root / "prior.json"],
                    bundle_id="review-batch-002", min_frame_gap=750)
    return json.loads((root / "prepared/selection.json").read_text())


class PilotGenerationTests(unittest.TestCase):
    def test_case_distinct_deterministic_selection_ignores_input_order(self):
        data = manifest()
        chosen, selection = GEN.select_images(data, 42)
        data["images"].reverse()
        repeated, _ = GEN.select_images(data, 42)
        self.assertEqual([i["videoId"] for i in chosen], [i["videoId"] for i in repeated])
        self.assertEqual(len({i["videoId"] for i in chosen}), 20)
        self.assertEqual(selection["anchorQuotas"], dict(zip(GEN.SMALL_CLASSES, [8, 8, 2, 2])))
        self.assertFalse(selection["modelScoresUsedForSelection"])

    def test_protected_case_or_non_train_manifest_fails(self):
        data = manifest(); data["images"][0]["videoId"] = 121
        with self.assertRaisesRegex(ValueError, "protected"):
            GEN.select_images(data)
        data = manifest(); data["split"] = "val"
        with self.assertRaises(ValueError): GEN.select_images(data)

    def test_quarantined_case_not_selected(self):
        data = manifest()
        data["images"][0]["quarantineReasons"] = ["black"]
        chosen, _ = GEN.select_images(data)
        self.assertNotIn(1, [i["videoId"] for i in chosen])

    def test_next_batch_uses_new_cases_two_separated_frames_and_all_classes(self):
        data = multi_frame_manifest()
        chosen, details = GEN.select_images(data, image_count=50, case_count=25,
                                           excluded_cases=range(1, 21), min_frame_gap=750)
        counts = Counter(i["videoId"] for i in chosen)
        self.assertEqual((len(chosen), len(counts), set(counts.values())), (50, 25, {2}))
        self.assertFalse(set(counts) & set(range(1, 21)))
        self.assertEqual(details["anchorQuotas"], dict(zip(GEN.SMALL_CLASSES, [10, 10, 3, 2])))
        for case in counts:
            frames = [i["frameNumber"] for i in chosen if i["videoId"] == case]
            self.assertGreaterEqual(abs(frames[1] - frames[0]), 750)
        self.assertEqual(len({b["annotationId"] for i in chosen for b in i["boxes"]}), 200)
        data["images"].reverse()
        repeated, repeated_details = GEN.select_images(data, image_count=50, case_count=25,
                                                      excluded_cases=range(1, 21), min_frame_gap=750)
        self.assertEqual(chosen, repeated)
        self.assertEqual(details, repeated_details)
        self.assertFalse(details["modelScoresUsedForSelection"])

    def test_insufficient_temporal_separation_or_cases_fails_instead_of_reusing(self):
        data = multi_frame_manifest()
        with self.assertRaisesRegex(ValueError, "frame gap"):
            GEN.select_images(data, image_count=50, case_count=25, min_frame_gap=1501)
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            GEN.select_images(data, image_count=50, case_count=25, excluded_cases=range(1, 40))
        with self.assertRaisesRegex(ValueError, "one or two"):
            GEN.select_images(data, image_count=51, case_count=25)

    def test_preparation_preserves_prior_bundle_and_validates_frozen_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frozen = write_preparation_fixture(root)
            self.assertEqual(frozen["bundleId"], "review-batch-002")
            record = frozen["selection"]["excludedPriorBundles"][0]
            self.assertEqual(record["sha256"], GEN.sha256(root / "prior.json"))
            self.assertEqual((root / "prepared" / record["snapshotPath"]).read_bytes(),
                             (root / "prior.json").read_bytes())
            self.assertEqual(record["cases"], sorted(map(str, range(1, 21))))
            GEN.validate_frozen_selection(frozen, root / "prepared")
            self.assertEqual(frozen["selection"]["proposalCountsByClass"],
                             dict.fromkeys(GEN.SMALL_CLASSES, 50))
            (root / "prepared" / record["snapshotPath"]).write_text("{}")
            with self.assertRaisesRegex(ValueError, "checksum"):
                GEN.validate_frozen_selection(frozen, root / "prepared")

    def test_generation_guard_rejects_changed_frames_boxes_and_prior_case(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frozen = write_preparation_fixture(root)
            changed = copy.deepcopy(frozen)
            changed["images"][0]["boxes"].pop()
            with self.assertRaisesRegex(ValueError, "boxes"):
                GEN.validate_frozen_selection(changed, root / "prepared")
            changed = copy.deepcopy(frozen)
            changed["images"][0]["frameNumber"] += 1
            with self.assertRaisesRegex(ValueError, "differ"):
                GEN.validate_frozen_selection(changed, root / "prepared")
            changed = copy.deepcopy(frozen)
            for image in changed["images"][:2]:
                image["videoId"] = 1
            with self.assertRaisesRegex(ValueError, "prior review"):
                GEN.validate_frozen_selection(changed, root / "prepared")
            changed = copy.deepcopy(frozen)
            changed["selection"]["minimumFrameGap"] = 1600
            with self.assertRaisesRegex(ValueError, "minimum frame gap"):
                GEN.validate_frozen_selection(changed, root / "prepared")

    def test_failed_preparation_does_not_publish_partial_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_preparation_fixture(root)
            data = json.loads((root / "manifest.json").read_text())
            for image in data["images"]:
                image["sha256"] = "0" * 64
            (root / "manifest.json").write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "checksum"):
                GEN.prepare(root / "manifest.json", root / "failed", root / "anatomy.json")
            self.assertFalse((root / "failed").exists())

    def test_rle_preserves_hole_disconnected_pixel_and_row_order(self):
        mask = np.array([[1,1,1,0,0], [1,0,1,0,1], [1,1,1,0,0]], dtype=np.uint8)
        rle, digest = GEN.encode_mask(mask)
        decoded = np.concatenate([np.full(n, i%2, dtype=np.uint8) for i,n in enumerate(rle["counts"])])
        np.testing.assert_array_equal(decoded.reshape(mask.shape), mask)
        self.assertEqual(rle["counts"][0], 0)
        self.assertTrue(all(n > 0 for n in rle["counts"][1:]))
        self.assertEqual(digest, hashlib.sha256(mask.tobytes()).hexdigest())
        empty, _ = GEN.encode_mask(np.zeros((2,3), dtype=bool))
        self.assertEqual(empty["counts"], [6])

    def test_geometric_warning_does_not_clip_mask(self):
        mask = np.zeros((20,30), dtype=bool); mask[1,1] = True; mask[5,5] = True
        before = mask.copy()
        flags = GEN.mask_warnings(mask, [4,4,4,4])
        self.assertIn("mask_extends_outside_source_box", flags)
        np.testing.assert_array_equal(mask, before)

    def test_path_escape_rejected(self):
        for path in ("../outside.jpg", "/outside.jpg", "images\\outside.jpg"):
            with self.assertRaises(ValueError): GEN.safe_path(Path("/tmp/input"), path)


if __name__ == "__main__":
    unittest.main()
