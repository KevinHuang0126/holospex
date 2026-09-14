import copy
import hashlib
import io
import unittest

from PIL import Image

from holospex_ml.box_data import inspect_content, plan_expansion, read_cases
from holospex_ml.download import DownloadError


def fixture():
    categories = [{"id": index + 1, "name": name} for index, name in enumerate(
        ("cystic_plate", "calot_triangle", "cystic_artery", "cystic_duct", "gallbladder", "tool"))]
    images = [{"id": video, "file_name": f"{video}_25.jpg", "video_id": video,
               "width": 854, "height": 480} for video in (1, 2)]
    coco = {"categories": categories, "images": images, "annotations": [
        {"id": 10, "image_id": 2, "category_id": 3, "bbox": [10., 20., 30., 40.],
         "area": 870, "iscrowd": 0}]}
    segmentation = {"categories": copy.deepcopy(categories), "images": [copy.deepcopy(images[0])]}
    cases = {"train": {1, 2}, "val": {3}, "test": {4}}
    seg_cases = {"train": {1}, "val": {3}, "test": {4}}
    return coco, segmentation, cases, seg_cases


class BoxExpansionTests(unittest.TestCase):
    def test_selects_only_new_cases_and_preserves_box_semantics(self):
        result = plan_expansion(*fixture())
        self.assertEqual(result["summary"]["additionalCases"], 1)
        self.assertEqual(result["summary"]["additionalImages"], 1)
        self.assertFalse(result["pixelMasksAvailable"])
        image = result["images"][0]
        self.assertEqual(image["annotationType"], "bounding_box")
        self.assertEqual(image["boxes"][0]["bboxXYWH"], [10., 20., 30., 40.])
        self.assertEqual(image["boxes"][0]["sourceArea"], 870)
        self.assertEqual(image["boxes"][0]["structureId"], "cystic_artery")
        self.assertNotIn("maskPath", image)
        self.assertEqual(result["categories"][1]["structureId"], "hepatocystic_triangle_dissection")

    def test_heldout_image_or_split_overlap_fails_before_acquisition(self):
        coco, seg, cases, seg_cases = fixture()
        coco["images"][1].update(file_name="3_25.jpg", video_id=3)
        with self.assertRaisesRegex(DownloadError, "held-out"):
            plan_expansion(coco, seg, cases, seg_cases)
        coco, seg, cases, seg_cases = fixture()
        cases["val"].add(2)
        with self.assertRaisesRegex(DownloadError, "Overlapping"):
            plan_expansion(coco, seg, cases, seg_cases)

    def test_extra_frame_from_existing_segmentation_case_requires_reinspection(self):
        coco, seg, cases, seg_cases = fixture()
        coco["images"][1].update(file_name="1_50.jpg", video_id=1)
        with self.assertRaisesRegex(DownloadError, "existing Seg50 case"):
            plan_expansion(coco, seg, cases, seg_cases)

    def test_invalid_geometry_and_unexpected_masks_are_rejected(self):
        for patch in ({"bbox": [850, 0, 20, 10]}, {"bbox": [0, 0, 0, 10]},
                      {"bbox": [float("nan"), 0, 1, 1]}, {"segmentation": [[0, 0, 1, 1]]}):
            coco, seg, cases, seg_cases = fixture()
            coco["annotations"][0].update(patch)
            with self.subTest(patch=patch), self.assertRaises(DownloadError):
                plan_expansion(coco, seg, cases, seg_cases)

    def test_category_drift_and_duplicate_identity_fail(self):
        coco, seg, cases, seg_cases = fixture()
        coco["categories"][0]["id"] = 9
        with self.assertRaisesRegex(DownloadError, "category maps"):
            plan_expansion(coco, seg, cases, seg_cases)
        coco, seg, cases, seg_cases = fixture()
        coco["images"].append(copy.deepcopy(coco["images"][1]))
        with self.assertRaisesRegex(DownloadError, "Duplicate source image"):
            plan_expansion(coco, seg, cases, seg_cases)

    def test_case_list_supports_author_scientific_notation_but_rejects_fractional_ids(self):
        self.assertEqual(read_cases("1.0000e+00\n2.000e+00"), {1, 2})
        for value in ("1.5", "nan", "1 1", "0", ""):
            with self.subTest(value=value), self.assertRaises(DownloadError):
                read_cases(value)

    def test_empty_annotations_stay_explicit_and_never_become_background_masks(self):
        coco, seg, cases, seg_cases = fixture()
        coco["annotations"] = []
        plan = plan_expansion(coco, seg, cases, seg_cases)
        self.assertEqual(plan["summary"]["framesWithoutBoxes"], 1)
        self.assertFalse(plan["images"][0]["hasBoxAnnotations"])
        self.assertEqual(plan["images"][0]["labelCompleteness"], "not_established")
        self.assertEqual(plan["pixelSupervision"], "none")

    def test_blank_duplicate_is_quarantined_with_exact_existing_split_matches(self):
        stream = io.BytesIO()
        Image.new("RGB", (10, 8), (0, 0, 0)).save(stream, format="JPEG")
        data = stream.getvalue()
        matches = [{"split": "val", "fileName": "3_25.jpg"}, {"split": "train", "fileName": "1_25.jpg"}]
        report = inspect_content({"width": 10, "height": 8}, data, {hashlib.sha256(data).hexdigest(): matches})
        self.assertEqual(report["acquisitionStatus"], "quarantined")
        self.assertEqual(report["existingSeg50Matches"], matches)
        self.assertEqual(report["quarantineReasons"], ["exact_content_match_to_existing_seg50", "uniform_black_frame"])

    def test_unique_nonblank_image_is_usable_but_mismatched_dimensions_fail(self):
        stream = io.BytesIO()
        Image.new("RGB", (10, 8), (60, 30, 10)).save(stream, format="JPEG")
        data = stream.getvalue()
        report = inspect_content({"width": 10, "height": 8}, data, {})
        self.assertEqual(report["acquisitionStatus"], "usable")
        self.assertEqual(report["quarantineReasons"], [])
        with self.assertRaisesRegex(DownloadError, "dimensions"):
            inspect_content({"width": 9, "height": 8}, data, {})


if __name__ == "__main__":
    unittest.main()
