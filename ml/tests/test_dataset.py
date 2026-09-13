import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from holospex_ml.dataset import DatasetError, encode_mask, prepare_dataset, read_image, read_semantic_mask


LABEL_MAP = "background\ncystic_plate\ncalot_triangle\ncystic_artery\ncystic_duct\ngallbladder\ntool\n"


class DatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "seg50"
        self.root.mkdir()
        (self.root / "seg_label_map.txt").write_text(LABEL_MAP)
        (self.root / "all").mkdir()
        for video, split in enumerate(("train", "val", "test"), start=1):
            (self.root / f"{split}_seg_vids.txt").write_text(f"{video:.18e}\n")
            split_root = self.root / f"{split}_seg"
            (split_root / "semseg").mkdir(parents=True)
            # IDs are intentionally unrelated to PNG IDs to catch accidental
            # category_id -> training-index assumptions.
            metadata = {
                "images": [{"id": video, "file_name": f"{video}_30.jpg", "width": 12, "height": 8, "video_id": video, "frame_id": None}],
                "categories": [{"id": 101, "name": "gallbladder"}],
                "annotations": [{"id": 1, "image_id": video, "category_id": 101,
                                 "segmentation": [[1, 1, 10, 1, 10, 7]], "bbox": [1, 1, 9, 6]}],
            }
            (split_root / "annotation_coco.json").write_text(json.dumps(metadata))
            Image.fromarray(np.full((8, 12, 3), 100, dtype=np.uint8)).save(self.root / "all" / f"{video}_30.jpg")
            mask = np.zeros((8, 12), dtype=np.uint8)
            mask[2:6, 2:8] = 5
            Image.fromarray(np.repeat(mask[:, :, None], 3, axis=2)).save(split_root / "semseg" / f"{video}_30.png")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def metadata(self, split: str = "train") -> tuple[Path, dict]:
        path = self.root / f"{split}_seg" / "annotation_coco.json"
        return path, json.loads(path.read_text())

    def test_png_id_mapping_is_independent_of_coco_category_ids(self) -> None:
        manifest = prepare_dataset(self.root, fps=30)
        self.assertEqual(manifest["report"]["counts"], {"train": 1, "val": 1, "test": 1})
        sample = manifest["samples"][0]
        self.assertTrue(Path(sample["imagePath"]).is_absolute())
        self.assertEqual(sample["timestampMs"], 1000)
        self.assertEqual(read_image(Path(sample["imagePath"])).shape, (8, 12, 3))
        mask = read_semantic_mask(Path(sample["maskPath"]))
        self.assertEqual(mask.shape, (8, 12))
        encoded = encode_mask(mask, manifest["classes"])
        self.assertEqual(encoded.dtype, np.int64)
        self.assertEqual(set(np.unique(encoded)), {0, 1})
        gallbladder = next(item for item in manifest["classes"] if item["structureId"] == "gallbladder")
        self.assertEqual(gallbladder, {"index": 1, "sourceId": 5, "structureId": "gallbladder"})

    def test_unknown_mask_ids_fail(self) -> None:
        path = self.root / "train_seg/semseg/1_30.png"
        Image.fromarray(np.full((8, 12), 99, dtype=np.uint8)).save(path)
        with self.assertRaisesRegex(DatasetError, "Unknown semantic label IDs"):
            prepare_dataset(self.root, fps=30)

    def test_official_shared_semseg_directory_is_supported(self) -> None:
        (self.root / "semseg").mkdir()
        for video, split in enumerate(("train", "val", "test"), start=1):
            source = self.root / f"{split}_seg/semseg/{video}_30.png"
            source.rename(self.root / "semseg" / source.name)
        manifest = prepare_dataset(self.root, fps=25)
        self.assertEqual(manifest["report"]["counts"], {"train": 1, "val": 1, "test": 1})
        self.assertTrue(all(Path(sample["maskPath"]).parent == (self.root / "semseg").resolve()
                            for sample in manifest["samples"]))

    def test_conflicting_shared_and_split_masks_are_rejected(self) -> None:
        (self.root / "semseg").mkdir()
        Image.fromarray(np.zeros((8, 12), dtype=np.uint8)).save(self.root / "semseg/1_30.png")
        with self.assertRaisesRegex(DatasetError, "Ambiguous semantic masks"):
            prepare_dataset(self.root, fps=25)

    def test_ignored_source_pixels_remain_distinct_from_background(self) -> None:
        mask_path = self.root / "train_seg/semseg/1_30.png"
        mask = read_semantic_mask(mask_path)
        mask[0, :2] = 255
        Image.fromarray(mask).save(mask_path)
        with self.assertRaisesRegex(DatasetError, "Unknown semantic label IDs"):
            prepare_dataset(self.root, fps=25)
        manifest = prepare_dataset(self.root, fps=25, ignore_source_ids=[255])
        encoded = encode_mask(mask, manifest["classes"], ignore_source_ids=[255])
        self.assertEqual(encoded[0, :3].tolist(), [255, 255, 0])
        self.assertEqual(manifest["ignoreIndex"], 255)
        self.assertEqual(manifest["ignoreSourceIds"], [255])
        ignored = manifest["report"]["ignoredLabelCounts"]["train"]
        self.assertEqual((ignored["images"], ignored["pixels"]), (1, 2))
        self.assertEqual(ignored["bySourceId"]["255"], {"images": 1, "pixels": 2})
        with self.assertRaisesRegex(DatasetError, "Cannot ignore known"):
            prepare_dataset(self.root, fps=25, ignore_source_ids=[0, 255])

    def test_explicit_exclusion_quarantines_unknown_frame_without_modifying_raw_mask(self) -> None:
        path, metadata = self.metadata("val")
        metadata["images"].append({"id": 20, "file_name": "2_60.jpg", "width": 12, "height": 8})
        metadata["annotations"].append({"id": 20, "image_id": 20, "category_id": 101,
                                        "segmentation": [[0, 0, 10, 0, 10, 7]]})
        path.write_text(json.dumps(metadata))
        Image.fromarray(np.full((8, 12, 3), 127, dtype=np.uint8)).save(self.root / "all/2_60.jpg")
        mask_path = self.root / "val_seg/semseg/2_60.png"
        Image.fromarray(np.full((8, 12), 7, dtype=np.uint8)).save(mask_path)
        original = mask_path.read_bytes()
        with self.assertRaisesRegex(DatasetError, "Unknown semantic label IDs"):
            prepare_dataset(self.root, fps=25, ignore_source_ids=[255])
        manifest = prepare_dataset(self.root, fps=25, ignore_source_ids=[255], exclude_frames=["2_60.png"])
        self.assertEqual(manifest["report"]["counts"]["val"], 1)
        exclusion = manifest["report"]["excludedFrames"][0]
        self.assertEqual(exclusion["frame"], "2_60")
        self.assertEqual(exclusion["unmappedSourceIds"], [7])
        self.assertIn("unmapped", exclusion["reason"])
        self.assertEqual(mask_path.read_bytes(), original)

    def test_nonexistent_exclusion_cannot_hide_a_typo(self) -> None:
        with self.assertRaisesRegex(DatasetError, "were not found"):
            prepare_dataset(self.root, fps=25, exclude_frames=["999_123"])

    def test_explicit_numeric_label_map_also_supported(self) -> None:
        (self.root / "seg_label_map.txt").write_text("\n".join(f"{index} {name}" for index, name in enumerate(LABEL_MAP.splitlines())))
        self.assertEqual(prepare_dataset(self.root, fps=30)["classes"][1]["sourceId"], 5)

    def test_case_level_split_leakage_fails_before_sampling(self) -> None:
        (self.root / "val_seg_vids.txt").write_text("1\n")
        with self.assertRaisesRegex(DatasetError, "split leakage"):
            prepare_dataset(self.root, fps=30)

    def test_missing_segmentation_mask_never_becomes_background(self) -> None:
        (self.root / "train_seg/semseg/1_30.png").unlink()
        with self.assertRaisesRegex(DatasetError, "Missing semantic mask"):
            prepare_dataset(self.root, fps=30)

    def test_bbox_only_missing_mask_is_skipped_and_reported(self) -> None:
        path, metadata = self.metadata()
        metadata["images"].append({"id": 10, "file_name": "1_60.jpg", "width": 12, "height": 8})
        metadata["annotations"].append({"id": 10, "image_id": 10, "category_id": 101, "bbox": [1, 1, 9, 6]})
        path.write_text(json.dumps(metadata))
        manifest = prepare_dataset(self.root, fps=30)
        self.assertEqual(manifest["report"]["counts"]["train"], 1)
        self.assertEqual(manifest["report"]["skippedWithoutSemanticMask"]["train"], 1)

    def test_nonidentical_rgb_mask_channels_fail(self) -> None:
        path = self.root / "train_seg/semseg/1_30.png"
        mask = np.zeros((8, 12, 3), dtype=np.uint8)
        mask[0, 0, 1] = 5
        Image.fromarray(mask).save(path)
        with self.assertRaisesRegex(DatasetError, "channels differ"):
            prepare_dataset(self.root, fps=30)

    def test_image_mask_and_metadata_dimensions_checked(self) -> None:
        path = self.root / "train_seg/semseg/1_30.png"
        Image.fromarray(np.zeros((4, 12), dtype=np.uint8)).save(path)
        with self.assertRaisesRegex(DatasetError, "dimension mismatch"):
            prepare_dataset(self.root, fps=30)

    def test_unknown_category_names_fail(self) -> None:
        path, metadata = self.metadata()
        metadata["categories"][0]["name"] = "unmapped-organ"
        path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(DatasetError, "Unrecognized anatomy"):
            prepare_dataset(self.root, fps=30)

    def test_image_must_belong_to_its_declared_video_split(self) -> None:
        (self.root / "train_seg_vids.txt").write_text("4\n")
        with self.assertRaisesRegex(DatasetError, "absent from"):
            prepare_dataset(self.root, fps=30)

    def test_internal_image_symlinks_resolve_to_one_file(self) -> None:
        (self.root / "train_seg/images").symlink_to(self.root / "all", target_is_directory=True)
        manifest = prepare_dataset(self.root, fps=30)
        self.assertEqual(manifest["samples"][0]["imagePath"], str((self.root / "all/1_30.jpg").resolve()))

    def test_external_symlink_fails(self) -> None:
        external = Path(self.temporary.name) / "external.jpg"
        Image.fromarray(np.full((8, 12, 3), 100, dtype=np.uint8)).save(external)
        local = self.root / "all/1_30.jpg"
        local.unlink()
        local.symlink_to(external)
        with self.assertRaisesRegex(DatasetError, "escapes root"):
            prepare_dataset(self.root, fps=30)

    def test_bad_fps_and_unparseable_labels_fail_actionably(self) -> None:
        for fps in (0, float("nan"), True):
            with self.assertRaisesRegex(DatasetError, "fps"):
                prepare_dataset(self.root, fps=fps)
        (self.root / "seg_label_map.txt").write_text("unknown layout")
        with self.assertRaisesRegex(DatasetError, "Cannot parse"):
            prepare_dataset(self.root, fps=30)


if __name__ == "__main__":
    unittest.main()
