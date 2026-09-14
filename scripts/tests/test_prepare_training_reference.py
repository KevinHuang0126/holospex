import importlib.util
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("prepare_training_reference", ROOT / "scripts/prepare-training-reference.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TrainingReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.data = self.folder / "endoscapes"
        (self.data / "all").mkdir(parents=True)
        (self.data / "semseg").mkdir()
        self.raw = bytes([0, 5, 5, 4, 6, 1, 2, 3, 255, 0, 5, 0, 0, 0, 0])
        for number in (30, 60):
            Image.new("RGB", (5, 3), (number, 65, 40)).save(self.data / "all" / f"1_{number}.jpg")
            Image.frombytes("L", (5, 3), self.raw).save(self.data / "semseg" / f"1_{number}.png")
        self.manifest = {
            "schemaVersion": "1.0.0", "dataset": "endoscapes-seg50", "root": str(self.data),
            "classes": MODULE.CLASSES, "ignoreIndex": 255, "ignoreSourceIds": [255],
            "report": {"splitVideoIds": {"train": [1], "val": [2], "test": [3]}},
            "samples": [{"split": split, "videoId": video, "frameNumber": number, "timestampMs": number * 40,
                         "imagePath": str(self.data / "all" / f"{video}_{number}.jpg"),
                         "maskPath": str(self.data / "semseg" / f"{video}_{number}.png")}
                        for split, video, number in (("test", 3, 30), ("train", 1, 60), ("val", 2, 30), ("train", 1, 30))],
        }
        # Validation/test source files deliberately do not exist: exporter must not read them.
        self.path = self.folder / "manifest.json"
        self.output = self.folder / "prepared"
        self.write_manifest()

    def tearDown(self):
        self.temp.cleanup()

    def write_manifest(self):
        self.path.write_text(json.dumps(self.manifest), encoding="utf-8")

    def test_training_only_export_preserves_sources_and_remaps_exact_pixels(self):
        identities = MODULE.prepare_training_reference(self.path, self.output)
        self.assertEqual(identities, ["1_30", "1_60"])
        case = self.output / "1_30"
        self.assertEqual((case / "image.jpg").read_bytes(), (self.data / "all/1_30.jpg").read_bytes())
        self.assertEqual((case / "labels-source.png").read_bytes(), (self.data / "semseg/1_30.png").read_bytes())
        with Image.open(case / "labels-index.png") as mask:
            self.assertEqual(mask.mode, "L")
            self.assertEqual(mask.size, (5, 3))
            self.assertEqual(mask.tobytes(), bytes([0, 1, 1, 2, 6, 4, 5, 3, 255, 0, 1, 0, 0, 0, 0]))
        labels = json.loads((case / "labels.json").read_text())
        self.assertEqual(labels["annotationSource"]["split"], "train")
        self.assertEqual(labels["annotationSource"]["kind"], "supplied_dataset_annotation")
        self.assertEqual(labels["frame"]["timestampMs"], 0)
        self.assertEqual(labels["frame"]["frameNumber"], 0)
        self.assertEqual(labels["structures"], [])
        self.assertEqual(labels["raster"]["ignoredPixelCount"], 1)
        self.assertEqual(labels["raster"]["pixelCounts"]["gallbladder"], 3)
        self.assertEqual(labels["raster"]["pixelCounts"]["cystic_duct"], 1)
        for name, digest in labels["filesSha256"].items():
            self.assertEqual(sha256((case / name).read_bytes()).hexdigest(), digest)
        self.assertFalse((self.output / "2_30").exists())
        self.assertFalse((self.output / "3_30").exists())
        self.assertEqual(MODULE.prepare_training_reference(self.path, self.output), identities)  # Identical rerun is safe.

    def test_batches_are_deterministic_and_cannot_silently_merge_old_cases(self):
        self.assertEqual(MODULE.prepare_training_reference(self.path, self.output, limit=1, offset=1), ["1_60"])
        index = json.loads((self.output / "sample-index.json").read_text())
        self.assertEqual((index["offset"], index["limit"], index["availableTrainingFrames"]), (1, 1, 2))
        before = {path.relative_to(self.output): path.read_bytes() for path in self.output.rglob("*") if path.is_file()}
        with self.assertRaisesRegex(MODULE.ReferenceError, "different reference pack"):
            MODULE.prepare_training_reference(self.path, self.output, limit=1, offset=0)
        self.assertEqual({path.relative_to(self.output): path.read_bytes() for path in self.output.rglob("*") if path.is_file()}, before)
        for limit, offset in ((41, 0), (0, 0), (1, -1), (1, 2)):
            with self.assertRaises(MODULE.ReferenceError):
                MODULE.prepare_training_reference(self.path, self.folder / "unused", limit=limit, offset=offset)

    def test_explicit_relocation_uses_manifest_root_not_a_filename_guess(self):
        old = "/Users/person1/work/holospex/ml/data/endoscapes"
        for entry in self.manifest["samples"]:
            for key in ("imagePath", "maskPath"):
                relative = Path(entry[key]).relative_to(self.data).as_posix()
                entry[key] = old + "/" + relative
        self.manifest["root"] = old
        self.write_manifest()
        self.assertEqual(MODULE.prepare_training_reference(self.path, self.output, data_root=self.data), ["1_30", "1_60"])
        self.manifest["samples"][-1]["imagePath"] = "/unrelated/all/1_30.jpg"
        self.write_manifest()
        with self.assertRaisesRegex(MODULE.ReferenceError, "outside the manifest root"):
            MODULE.prepare_training_reference(self.path, self.folder / "bad", data_root=self.data)
        self.assertFalse((self.folder / "bad").exists())

    def test_unknown_ids_unconfigured_ignore_wrong_mapping_and_split_leakage_fail(self):
        original = json.loads(json.dumps(self.manifest))
        for mutate in (
            lambda value: value["classes"][1].update(sourceId=4),
            lambda value: value.update(ignoreSourceIds=[7, 255]),
            lambda value: value["report"]["splitVideoIds"]["val"].append(1),
            lambda value: value["samples"][-1].update(split="test"),
        ):
            self.manifest = json.loads(json.dumps(original)); mutate(self.manifest); self.write_manifest()
            with self.assertRaises(MODULE.ReferenceError):
                MODULE.prepare_training_reference(self.path, self.output)
            self.assertFalse(self.output.exists())
        self.manifest = original; self.write_manifest()
        Image.frombytes("L", (5, 3), bytes([7]) + self.raw[1:]).save(self.data / "semseg/1_30.png")
        with self.assertRaisesRegex(MODULE.ReferenceError, "unknown source mask IDs"):
            MODULE.prepare_training_reference(self.path, self.output)
        Image.frombytes("L", (5, 3), self.raw).save(self.data / "semseg/1_30.png")
        self.manifest["ignoreSourceIds"] = []; self.write_manifest()
        with self.assertRaisesRegex(MODULE.ReferenceError, "255"):
            MODULE.prepare_training_reference(self.path, self.output)

    def test_palette_indices_are_retained_and_colored_rgb_masks_are_rejected(self):
        palette = Image.frombytes("P", (5, 3), self.raw)
        palette.putpalette([value for index in range(256) for value in (255 - index, index, 20)])
        palette.save(self.data / "semseg/1_30.png")
        MODULE.prepare_training_reference(self.path, self.output)
        with Image.open(self.output / "1_30/labels-index.png") as mask:
            self.assertEqual(mask.tobytes()[1], 1)
            self.assertEqual(mask.tobytes()[8], 255)
        Image.new("RGB", (5, 3), (5, 4, 5)).save(self.data / "semseg/1_30.png")
        with self.assertRaisesRegex(MODULE.ReferenceError, "RGB mask channels differ"):
            MODULE.prepare_training_reference(self.path, self.folder / "rgb")
        Image.new("L", (6, 3), 5).save(self.data / "semseg/1_30.png")
        with self.assertRaisesRegex(MODULE.ReferenceError, "dimensions"):
            MODULE.prepare_training_reference(self.path, self.folder / "size")

    @unittest.skipUnless(shutil.which("node"), "Node is required for the browser-boundary integration check")
    def test_pack_passes_browser_import_and_exact_mask_validation(self):
        MODULE.prepare_training_reference(self.path, self.output)
        javascript = """
import assert from 'node:assert/strict';
import { readdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { importDatasetSamples } from './apps/web/src/overlays/datasetSamples.ts';
import { decodeIndexMask } from './apps/web/src/overlays/decodeIndexMask.ts';
import { buildDatasetRaster } from './apps/web/src/overlays/datasetRaster.ts';
const files = [];
async function collect(directory) {
  for (const item of await readdir(directory, { withFileTypes: true })) {
    const path = join(directory, item.name);
    if (item.isDirectory()) await collect(path);
    else files.push(new File([await readFile(path)], item.name));
  }
}
await collect(process.argv[1]);
const imported = await importDatasetSamples(files);
assert.deepEqual(imported.issues, []);
assert.equal(imported.samples.length, 2);
for (const sample of imported.samples) {
  assert.equal(sample.labels.annotationSource.split, 'train');
  const { width, height } = sample.labels.frame;
  const raster = buildDatasetRaster(sample.labels, await decodeIndexMask(sample.indexMask, width, height));
  assert.equal(raster.pixels[8], 255);
  assert.equal(raster.pixels[1], 1);
}
"""
        result = subprocess.run([shutil.which("node"), "--import", "tsx", "--input-type=module", "-e", javascript, str(self.output)],
                                cwd=ROOT, text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
