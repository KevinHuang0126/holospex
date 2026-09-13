import assert from "node:assert/strict";
import test from "node:test";
import { anatomy } from "@holospex/contracts";
import { parseDatasetSample, sha256, type LoadedDatasetSample } from "../src/overlays/datasetSamples";
import { exportPortableSample, importPortableSample, PORTABLE_SAMPLE_MAX_BYTES } from "../src/overlays/portableSample";

async function fixture(): Promise<LoadedDatasetSample> {
  // Synthetic file headers exercise transfer/integrity. Rendering separately validates image decoding.
  const image = new Uint8Array([255, 216, 255, 224]);
  const mask = new Uint8Array(33); mask.set([137, 80, 78, 71, 13, 10, 26, 10]);
  const header = new DataView(mask.buffer); header.setUint32(16, 5); header.setUint32(20, 3); mask[24] = 8;
  const labels = parseDatasetSample({
    artifactType: "dataset_annotation_sample",
    annotationSource: { dataset: "Endoscapes-Seg50", kind: "supplied_dataset_annotation", split: "train",
      sourceUrl: "https://github.com/CAMMA-public/Endoscapes", videoId: 1, sourceFrameNumber: 2 },
    frame: { mediaId: "endoscapes-still-1_2", frameNumber: 0, timestampMs: 0, width: 5, height: 3, coordinateSpace: "original_pixels" },
    classes: ["background", ...Object.keys(anatomy)].map((structureId, index) => ({ index, structureId, sourceId: [0, 5, 4, 3, 1, 2, 6][index] })),
    raster: { dtype: "uint8", shape: [3, 5], ignoreValue: 255, ignoredPixelCount: 0,
      pixelCounts: Object.fromEntries(Object.keys(anatomy).map(id => [id, id === "gallbladder" ? 10 : 0])) },
    structures: [], conversion: { geometry: "Synthetic fixture", visibility: "Synthetic fixture", withheldComponents: {} },
    filesSha256: { "image.jpg": await sha256(image.buffer), "labels-index.png": await sha256(mask.buffer) },
  });
  return { id: "1_2", labels, image: new Blob([image]), indexMask: new Blob([mask]) };
}
const jsonFile = (value: unknown) => new File([JSON.stringify(value)], "sample.holospex.json");

test("portable sample round trip preserves source annotation, original bytes and exact mask hashes", async () => {
  const original = await fixture();
  const file = await exportPortableSample(original);
  const result = await importPortableSample(file);
  assert.deepEqual(result.issues, []);
  assert.equal(result.samples.length, 1);
  const imported = result.samples[0];
  assert.deepEqual(imported.labels, original.labels);
  assert.equal(imported.labels.annotationSource.kind, "supplied_dataset_annotation");
  assert.equal(imported.id, original.id);
  assert.equal(imported.image.type, "image/jpeg");
  assert.equal(imported.indexMask.type, "image/png");
  assert.deepEqual(await imported.image.arrayBuffer(), await original.image.arrayBuffer());
  assert.deepEqual(await imported.indexMask.arrayBuffer(), await original.indexMask.arrayBuffer());
});

test("portable import withholds tampered assets and retains the existing provenance gate", async () => {
  const envelope = JSON.parse(await (await exportPortableSample(await fixture())).text());
  for (const key of ["imageBase64", "indexMaskBase64"]) {
    const changed = structuredClone(envelope);
    changed[key] = btoa("changed bytes");
    const result = await importPortableSample(jsonFile(changed));
    assert.equal(result.samples.length, 0);
    assert.match(result.issues.join(" "), /hash mismatch/);
  }
  envelope.labels.annotationSource.kind = "reviewed_annotation";
  const result = await importPortableSample(jsonFile(envelope));
  assert.equal(result.samples.length, 0);
  assert.match(result.issues.join(" "), /annotationSource/);
});

test("portable import rejects malformed envelopes and noncanonical base64 without fetching URLs", async () => {
  const envelope = JSON.parse(await (await exportPortableSample(await fixture())).text());
  await assert.rejects(importPortableSample(new File(["{broken"], "sample.holospex.json")), /malformed JSON/);
  for (const value of [null, [], {}, { ...envelope, version: 2 }, { ...envelope, artifactType: "other" }])
    await assert.rejects(importPortableSample(jsonFile(value)), /format|sample file/);
  for (const imageBase64 of ["", "https://example.com/image.jpg", "data:image/jpeg;base64,/9j/4A==", "/9j/4A==\n", "/9j/4B==", "=AAA", "AAA"])
    await assert.rejects(importPortableSample(jsonFile({ ...envelope, imageBase64 })), /base64/);
});

test("portable sample size limits apply before reading and to actual contents and export", async () => {
  let wasRead = false;
  await assert.rejects(importPortableSample({ size: PORTABLE_SAMPLE_MAX_BYTES + 1, text: async () => { wasRead = true; return "{}"; } }), /20 MB/);
  assert.equal(wasRead, false);
  await assert.rejects(importPortableSample({ size: 1, text: async () => " ".repeat(PORTABLE_SAMPLE_MAX_BYTES + 1) }), /20 MB/);
  const sample = await fixture();
  sample.image = new Blob([new Uint8Array(PORTABLE_SAMPLE_MAX_BYTES + 1)]);
  await assert.rejects(exportPortableSample(sample), /20 MB/);
});

test("portable export refuses a sample whose image no longer matches its supplied hash", async () => {
  const sample = await fixture();
  sample.image = new Blob([new Uint8Array([255, 216, 255, 225])]);
  await assert.rejects(exportPortableSample(sample), /hash mismatch/);
});
