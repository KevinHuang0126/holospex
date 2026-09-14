import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import test from "node:test";
import { anatomy } from "@holospex/contracts";
import { readLocalTrainingFiles, trainingReferenceDirectory } from "../dev/localTraining";
import { importDatasetSamples } from "../src/overlays/datasetSamples";
import { loadLocalDatasetSamples, trainingReferencesOnly } from "../src/overlays/localSampleInput";
import { encodeIndexMaskPng } from "./helpers/indexMaskPng";

function fixture(split: string, videoId: number) {
  // Tiny synthetic importer fixture only; this is not a training data export.
  const image = Buffer.from([255, 216, 255, 224, videoId]);
  const indices = [1, 1, 1, 1, 0, 1, 255, 0, 1, 0, 1, 1, 1, 1, 0];
  const rotated = [...indices.slice(videoId), ...indices.slice(0, videoId)];
  const mask = Buffer.from(encodeIndexMaskPng(5, 3, rotated));
  const hash = (bytes: Buffer) => createHash("sha256").update(bytes).digest("hex");
  const labels = {
    artifactType: "dataset_annotation_sample", annotationSource: { dataset: "Endoscapes-Seg50", kind: "supplied_dataset_annotation", split,
      sourceUrl: "https://github.com/CAMMA-public/Endoscapes", videoId, sourceFrameNumber: 2 },
    frame: { mediaId: `endoscapes-still-${videoId}_2`, frameNumber: 0, timestampMs: 0, width: 5, height: 3, coordinateSpace: "original_pixels" },
    classes: ["background", ...Object.keys(anatomy)].map((structureId, index) => ({ index, structureId, sourceId: [0, 5, 4, 3, 1, 2, 6][index] })),
    raster: { dtype: "uint8", shape: [3, 5], ignoreValue: 255, ignoredPixelCount: 1,
      pixelCounts: Object.fromEntries(Object.keys(anatomy).map(id => [id, id === "gallbladder" ? 10 : 0])) },
    structures: [], conversion: { geometry: "Exact raster only", visibility: "Not assessed", withheldComponents: {} },
    filesSha256: { "image.jpg": hash(image), "labels-index.png": hash(mask) },
  };
  return [
    { name: `${split}-image-${videoId}.jpg`, bytes: image },
    { name: `${split}-mask-${videoId}.png`, bytes: mask },
    { name: `${split}-labels-${videoId}.json`, bytes: Buffer.from(JSON.stringify(labels)) },
  ].map(file => ({ ...file, hash: hash(file.bytes) }));
}
const filesAsBrowserInput = (files: ReturnType<typeof fixture>) => files.map(file => new File([file.bytes], file.name));

test("training selection preserves declared provenance and excludes validation, test and unknown splits", async () => {
  const files = [...fixture("train", 1), ...fixture("val", 2), ...fixture("test", 3), ...fixture("unspecified", 4)];
  const original = await importDatasetSamples(filesAsBrowserInput(files));
  assert.equal(original.samples.length, 4);
  const training = trainingReferencesOnly(original);
  assert.deepEqual(training.samples.map(sample => sample.id), ["1_2"]);
  assert.equal(training.samples[0].labels.annotationSource.split, "train");
  assert.equal(training.samples[0].labels.annotationSource.kind, "supplied_dataset_annotation");
  assert.equal(original.samples.length, 4, "The gate must not mutate imported data or change its split");
  assert.equal(training.issues.filter(issue => issue.includes("excluded")).length, 3);
  const noTraining = trainingReferencesOnly({ samples: original.samples.slice(1), issues: [] });
  assert.equal(noTraining.samples.length, 0); assert.match(noTraining.issues.join(" "), /not used as a fallback/);
  assert.deepEqual(trainingReferencesOnly(noTraining), noTraining, "Repeated guards must not duplicate unavailable messages");
});

test("the local training server exposes complete training pairs and withholds nontraining or corrupted pairs", async t => {
  const temporaryRoot = resolve(tmpdir()), folder = await mkdtemp(join(temporaryRoot, "holospex-training-"));
  t.after(async () => {
    assert.equal(dirname(resolve(folder)), temporaryRoot); assert.ok(folder.includes("holospex-training-"));
    await rm(folder, { recursive: true, force: true });
  });
  await mkdir(join(folder, "references"));
  const train = fixture("train", 1), other = [...fixture("val", 2), ...fixture("test", 3)];
  for (const file of [...train, ...other]) await writeFile(join(folder, "references", file.name), file.bytes);
  await writeFile(join(folder, "unreferenced-private.txt"), "Must not be served.");
  await writeFile(join(folder, "sample-index.json"), JSON.stringify({ samples: [{ id: "1_2" }, { id: "2_2" }, { id: "3_2" }] }));
  const selected = await readLocalTrainingFiles(folder);
  assert.deepEqual(selected.map(file => file.name).sort(), train.map(file => file.name).sort());
  assert.deepEqual(selected.map(file => file.hash).sort(), train.map(file => file.hash).sort());
  await writeFile(join(folder, "references", train[0].name), Buffer.from([255, 216, 255, 225]));
  await assert.rejects(readLocalTrainingFiles(folder), /hash mismatch/);
});

test("automatic training import rechecks the envelope split even when a manifest claims train", async t => {
  const files = [...fixture("train", 1), ...fixture("test", 3)];
  const assets = new Map(files.map(file => [`/__local-training/${file.hash}`, file]));
  let requests = 0;
  t.mock.method(globalThis, "fetch", async (url: string) => {
    requests++;
    if (url === "/__local-training/manifest") return Response.json({ source: "training_reference", split: "train",
      files: files.map(file => ({ name: file.name, url: `/__local-training/${file.hash}` })) });
    const file = assets.get(url);
    return file ? new Response(file.bytes) : new Response(null, { status: 404 });
  });
  const pack = await loadLocalDatasetSamples(new AbortController().signal);
  assert.deepEqual(pack.samples.map(sample => sample.id), ["1_2"]);
  assert.match(pack.issues.join(" "), /test split excluded/);
  assert.equal(requests, files.length + 1);
});

test("default training paths never refer to test fixtures and overrides resolve against the repo", () => {
  const root = resolve("training-path-test-root");
  assert.equal(trainingReferenceDirectory(root), join(root, "runs", "training-reference"));
  assert.equal(trainingReferenceDirectory(root, "prepared/from-person-1"), join(root, "prepared", "from-person-1"));
  const external = resolve(tmpdir(), "prepared-endoscapes-reference");
  assert.equal(trainingReferenceDirectory(root, external), external);
  assert.ok(!trainingReferenceDirectory(root).includes("samples_tst"));
});
