import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtemp, writeFile, mkdir, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import test from "node:test";
import { readLocalSampleFiles } from "../dev/localSamples";
import { loadLocalDatasetSamples } from "../src/overlays/localSampleInput";

test("local server exposes only sample metadata and bytes referenced by supplied image/mask hashes", async t => {
  const temporaryRoot = resolve(tmpdir()), folder = await mkdtemp(join(temporaryRoot, "holospex-samples-"));
  t.after(async () => {
    assert.equal(dirname(resolve(folder)), temporaryRoot);
    assert.ok(folder.includes("holospex-samples-"));
    await rm(folder, { recursive: true, force: true });
  });
  await mkdir(join(folder, "case"));
  const image = Buffer.from([255, 216, 255]), mask = Buffer.from([137, 80, 78, 71]);
  const hash = (value: Buffer) => createHash("sha256").update(value).digest("hex");
  await writeFile(join(folder, "case", "wrong.json"), image);
  await writeFile(join(folder, "case", "wrong.jpg"), mask);
  await writeFile(join(folder, "labels.png"), JSON.stringify({ artifactType: "dataset_annotation_sample", filesSha256: { "image.jpg": hash(image), "labels-index.png": hash(mask) } }));
  await writeFile(join(folder, "index.txt"), JSON.stringify({ samples: [{ id: "1_2" }] }));
  await writeFile(join(folder, "unrelated.txt"), "Unrelated file stays out of the manifest.");
  await writeFile(join(folder, "unreferenced.png"), Buffer.from([137, 80, 78, 71, 10]));
  const files = await readLocalSampleFiles(folder);
  assert.deepEqual(files.map(file => file.name).sort(), ["index.txt", "labels.png", "wrong.jpg", "wrong.json"]);
  assert.equal(new Set(files.map(file => file.hash)).size, files.length);
});

test("automatic import refuses a manifest that redirects sample reads outside the local endpoint", async t => {
  let requests = 0;
  t.mock.method(globalThis, "fetch", async () => {
    requests += 1;
    return Response.json({ source: "training_reference", split: "train",
      files: [{ name: "image.jpg", url: "https://example.com/image.jpg" }] });
  });
  await assert.rejects(loadLocalDatasetSamples(new AbortController().signal), /manifest is invalid/);
  assert.equal(requests, 1);
});

test("automatic import provides a folder-picker fallback when served without the dev sample endpoint", async t => {
  t.mock.method(globalThis, "fetch", async () => new Response("<!doctype html>", { headers: { "Content-Type": "text/html" } }));
  await assert.rejects(loadLocalDatasetSamples(new AbortController().signal), /choose the folder/);
});
