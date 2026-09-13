import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { readdir, readFile, stat } from "node:fs/promises";
import test from "node:test";
import { anatomy, parseFrameResult } from "@holospex/contracts";
import { importDatasetSamples, parseDatasetSample, sha256 } from "../src/overlays/datasetSamples";
import { buildDatasetRaster, datasetOverlayState, selectDatasetPixel } from "../src/overlays/datasetRaster";
import { drawHud, type HudScene } from "../src/overlays/drawHud";

const pixels = [1, 1, 1, 1, 0, 1, 255, 0, 1, 0, 1, 1, 1, 1, 0];
function labels() {
  return {
    artifactType: "dataset_annotation_sample",
    annotationSource: { dataset: "Endoscapes-Seg50", kind: "supplied_dataset_annotation", split: "train",
      sourceUrl: "https://github.com/CAMMA-public/Endoscapes", videoId: 1, sourceFrameNumber: 2 },
    frame: { mediaId: "endoscapes-still-1_2", frameNumber: 0, timestampMs: 0, width: 5, height: 3, coordinateSpace: "original_pixels" },
    classes: ["background", ...Object.keys(anatomy)].map((structureId, index) => ({ index, structureId, sourceId: [0, 5, 4, 3, 1, 2, 6][index] })),
    raster: { dtype: "uint8", shape: [3, 5], ignoreValue: 255, ignoredPixelCount: 1,
      pixelCounts: Object.fromEntries(Object.keys(anatomy).map(id => [id, id === "gallbladder" ? 10 : 0])) },
    structures: [], // A component with a hole can be withheld from polygon exports.
    conversion: { geometry: "Approximate; holes withheld.", visibility: "Border contact only.", withheldComponents: { holes: 1 } },
    filesSha256: { "image.jpg": "a".repeat(64), "labels-index.png": "b".repeat(64) },
  };
}
const rgba = () => new Uint8ClampedArray(pixels.flatMap(value => [value, value, value, 255]));

test("dataset provenance remains separate from predictions and reviewed lesson answers", () => {
  const sample = parseDatasetSample(labels());
  assert.equal(sample.annotationSource.kind, "supplied_dataset_annotation");
  assert.throws(() => parseFrameResult(sample));
  for (const kind of ["ml_prediction", "reviewed_annotation", "synthetic_mock"]) {
    const value = labels(); value.annotationSource.kind = kind;
    assert.throws(() => parseDatasetSample(value));
  }
  for (const mode of ["identify", "assess"] as const) {
    for (const view of ["raster", "polygons"] as const) assert.deepEqual(datasetOverlayState(true, mode, view), { show: false, warning: null });
  }
  assert.equal(datasetOverlayState(true, "feedback", "raster").show, false);
  assert.match(datasetOverlayState(true, "feedback", "raster").warning!, /reviewed lesson feedback/);
  assert.equal(datasetOverlayState(false, "learn", "raster").show, false);
});

test("sample parser rejects a video timebase, contradictory identity, dimensions and class remapping", () => {
  const wrongTime = labels(); wrongTime.frame.timestampMs = 1000;
  const wrongFrame = labels(); wrongFrame.frame.frameNumber = 2;
  const wrongIdentity = labels(); wrongIdentity.annotationSource.videoId = 4;
  const wrongShape = labels(); wrongShape.raster.shape = [5, 3];
  const wrongMapping = labels(); wrongMapping.classes[1].sourceId = 1;
  const duplicateMapping = labels(); duplicateMapping.classes[2] = duplicateMapping.classes[1];
  for (const value of [wrongTime, wrongFrame, wrongIdentity, wrongShape, wrongMapping, duplicateMapping]) assert.throws(() => parseDatasetSample(value));
});

test("exact masks retain regions omitted by polygons, holes and ignored pixels; label anchors lie on anatomy", () => {
  const sample = parseDatasetSample(labels()), result = buildDatasetRaster(sample, rgba());
  assert.equal(sample.structures.length, 0);
  assert.equal(result.anchors.length, 1);
  assert.deepEqual(Array.from(result.fill.slice(0, 4)), [69, 214, 170, 255]);
  for (const p of [4, 6, 7]) {
    assert.equal(result.fill[p * 4 + 3], 0);
    assert.equal(result.outline[p * 4 + 3], 0);
  }
  assert.equal(result.outline[5 * 4 + 3], 255);
  const anchor = result.anchors[0];
  assert.equal(selectDatasetPixel(sample, result.pixels, [anchor.x, anchor.y])?.structureId, "gallbladder");
  assert.deepEqual(selectDatasetPixel(sample, result.pixels, [1.5, 1.5]), { structureId: null, status: "ignored" });
  assert.deepEqual(selectDatasetPixel(sample, result.pixels, [2.5, 1.5]), { structureId: null, status: "background" });
  for (const point of [[-1, 0], [5, 0], [0, 3], [NaN, 0]] as [number, number][]) assert.equal(selectDatasetPixel(sample, result.pixels, point), null);
  const altered = rgba(); altered[0] = 2;
  assert.throws(() => buildDatasetRaster(sample, altered), /unsupported or altered/);
  const wrongCounts = parseDatasetSample(labels()); wrongCounts.raster.pixelCounts.gallbladder = 9;
  assert.throws(() => buildDatasetRaster(wrongCounts, rgba()), /pixel counts/);
  const wrongIgnore = parseDatasetSample(labels()); wrongIgnore.raster.ignoredPixelCount = 0;
  assert.throws(() => buildDatasetRaster(wrongIgnore, rgba()), /ignored-pixel count/);
});

async function syntheticPack() {
  // Minimal headers suffice for importer tests; actual image decoding is the renderer's next gate.
  const image = new Uint8Array([255, 216, 255, 224]);
  const mask = new Uint8Array(33); mask.set([137, 80, 78, 71, 13, 10, 26, 10]);
  const header = new DataView(mask.buffer); header.setUint32(16, 5); header.setUint32(20, 3); mask[24] = 8;
  const value = labels(); value.filesSha256 = { "image.jpg": await sha256(image.buffer), "labels-index.png": await sha256(mask.buffer) };
  return [new File([image], "wrong.json"), new File([mask], "wrong.jpg"), new File([JSON.stringify(value)], "wrong.png")];
}
test("import resolves renamed/flattened files by hash and withholds missing or corrupted counterparts", async () => {
  const files = await syntheticPack();
  const result = await importDatasetSamples(files);
  assert.equal(result.samples.length, 1);
  assert.equal(result.samples[0].image.type, "image/jpeg");
  assert.equal(result.samples[0].indexMask.type, "image/png");
  assert.deepEqual(result.issues, []);
  assert.equal((await importDatasetSamples(files.slice(1))).samples.length, 0);
  const corrupted = [new File([new Uint8Array([255, 216, 255, 225])], "wrong.json"), ...files.slice(1)];
  assert.match((await importDatasetSamples(corrupted)).issues.join(" "), /hash mismatch/);
  const index = new File([JSON.stringify({ samples: [{ id: "4_36700" }] })], "index.txt");
  assert.match((await importDatasetSamples([...files, index])).issues.join(" "), /4_36700.*Unable to assess/);
});
test("conflicting label records with the same still identity cannot silently select a winner", async () => {
  const files = await syntheticPack();
  const duplicate = JSON.parse(await files[2].text()); duplicate.conversion.geometry = "Different export";
  const result = await importDatasetSamples([...files, new File([JSON.stringify(duplicate)], "second.json")]);
  assert.equal(result.samples.length, 0); assert.match(result.issues.join(" "), /Conflicting label records/);
});

test("hiding dataset answers keeps image alignment stable and removes raster, leader lines and anatomical labels", () => {
  const image = {} as CanvasImageSource, fill = {} as CanvasImageSource, outline = {} as CanvasImageSource;
  const calls: { name: string; args: unknown[] }[] = [];
  const context = new Proxy({}, { get: (_, name: string) => (...args: unknown[]) => {
    calls.push({ name, args });
    if (name === "measureText") return { width: String(args[0]).length * 9 };
  } });
  for (const width of [400, 960, 1200]) {
    const canvas = { width: 0, height: 0, clientWidth: width, getContext: () => context } as unknown as HTMLCanvasElement;
    const scene: HudScene = { width: 854, height: 480, sourceLabel: "Supplied dataset annotation", labelSlots: 6,
      structures: [], anchors: [{ id: "gallbladder", structureId: "gallbladder", x: 200, y: 200 }], raster: { fill, outline }, warning: null };
    calls.length = 0;
    const shown = drawHud(canvas, image, scene);
    assert.equal(calls.filter(call => call.name === "drawImage").length, 3);
    assert.ok(calls.some(call => call.name === "fillText" && call.args[0] === "Gallbladder"));
    calls.length = 0;
    const hidden = drawHud(canvas, image, { ...scene, anchors: [], raster: undefined });
    assert.deepEqual(hidden, shown);
    assert.equal(calls.filter(call => call.name === "drawImage").length, 1);
    assert.ok(!calls.some(call => call.name === "lineTo" || (call.name === "fillText" && call.args[0] === "Gallbladder")));
  }
});

test("fill, boundaries and labels switch independently without changing image alignment", () => {
  const image = {} as CanvasImageSource, fill = {} as CanvasImageSource, outline = {} as CanvasImageSource;
  const calls: { name: string; args: unknown[]; alpha: number }[] = [];
  const properties: Record<string, unknown> = { globalAlpha: 1 };
  const context = new Proxy(properties, { get: (target, name: string) => {
    if (name in target) return target[name];
    return (...args: unknown[]) => {
      calls.push({ name, args, alpha: Number(target.globalAlpha) });
      if (name === "measureText") return { width: String(args[0]).length * 9 };
    };
  } });
  const canvas = { width: 0, height: 0, clientWidth: 960, getContext: () => context } as unknown as HTMLCanvasElement;
  const scene: HudScene = { width: 854, height: 480, sourceLabel: "Supplied dataset annotation", labelSlots: 6,
    structures: [], anchors: [{ id: "gallbladder", structureId: "gallbladder", x: 200, y: 200 }], raster: { fill, outline }, warning: null };
  const originalLayout = drawHud(canvas, image, scene);
  for (const opacity of [0, 0.65, 1, -1, 2, NaN]) {
    calls.length = 0;
    const layout = drawHud(canvas, image, { ...scene, appearance: { fillOpacity: opacity, showBoundaries: false, showLabels: false } });
    assert.deepEqual(layout, originalLayout);
    assert.equal(calls.find(call => call.name === "drawImage" && call.args[0] === image)?.alpha, 1);
    const expectedOpacity = Number.isNaN(opacity) ? 0.28 : Math.max(0, Math.min(1, opacity));
    const fillCall = calls.find(call => call.name === "drawImage" && call.args[0] === fill);
    if (!expectedOpacity) assert.equal(fillCall, undefined); else assert.equal(fillCall?.alpha, expectedOpacity);
    assert.ok(!calls.some(call => call.name === "drawImage" && call.args[0] === outline));
    assert.ok(!calls.some(call => call.name === "fillText" && call.args[0] === "Gallbladder"));
  }
  calls.length = 0;
  drawHud(canvas, image, { ...scene, appearance: { fillOpacity: 0, showBoundaries: true, showLabels: false } });
  assert.deepEqual(calls.filter(call => call.name === "drawImage").map(call => call.args[0]), [image, outline]);
  calls.length = 0;
  drawHud(canvas, image, { ...scene, appearance: { fillOpacity: 0, showBoundaries: false, showLabels: true } });
  assert.deepEqual(calls.filter(call => call.name === "drawImage").map(call => call.args[0]), [image]);
  assert.ok(calls.some(call => call.name === "fillText" && call.args[0] === "Gallbladder"));
});

test("thicker raster boundaries stay inside annotated pixels and preserve interior holes", () => {
  const value = labels();
  value.frame.width = 11; value.frame.height = 11; value.raster.shape = [11, 11]; value.raster.pixelCounts.gallbladder = 120;
  const native = Array(121).fill(1); native[60] = 255;
  const sample = parseDatasetSample(value);
  const result = buildDatasetRaster(sample, new Uint8ClampedArray(native.flatMap(value => [value, value, value, 255])));
  const alphaAt = (x: number, y: number) => result.outline[(y * 11 + x) * 4 + 3];
  assert.equal(alphaAt(5, 5), 0); // Ignored hole.
  assert.equal(alphaAt(2, 2), 0); // Interior, beyond both boundary bands.
  assert.equal(alphaAt(1, 2), 255); // Second row inside the image edge.
  assert.equal(alphaAt(3, 5), 255); // Second pixel from the ignored hole.
});

const localPack = new URL("./samples_tst/", import.meta.url);
test("local Person 1 handoff: import every usable case without relying on misleading filenames", { skip: !existsSync(localPack) }, async () => {
  const entries = await readdir(localPack);
  const files: File[] = [];
  for (const name of entries) {
    const path = new URL(name, localPack);
    // OneDrive placeholders may report as reparse points in readdir.
    if ((await stat(path)).isFile()) files.push(new File([await readFile(path)], name));
  }
  const imported = await importDatasetSamples(files);
  assert.ok(imported.samples.length > 0, imported.issues.join("\n"));
  for (const sample of imported.samples) {
    assert.equal(await sha256(await sample.image.arrayBuffer()), sample.labels.filesSha256["image.jpg"]);
    assert.equal(await sha256(await sample.indexMask.arrayBuffer()), sample.labels.filesSha256["labels-index.png"]);
    assert.equal(sample.labels.frame.frameNumber, 0);
  }
});
