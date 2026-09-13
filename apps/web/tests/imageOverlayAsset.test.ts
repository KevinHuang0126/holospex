import assert from "node:assert/strict";
import test from "node:test";
import { anatomy } from "@holospex/contracts";
import { parseDatasetSample, type LoadedDatasetSample } from "../src/overlays/datasetSamples";
import { composeImageOverlayPixels, loadImageOverlayAsset } from "../src/camera/imageOverlayAsset";
import { encodeIndexMaskPng } from "./helpers/indexMaskPng";

function fixture() {
  const width = 11, height = 11, indices = Array<number>(width * height).fill(0);
  for (let y = 1; y < 10; y++) for (let x = 1; x < 10; x++) indices[y * width + x] = 1;
  indices[0] = 6; indices[60] = 255; indices[108] = 2; indices[97] = 3;
  const classes = ["background", ...Object.keys(anatomy)].map((structureId, index) => ({ index, structureId, sourceId: [0, 5, 4, 3, 1, 2, 6][index] }));
  const labels = parseDatasetSample({
    artifactType: "dataset_annotation_sample",
    annotationSource: { dataset: "Endoscapes-Seg50", kind: "supplied_dataset_annotation", split: "train",
      sourceUrl: "https://github.com/CAMMA-public/Endoscapes", videoId: 1, sourceFrameNumber: 2 },
    frame: { mediaId: "endoscapes-still-1_2", frameNumber: 0, timestampMs: 0, width, height, coordinateSpace: "original_pixels" },
    classes,
    raster: { dtype: "uint8", shape: [height, width], ignoreValue: 255, ignoredPixelCount: 1,
      pixelCounts: Object.fromEntries(classes.filter(item => item.index > 0).map(item => [item.structureId, indices.filter(index => index === item.index).length])) },
    structures: [], conversion: { geometry: "Native exact mask", visibility: "Supplied", withheldComponents: {} },
    filesSha256: { "image.jpg": "a".repeat(64), "labels-index.png": "b".repeat(64) },
  });
  return { labels, indices, original: new Uint8ClampedArray(indices.flatMap(() => [110, 65, 40, 255])),
    mask: new Uint8ClampedArray(indices.flatMap(value => [value, value, value, 255])) };
}

test("camera image textures retain photographed tissue, exact mask holes and source-aligned anchors", () => {
  const { labels, indices, original, mask } = fixture();
  const originalCopy = original.slice(), maskCopy = mask.slice();
  const result = composeImageOverlayPixels(labels, original, mask);
  const pixel = (rgba: Uint8ClampedArray, p: number) => Array.from(rgba.slice(p * 4, p * 4 + 4));
  assert.deepEqual(pixel(result.scene.rgba, 1), [110, 65, 40, 255]); // Unlabeled photograph unchanged.
  assert.deepEqual(pixel(result.scene.rgba, 60), [110, 65, 40, 255]); // Ignore retains original scene.
  assert.deepEqual(pixel(result.scene.rgba, 36), [105, 83, 56, 255]); // Only 12% tint in tissue interior.
  for (const p of [0, 1, 60]) assert.deepEqual(pixel(result.anatomy.rgba, p), [0, 0, 0, 0]); // Tool, background, ignored hole.
  for (const p of [12, 36, 97, 108]) assert.deepEqual(pixel(result.anatomy.rgba, p), pixel(result.scene.rgba, p));
  assert.ok(result.scene.anchors.some(anchor => anchor.structureId === "tool"));
  assert.ok(result.anatomy.anchors.length === 3 && result.anatomy.anchors.every(anchor => anchor.structureId !== "tool"));
  for (const anchor of result.anatomy.anchors) {
    const p = Math.floor(anchor.y) * labels.frame.width + Math.floor(anchor.x);
    assert.equal(labels.classes.find(item => item.index === indices[p])?.structureId, anchor.structureId);
    assert.equal(result.anatomy.rgba[p * 4 + 3], 255);
  }
  assert.deepEqual(original, originalCopy); assert.deepEqual(mask, maskCopy);
});

test("camera texture composition rejects altered masks and contradictory source pixels", () => {
  const { labels, original, mask } = fixture();
  assert.throws(() => composeImageOverlayPixels(labels, original.subarray(4), mask), /dimensions/);
  const altered = mask.slice(); altered[0] = 1;
  assert.throws(() => composeImageOverlayPixels(labels, original, altered), /altered pixel values/);
  const wrongCounts = structuredClone(labels); wrongCounts.raster.pixelCounts.gallbladder--;
  assert.throws(() => composeImageOverlayPixels(wrongCounts, original, mask), /pixel counts/);
  const transparent = original.slice(); transparent[3] = 0;
  assert.throws(() => composeImageOverlayPixels(labels, transparent, mask), /opaque image/);
});

test("exact anatomy regions retain asymmetric mask geometry and holes omitted from polygon exports", () => {
  const { labels } = fixture();
  const width = 5, height = 4;
  const indices = [
    0, 1, 1, 1, 6,
    0, 1, 255, 1, 0,
    2, 1, 1, 0, 0,
    2, 0, 0, 0, 0,
  ];
  labels.frame.width = width; labels.frame.height = height;
  labels.raster.shape = [height, width];
  labels.raster.pixelCounts = Object.fromEntries(labels.classes.filter(item => item.structureId !== "background")
    .map(item => [item.structureId, indices.filter(value => value === item.index).length])) as typeof labels.raster.pixelCounts;
  labels.conversion.withheldComponents = { holes: 1 };
  const sample = parseDatasetSample(labels);
  assert.equal(sample.structures.length, 0);
  const original = new Uint8ClampedArray(indices.flatMap(() => [110, 65, 40, 255]));
  const mask = new Uint8ClampedArray(indices.flatMap(value => [value, value, value, 255]));
  const { regions } = composeImageOverlayPixels(sample, original, mask);
  assert.deepEqual(regions.map(region => region.structureId), ["gallbladder", "cystic_duct"]);
  const [gallbladder, duct] = regions;
  assert.deepEqual(gallbladder.bounds, { x: 1 / 5, y: 0, width: 3 / 5, height: 3 / 4 });
  assert.equal(gallbladder.pixelCount, 7); // Ignore hole and the missing lower-right pixel are excluded.
  assert.ok(Math.abs(gallbladder.centroid.x - 33 / 70) < 1e-12);
  assert.ok(Math.abs(gallbladder.centroid.y - 19 / 56) < 1e-12);
  assert.deepEqual(duct, { structureId: "cystic_duct", bounds: { x: 0, y: 1 / 2, width: 1 / 5, height: 1 / 2 },
    centroid: { x: 1 / 10, y: 3 / 4 }, pixelCount: 2 });
  for (const region of regions) {
    assert.ok([...Object.values(region.bounds), ...Object.values(region.centroid)].every(value => Number.isFinite(value) && value >= 0 && value <= 1));
    assert.ok(region.bounds.x + region.bounds.width <= 1 && region.bounds.y + region.bounds.height <= 1);
  }
});

test("camera loading bypasses browser pixel decoding for mask IDs and releases image resources", async () => {
  const data = fixture();
  const sample: LoadedDatasetSample = { id: "1_2", labels: data.labels, image: new Blob(["jpeg"]),
    indexMask: new Blob([encodeIndexMaskPng(11, 11, data.indices)]) };
  const previous = ["document", "createImageBitmap"].map(name => [name, Object.getOwnPropertyDescriptor(globalThis, name)] as const);
  let closed = 0, decodes = 0, wrongDimensions = false;
  let canvasReads = 0;
  const written: Uint8ClampedArray[] = [];
  try {
    Object.defineProperty(globalThis, "createImageBitmap", { configurable: true, value: async (blob: Blob) => {
      decodes++;
      // Simulate a browser that changes image color values: mask decoding here would fail validation.
      const pixels = blob === sample.image ? data.original : new Uint8ClampedArray(data.mask.length).fill(42);
      return { width: wrongDimensions ? 12 : 11, height: 11, pixels, close: () => { closed++; } };
    } });
    Object.defineProperty(globalThis, "document", { configurable: true, value: { createElement: () => {
      let source: { pixels: Uint8ClampedArray };
      return { width: 0, height: 0, getContext: () => ({ drawImage: (image: typeof source) => { source = image; },
        getImageData: () => { canvasReads++; return { data: source.pixels }; },
        createImageData: (width: number, height: number) => ({ data: new Uint8ClampedArray(width * height * 4) }),
        putImageData: (image: { data: Uint8ClampedArray }) => { written.push(image.data.slice()); } }) };
    } } });
    const asset = await loadImageOverlayAsset(sample);
    assert.equal(asset.sourceLabel, "Supplied dataset annotation");
    assert.equal(asset.credit.license, "CC BY-NC-SA 4.0");
    assert.equal(asset.width, 11); assert.equal(asset.scene.canvas.width, 11);
    assert.equal(decodes, 1); assert.equal(canvasReads, 1); assert.equal(closed, 1);
    const expected = composeImageOverlayPixels(data.labels, data.original, data.mask);
    assert.deepEqual(asset.regions, expected.regions);
    assert.deepEqual(written, [expected.scene.rgba, expected.anatomy.rgba]);
    asset.dispose(); assert.equal(asset.scene.canvas.width, 1); assert.equal(asset.anatomy.canvas.width, 1);
    wrongDimensions = true;
    await assert.rejects(loadImageOverlayAsset(sample), /dimensions/);
    assert.equal(closed, 2);
    wrongDimensions = false;
    await assert.rejects(loadImageOverlayAsset({ ...sample, indexMask: new Blob(["corrupt mask"]) }), /mask|PNG/i);
    assert.equal(closed, 3); assert.equal(written.length, 2); // A bad mask never makes a texture.
    const controller = new AbortController(); controller.abort();
    await assert.rejects(loadImageOverlayAsset(sample, controller.signal), { name: "AbortError" });
    assert.equal(decodes, 3);
  } finally {
    for (const [name, descriptor] of previous) {
      if (descriptor) Object.defineProperty(globalThis, name, descriptor); else Reflect.deleteProperty(globalThis, name);
    }
  }
});

test("the image-element fallback revokes its object URL when native size validation fails", async () => {
  const data = fixture();
  const sample: LoadedDatasetSample = { id: "1_2", labels: data.labels, image: new Blob(["jpeg"]), indexMask: new Blob(["mask"]) };
  const previous = ["Image", "createImageBitmap"].map(name => [name, Object.getOwnPropertyDescriptor(globalThis, name)] as const);
  const create = URL.createObjectURL, revoke = URL.revokeObjectURL;
  const revoked: string[] = [];
  try {
    Object.defineProperty(globalThis, "createImageBitmap", { configurable: true, value: undefined });
    Object.defineProperty(globalThis, "Image", { configurable: true, value: class {
      naturalWidth = 12; naturalHeight = 11;
      onload: (() => void) | null = null; onerror: (() => void) | null = null;
      set src(value: string) { if (value) queueMicrotask(() => this.onload?.()); }
    } });
    URL.createObjectURL = () => "blob:local-image";
    URL.revokeObjectURL = url => { revoked.push(url); };
    await assert.rejects(loadImageOverlayAsset(sample), /dimensions/);
    assert.deepEqual(revoked, ["blob:local-image"]);
  } finally {
    URL.createObjectURL = create; URL.revokeObjectURL = revoke;
    for (const [name, descriptor] of previous) {
      if (descriptor) Object.defineProperty(globalThis, name, descriptor); else Reflect.deleteProperty(globalThis, name);
    }
  }
});
