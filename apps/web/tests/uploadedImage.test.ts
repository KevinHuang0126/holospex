import assert from "node:assert/strict";
import test, { type TestContext } from "node:test";
import { decodeUploadedImage, MAX_UPLOADED_IMAGE_BYTES, uploadedImageSize } from "../src/camera/uploadedImage";

const jpeg = () => new File([new Uint8Array([255, 216, 255, 224])], "oriented.jpg", { type: "image/jpeg" });
const flush = () => new Promise<void>(resolve => setImmediate(resolve));
function globalValue(t: TestContext, name: string, value: unknown) {
  const original = Object.getOwnPropertyDescriptor(globalThis, name);
  Object.defineProperty(globalThis, name, { configurable: true, writable: true, value });
  t.after(() => { if (original) Object.defineProperty(globalThis, name, original); else Reflect.deleteProperty(globalThis, name); });
}
function canvasEnvironment(t: TestContext) {
  const operations: unknown[][] = [];
  const context = { fillStyle: "", imageSmoothingEnabled: false, imageSmoothingQuality: "low",
    fillRect: (...args: number[]) => operations.push(["fill", ...args]),
    drawImage: (...args: unknown[]) => operations.push(["draw", ...args]) };
  const canvas = { width: 0, height: 0, getContext: () => context };
  globalValue(t, "document", { createElement: (tag: string) => { assert.equal(tag, "canvas"); return canvas; } });
  return { canvas, context, operations };
}

test("image fitting preserves portrait, landscape and small images and rejects oversized dimensions", () => {
  assert.deepEqual(uploadedImageSize(4000, 3000), { width: 960, height: 720 });
  assert.deepEqual(uploadedImageSize(3000, 4000), { width: 540, height: 720 });
  assert.deepEqual(uploadedImageSize(854, 480), { width: 854, height: 480 });
  for (const [width, height] of [[0, 1], [1, NaN], [1.5, 10], [16385, 1], [8000, 6000]])
    assert.throws(() => uploadedImageSize(width, height), /40 megapixels/);
});

test("decoding requests EXIF orientation, fits oriented pixels once and releases the bitmap", async t => {
  const { canvas, context, operations } = canvasEnvironment(t);
  let closes = 0;
  const bitmap = { width: 3000, height: 4000, close: () => closes++ };
  globalValue(t, "createImageBitmap", async (blob: Blob, options: unknown) => {
    assert.equal(blob.type, "image/jpeg"); assert.deepEqual(options, { imageOrientation: "from-image" }); return bitmap;
  });
  const result = await decodeUploadedImage(jpeg(), new AbortController().signal);
  assert.deepEqual(result, { image: canvas, width: 540, height: 720 });
  assert.deepEqual(operations, [["fill", 0, 0, 540, 720], ["draw", bitmap, 0, 0, 540, 720]]);
  assert.equal(context.fillStyle, "#000"); assert.equal(context.imageSmoothingQuality, "high"); assert.equal(closes, 1);
});

test("signatures gate supported formats and reject empty, oversized, disguised and HEIC files before decoding", async t => {
  canvasEnvironment(t); let decodes = 0;
  globalValue(t, "createImageBitmap", async () => { decodes++; return { width: 100, height: 80, close() {} }; });
  const signal = new AbortController().signal;
  for (const [file, message] of [
    [new File([], "empty.jpg"), /empty/], [new File(["heic"], "photo.HEIC"), /Convert.*JPEG or PNG/],
    [new File(["heif"], "photo.bin", { type: "image/heif" }), /Convert.*JPEG or PNG/],
    [new File(["<svg />"], "pretend.png", { type: "image/png" }), /Unsupported/],
    [new File([new Uint8Array(MAX_UPLOADED_IMAGE_BYTES + 1)], "large.jpg"), /20 MiB/],
  ] as const) await assert.rejects(decodeUploadedImage(file, signal), message);
  assert.equal(decodes, 0);
  await decodeUploadedImage(new File([new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10])], "sample.png"), signal);
  await decodeUploadedImage(new File(["RIFF0000WEBP"], "sample.webp"), signal);
  assert.equal(decodes, 2);
});

test("cancelled decoding closes late bitmaps and never paints obsolete pixels", async t => {
  const { operations } = canvasEnvironment(t);
  let resolve!: (value: unknown) => void, closes = 0;
  globalValue(t, "createImageBitmap", () => new Promise(value => { resolve = value; }));
  const controller = new AbortController();
  const task = decodeUploadedImage(jpeg(), controller.signal); await flush(); controller.abort();
  await assert.rejects(task, { name: "AbortError" });
  resolve({ width: 100, height: 80, close: () => closes++ }); await flush();
  assert.equal(closes, 1); assert.deepEqual(operations, []);
});

test("browser image fallback cleans its object URL on success and sanitizes decoder failures", async t => {
  const { operations } = canvasEnvironment(t);
  let revokes = 0, removals = 0;
  globalValue(t, "createImageBitmap", undefined);
  globalValue(t, "Image", class {
    src = ""; naturalWidth = 640; naturalHeight = 480;
    decode = async () => {};
    removeAttribute(name: string) { assert.equal(name, "src"); removals++; }
  });
  t.mock.method(URL, "createObjectURL", () => "blob:local-image");
  t.mock.method(URL, "revokeObjectURL", (value: string) => { assert.equal(value, "blob:local-image"); revokes++; });
  const result = await decodeUploadedImage(jpeg(), new AbortController().signal);
  assert.equal(result.width, 640); assert.equal(result.height, 480); assert.equal(operations.length, 2);
  assert.equal(revokes, 1); assert.equal(removals, 1);
  globalValue(t, "createImageBitmap", async () => { throw new Error("private decoder internals"); });
  await assert.rejects(decodeUploadedImage(jpeg(), new AbortController().signal), cause => {
    assert.match((cause as Error).message, /could not be decoded/); assert.doesNotMatch((cause as Error).message, /private/); return true;
  });
});
