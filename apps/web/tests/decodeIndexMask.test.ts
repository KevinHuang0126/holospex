import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { deflateSync } from "node:zlib";
import { decodeIndexMask } from "../src/overlays/decodeIndexMask";
import { buildDatasetRaster } from "../src/overlays/datasetRaster";
import { parseDatasetSample } from "../src/overlays/datasetSamples";
import { encodeIndexMaskPng, pngChunk } from "./helpers/indexMaskPng";

const signature = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
function header(width: number, height: number) {
  const bytes = Buffer.alloc(13); bytes.writeUInt32BE(width); bytes.writeUInt32BE(height, 4); bytes[8] = 8;
  return bytes;
}
function png(width: number, height: number, rows: number[], chunks?: Buffer[]) {
  return Buffer.concat([signature, pngChunk("IHDR", header(width, height)),
    ...(chunks ?? [pngChunk("IDAT", deflateSync(Buffer.from(rows)))]), pngChunk("IEND")]);
}
const decode = (bytes: Uint8Array, width: number, height: number, signal?: AbortSignal) => decodeIndexMask(new Blob([Uint8Array.from(bytes)]), width, height, signal);

test("raw categorical bytes survive all five PNG row filters and modulo-256 reconstruction", async () => {
  // Hand-calculated encoded rows exercise previous-row, left-edge and Paeth dependencies.
  const rows = [0, 0, 1, 6, 255, 1, 255, 7, 251, 255, 2, 2, 252, 2, 4, 3, 5, 3, 3, 3, 4, 4, 247, 255, 2];
  const values = [0, 1, 6, 255, 255, 6, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 255, 1];
  assert.deepEqual(await decode(png(4, 5, rows), 4, 5), new Uint8ClampedArray(values.flatMap(value => [value, value, value, 255])));
  const firstRows = [[0, 0, 1, 6, 255], [1, 0, 1, 5, 249], [2, 0, 1, 6, 255], [3, 0, 1, 6, 252], [4, 0, 1, 5, 249]];
  for (const row of firstRows) assert.deepEqual(await decode(png(4, 1, row), 4, 1), new Uint8ClampedArray([0, 1, 6, 255].flatMap(value => [value, value, value, 255])));
});

test("IDAT chunks form one zlib stream and display color metadata cannot alter label indices", async () => {
  const compressed = deflateSync(Buffer.from([0, 0, 1, 2, 3, 4, 5, 6, 255]));
  const gamma = Buffer.alloc(4); gamma.writeUInt32BE(100000);
  const chunks = [pngChunk("gAMA", gamma), pngChunk("sRGB", Buffer.from([0])),
    ...Array.from(compressed, value => pngChunk("IDAT", Buffer.from([value]))), pngChunk("tEXt", Buffer.from("note\0categorical values"))];
  const rgba = await decode(png(8, 1, [], chunks), 8, 1);
  assert.deepEqual(Array.from(rgba.filter((_, index) => index % 4 === 0)), [0, 1, 2, 3, 4, 5, 6, 255]);
});

test("corrupt or unsupported PNG structure fails before categorical pixels can be returned", async () => {
  const valid = encodeIndexMaskPng(2, 1, [1, 255]);
  const badSignature = Buffer.from(valid); badSignature[0] = 0;
  const badCrc = Buffer.from(valid); badCrc[29] ^= 1;
  const compressed = deflateSync(Buffer.from([0, 1, 255]));
  const malformed = [badSignature, badCrc, valid.subarray(0, valid.length - 1), Buffer.concat([valid, Buffer.from([0])]),
    Buffer.concat([signature, pngChunk("tEXt"), valid.subarray(8)]),
    png(2, 1, [], [pngChunk("IHDR", header(2, 1)), pngChunk("IDAT", compressed)]),
    png(2, 1, [], []),
    png(2, 1, [], [pngChunk("IDAT", compressed.subarray(0, 2)), pngChunk("tEXt"), pngChunk("IDAT", compressed.subarray(2))]),
    ...["PLTE", "ABcD", "tRNS", "acTL", "fcTL", "fdAT"].map(type => png(2, 1, [], [pngChunk(type), pngChunk("IDAT", compressed)])),
    Buffer.concat([valid.subarray(0, valid.length - 12), pngChunk("IEND", Buffer.from([0]))]),
  ];
  for (const bytes of malformed) await assert.rejects(decode(bytes, 2, 1), /Invalid index PNG/);
  await assert.rejects(decode(valid, 3, 1), /dimensions disagree/);
  for (const [offset, value] of [[8, 16], [9, 2], [10, 1], [11, 1], [12, 1]]) {
    const altered = header(2, 1); altered[offset] = value;
    const bytes = Buffer.concat([signature, pngChunk("IHDR", altered), pngChunk("IDAT", compressed), pngChunk("IEND")]);
    await assert.rejects(decode(bytes, 2, 1), /only non-interlaced, 8-bit grayscale/);
  }
});

test("inflate length, filter, file size and image area limits reject malformed or oversized data", async () => {
  for (const rows of [[0, 1], [0, 1, 2, 3], [5, 1, 2]]) await assert.rejects(decode(png(2, 1, rows), 2, 1), /inflated data|scanline filter/);
  const bomb = png(2, 1, [], [pngChunk("IDAT", deflateSync(Buffer.alloc(1024 * 1024)))]);
  await assert.rejects(decode(bomb, 2, 1), /exceeds the declared mask size/);
  await assert.rejects(decode(png(2, 1, [], [pngChunk("IDAT", Buffer.from([0, 1, 2, 3]))]), 2, 1), /compressed image data is corrupt/);
  const valid = encodeIndexMaskPng(1, 1, [1]);
  for (const [width, height] of [[0, 1], [1.5, 1], [NaN, 1], [8193, 1], [4001, 4000]]) await assert.rejects(decode(valid, width, height), /dimensions exceed/);
  await assert.rejects(decodeIndexMask({ size: 32 * 1024 * 1024 + 1, arrayBuffer: () => { throw new Error("Must not read an oversized file."); } } as unknown as Blob, 1, 1), /smaller than 32 MB/);
});

test("cancellation before and during asynchronous reads returns AbortError", async () => {
  const bytes = encodeIndexMaskPng(2, 1, [1, 255]);
  const controller = new AbortController(); controller.abort();
  await assert.rejects(decode(bytes, 2, 1, controller.signal), { name: "AbortError" });
  const reading = new AbortController();
  await assert.rejects(decodeIndexMask({ size: bytes.length, arrayBuffer: async () => {
    reading.abort(); return Uint8Array.from(bytes).buffer;
  } } as unknown as Blob, 2, 1, reading.signal), { name: "AbortError" });
  const inflating = new AbortController();
  const promise = decode(encodeIndexMaskPng(1000, 1000, new Uint8Array(1_000_000)), 1000, 1000, inflating.signal);
  setTimeout(() => inflating.abort(), 0);
  await assert.rejects(promise, { name: "AbortError" });
});

const localSample = new URL("../../../runs/camera-samples/case-116_35325.holospex.json", import.meta.url);
const pillowReference = new URL("../../../runs/camera-mask.rgba", import.meta.url);
test("the prepared phone sample decodes to exact supplied counts and Pillow reference bytes", { skip: !existsSync(localSample) }, async () => {
  const envelope = JSON.parse(await readFile(localSample, "utf8"));
  const sample = parseDatasetSample(envelope.labels);
  const rgba = await decode(Buffer.from(envelope.indexMaskBase64, "base64"), sample.frame.width, sample.frame.height);
  const result = buildDatasetRaster(sample, rgba); // Validates every category and all supplied pixel counts.
  assert.ok(result.anchors.length > 0);
  assert.ok(result.pixels.every(value => value <= 6 || value === 255));
  if (existsSync(pillowReference)) assert.deepEqual(Buffer.from(rgba), await readFile(pillowReference));
});
