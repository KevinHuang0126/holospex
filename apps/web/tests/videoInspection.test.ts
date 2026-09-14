import assert from "node:assert/strict";
import test from "node:test";
import type { FrameResult } from "@holospex/contracts";
import { freezeVideoFrame } from "../src/camera/videoInspection";

const frame = { mediaId: "upload:test", frameNumber: 12, timestampMs: 2400, width: 1280, height: 720 };
const result: FrameResult = { ...frame, schemaVersion: "1.0.0", coordinateSpace: "original_pixels", source: "ml_prediction",
  model: { id: "test", version: "1" }, status: "ok", structures: [] };

test("inspection keeps the visible result and copies its exact pixels, independent of advancing playback", () => {
  const shown = { image: { pixels: "frame 12" }, frame: { ...frame }, capturedAt: 3000 };
  const identified = { ...shown, result };
  const frozen = freezeVideoFrame(shown, identified, image => ({ ...image }));
  shown.image.pixels = "frame 90"; shown.frame.frameNumber = 90; shown.frame.timestampMs = 18000;
  assert.equal(frozen.image.pixels, "frame 12");
  assert.notEqual(frozen.image, shown.image);
  assert.deepEqual(frozen.frame, frame);
  assert.equal(frozen.result, result);
  assert.equal(frozen.capturedAt, 3000);
});

test("inspection of a newer preview never reuses identification from another frame or image", () => {
  const shown = { image: { pixels: "frame 12" }, frame: { ...frame }, capturedAt: 3000 };
  const copy = (image: { pixels: string }) => ({ ...image });
  assert.equal(freezeVideoFrame(shown, null, copy).result, undefined);
  assert.equal(freezeVideoFrame(shown, { ...shown, image: { pixels: "other pixels" }, result }, copy).result, undefined);
  for (const mismatch of [ { mediaId: "upload:other" }, { frameNumber: 13 }, { timestampMs: 2500 }, { width: 640 }, { height: 360 } ]) {
    assert.equal(freezeVideoFrame(shown, { ...shown, result: { ...result, ...mismatch } }, copy).result, undefined);
  }
});
