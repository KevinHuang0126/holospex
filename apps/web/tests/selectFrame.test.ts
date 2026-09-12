import assert from "node:assert/strict";
import test from "node:test";
import type { FrameResult } from "@holospex/contracts";
import { matchesDisplayedFrame } from "../src/overlays/selectFrame";

const displayed = { mediaId: "clip-a", frameNumber: 30, timestampMs: 1_000, width: 800, height: 450 };
const result: FrameResult = {
  schemaVersion: "1.0.0", ...displayed,
  coordinateSpace: "original_pixels", source: "synthetic_mock", status: "ok",
  structures: [],
};

test("a successful empty result is displayable; unavailable results are not", () => {
  assert.equal(matchesDisplayedFrame(result, displayed), true);
  assert.equal(matchesDisplayedFrame(null, displayed), false);
  for (const status of ["missing", "error", "unsupported"] as const) {
    assert.equal(matchesDisplayedFrame({ ...result, status, statusReason: "Unavailable" }, displayed), false);
  }
});

test("seeking or switching clips never displays a previous frame result", () => {
  assert.equal(matchesDisplayedFrame(result, { ...displayed, frameNumber: 31, timestampMs: 1_033 }), false);
  assert.equal(matchesDisplayedFrame(result, { ...displayed, mediaId: "clip-b" }), false);
  assert.equal(matchesDisplayedFrame(result, { ...displayed, timestampMs: 999 }), false);
});

test("results exported at resized dimensions cannot silently misalign source-frame polygons", () => {
  assert.equal(matchesDisplayedFrame({ ...result, width: 400, height: 225 }, displayed), false);
  assert.equal(matchesDisplayedFrame({ ...result, height: 449 }, displayed), false);
});
