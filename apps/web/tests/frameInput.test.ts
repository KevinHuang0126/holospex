import assert from "node:assert/strict";
import test from "node:test";
import fixture from "../../../assets/demo/frame-000.json";
import { readHudFrame } from "../src/overlays/frameInput";

const frame = { mediaId: fixture.mediaId, frameNumber: fixture.frameNumber, timestampMs: fixture.timestampMs, width: fixture.width, height: fixture.height };
test("missing and malformed inputs clear results without throwing", () => {
  for (const input of [null, undefined, {}, { ...fixture, width: 0 }, { ...fixture, structures: [{ polygon: null }] }]) {
    const result = readHudFrame(input, frame, "synthetic_mock");
    assert.equal(result.result, null);
    assert.match(result.message!, /Unable to assess/);
  }
});
test("reviewed/prediction source switches and frame mismatches reject previous results", () => {
  assert.equal(readHudFrame(fixture, frame, "reviewed_annotation").status, "mismatch");
  assert.equal(readHudFrame(fixture, { ...frame, frameNumber: 1 }, "synthetic_mock").status, "mismatch");
  assert.equal(readHudFrame(fixture, null, "synthetic_mock").status, "mismatch");
});
test("unavailability remains distinct from successful output with zero detections", () => {
  for (const status of ["missing", "unsupported", "error"] as const) {
    const result = readHudFrame({ ...fixture, status, structures: [], statusReason: "Test reason" }, frame, "synthetic_mock");
    assert.equal(result.status, status);
    assert.equal(result.result, null);
    assert.match(result.message!, /Test reason/);
  }
  const result = readHudFrame({ ...fixture, structures: [] }, frame, "synthetic_mock");
  assert.equal(result.status, "ready");
  assert.deepEqual(result.result?.structures, []);
});
test("predictions require confidence/model identity and propagation keeps provenance", () => {
  const predicted = { ...fixture, source: "ml_prediction" };
  assert.equal(readHudFrame(predicted, frame, "ml_prediction").status, "invalid");
  const propagated = { ...fixture, timestampMs: 100, source: "propagated_prediction", model: { id: "test", version: "1" }, propagatedFromTimestampMs: 0, structures: fixture.structures.map(item => ({ ...item, confidence: 0.6, visibility: "partial" })) };
  const result = readHudFrame(propagated, { ...frame, timestampMs: 100 }, "propagated_prediction");
  assert.equal(result.status, "ready");
  assert.equal(result.result?.source, "propagated_prediction");
  assert.equal(result.result?.structures[0].visibility, "partial");
});
