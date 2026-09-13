import assert from "node:assert/strict";
import test from "node:test";
import fixture from "../../../assets/demo/frame-000.json";
import { parseResultSet, findVideoResult, visibleStructures } from "../src/overlays/videoResults";
import { containedRect } from "../src/overlays/drawHud";

const [frame] = parseResultSet(fixture);
test("sparse video results match only their frame, dimensions, media and source", () => {
  assert.equal(findVideoResult([frame], frame.mediaId, frame.source, 0.1, 800, 450), frame);
  for (const time of [33.33, 1000, Infinity, NaN]) assert.equal(findVideoResult([frame], frame.mediaId, frame.source, time, 800, 450), null);
  assert.equal(findVideoResult([frame], "wrong-media", frame.source, 0, 800, 450), null);
  assert.equal(findVideoResult([frame], frame.mediaId, "ml_prediction", 0, 800, 450), null);
  assert.equal(findVideoResult([frame], frame.mediaId, frame.source, 0, 400, 225), null);
  assert.equal(findVideoResult([frame, { ...frame, timestampMs: 0.4 }], frame.mediaId, frame.source, 0.2, 800, 450), null);
});
test("a result collection refuses mixed media, duplicate timestamps and contradictory frame IDs", () => {
  assert.throws(() => parseResultSet([fixture, fixture]));
  assert.throws(() => parseResultSet([fixture, { ...fixture, mediaId: "another" }]));
  assert.throws(() => parseResultSet([fixture, { ...fixture, source: "reviewed_annotation", frameNumber: 5 }]));
  assert.throws(() => parseResultSet([fixture, { ...fixture, source: "reviewed_annotation", timestampMs: 40 }]));
});
test("unavailable/low-confidence anatomy clears, partial views retain explicit status, quizzes reveal nothing", () => {
  const [prediction] = parseResultSet({ ...fixture, source: "ml_prediction", model: { id: "test", version: "1" }, structures: fixture.structures.map(item => ({ ...item, confidence: 0.4, visibility: "partial" })) });
  assert.equal(visibleStructures(prediction, true, "learn").structures.length, 0);
  assert.equal(visibleStructures(prediction, true, "learn", 0.5).structures.length, 0);
  assert.match(visibleStructures(prediction, true, "learn", 0.3).warning!, /Partial/);
  assert.equal(visibleStructures(prediction, true, "feedback", 0.3).structures.length, 0);
  for (const mode of ["identify", "assess"] as const) assert.deepEqual(visibleStructures(prediction, true, mode, 0.3), { structures: [], warning: null });
  assert.deepEqual(visibleStructures({ ...prediction, structures: [] }, true, "learn"), { structures: [], warning: null });
  assert.match(visibleStructures(null, true, "learn").warning!, /Unable to assess/);
  assert.equal(visibleStructures(frame, false, "learn").structures.length, 0);
});
test("letterbox coordinate mapping preserves aspect and original corners at multiple display sizes", () => {
  const portrait = containedRect(800, 450, 400, 600);
  assert.deepEqual(portrait, { x: 0, y: 187.5, width: 400, height: 225, scale: 0.5 });
  const wide = containedRect(800, 450, 1200, 450);
  assert.deepEqual(wide, { x: 200, y: 0, width: 800, height: 450, scale: 1 });
  assert.equal(portrait.y + 450 * portrait.scale, 412.5);
});
