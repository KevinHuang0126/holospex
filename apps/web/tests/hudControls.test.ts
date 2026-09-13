import assert from "node:assert/strict";
import test from "node:test";
import { hudReducer, initialHudState, hudOverlaysVisible, isCurrentSelection, seekVideo, type VideoAnatomySelection } from "../src/overlays/hudControls";

const frame = { mediaId: "clip-a", frameNumber: 31, timestampMs: 1033.33, width: 800, height: 450 };
const active = hudReducer(initialHudState, { type: "media", mediaId: frame.mediaId });
const displayed = hudReducer(active, { type: "frame", frame, revision: active.revision });

test("seek, source, media, mode and input changes invalidate old frame callbacks", () => {
  for (const action of [
    { type: "invalidate" }, { type: "source", source: "ml_prediction" },
    { type: "media", mediaId: "clip-b" }, { type: "mode", mode: "assess" },
    { type: "experience", experience: "model" },
  ] as const) {
    const next = hudReducer(displayed, action);
    assert.equal(next.displayedFrame, null);
    assert.equal(hudReducer(next, { type: "frame", frame, revision: active.revision }).displayedFrame, null);
  }
});

test("frame reports preserve original fractional timestamps and reject invalid identity", () => {
  assert.deepEqual(displayed.displayedFrame, frame);
  for (const invalid of [{ ...frame, mediaId: "wrong" }, { ...frame, width: 0 }, { ...frame, frameNumber: 1.5 }, { ...frame, timestampMs: NaN }]) {
    assert.equal(hudReducer(active, { type: "frame", frame: invalid, revision: active.revision }).displayedFrame, null);
  }
});

test("Identify and Assess hide answers regardless of show requests, and ML cannot reveal feedback", () => {
  for (const mode of ["identify", "assess"] as const) {
    assert.equal(hudOverlaysVisible({ ...initialHudState, mode, overlaysRequested: true }), false);
  }
  assert.equal(hudOverlaysVisible({ ...initialHudState, mode: "feedback", source: "ml_prediction" }), false);
  assert.equal(hudOverlaysVisible({ ...initialHudState, mode: "feedback", source: "reviewed_annotation" }), true);
  assert.equal(hudOverlaysVisible({ ...initialHudState, overlaysRequested: false }), false);
  assert.equal(hudOverlaysVisible({ ...initialHudState, mode: "feedback", experience: "model" }), true);
});

test("selection callbacks retain frame/source and reject clicks from an earlier display", () => {
  const selection: VideoAnatomySelection = { experience: "video", frame, source: displayed.source, revision: displayed.revision, structureId: "gallbladder", instanceId: "one", point: [200, 100] };
  assert.equal(isCurrentSelection(displayed, selection), true);
  assert.equal(isCurrentSelection(displayed, { ...selection, point: [-1, 10] }), false);
  assert.equal(isCurrentSelection(displayed, { ...selection, source: "synthetic_mock" }), false);
  assert.equal(isCurrentSelection(hudReducer(displayed, { type: "invalidate" }), selection), false);
});

test("seek converts milliseconds, bounds to duration, and refuses unloaded or invalid requests", () => {
  const video = { duration: 10, currentTime: 0, readyState: 2 };
  seekVideo(video, 2500);
  assert.equal(video.currentTime, 2.5);
  seekVideo(video, 15000);
  assert.equal(video.currentTime, 10);
  for (const time of [-1, NaN, Infinity]) assert.throws(() => seekVideo(video, time));
  assert.throws(() => seekVideo({ ...video, readyState: 0 }, 1));
  assert.throws(() => seekVideo({ ...video, duration: Infinity }, 1));
});
