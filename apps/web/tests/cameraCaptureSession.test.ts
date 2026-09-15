import assert from "node:assert/strict";
import test from "node:test";
import { createCameraCaptureSession, MAX_CAMERA_PREVIEW_AGE_MS } from "../src/camera/cameraCaptureSession";
import { MAX_LIVE_FRAME_AGE_MS } from "../src/input/liveFramePipeline";
import { IdentificationBusyError } from "../src/input/liveIdentification";
import type { DisplayedFrame } from "../src/overlays/selectFrame";

const flush = () => new Promise<void>(resolve => setImmediate(resolve));
function deferred<T>() {
  let resolve!: (value: T) => void, reject!: (cause: unknown) => void;
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}
const frame = (frameNumber = 0): DisplayedFrame => ({ mediaId: "camera-capture-1", frameNumber, timestampMs: frameNumber * 100, width: 640, height: 480 });
const prediction = (input = frame()) => ({ schemaVersion: "1.0.0", ...input, coordinateSpace: "original_pixels", source: "ml_prediction", status: "ok",
  model: { id: "test-model", version: "test-v1" }, structures: [{ instanceId: "gallbladder-0", structureId: "gallbladder", confidence: 0.9,
    visibility: "visible", polygon: [[10, 10], [50, 10], [50, 50], [10, 50]] }] });
function clock() {
  let time = 100;
  const timers = new Map<object, { due: number; callback: () => void }>();
  return {
    now: () => time,
    scheduleTimeout: (callback: () => void, delayMs: number) => { const id = {}; timers.set(id, { due: time + delayMs, callback }); return id; },
    cancelTimeout: (id: unknown) => { timers.delete(id as object); },
    advance: (duration: number) => { time += duration; for (const [id, timer] of timers) if (timer.due <= time) { timers.delete(id); timer.callback(); } },
    get pending() { return timers.size; },
  };
}
const preview = (capturedAt = 100, number = 0) => ({ image: { pixels: [1, 2, 3] }, frame: frame(number), capturedAt });

test("only explicit capture sends one immutable owned image; successful stills persist until retake", async () => {
  const timer = clock(), output = deferred<unknown>(), encoding = deferred<Blob>();
  let encodes = 0, requests = 0, changes = 0;
  const session = createCameraCaptureSession({ ...timer, copyImage: (image: { pixels: number[] }) => ({ pixels: [...image.pixels] }),
    encode: image => { encodes++; assert.deepEqual(image.pixels, [1, 2, 3]); return encoding.promise; }, onChange: () => { changes++; } });
  assert.equal(session.status, "preview"); assert.equal(session.snapshot, null);
  session.setIdentifier(async (input, bytes) => { requests++; assert.deepEqual(input, frame()); assert.equal(await bytes.text(), "owned"); return output.promise; });
  await flush(); assert.equal(requests, 0); assert.equal(encodes, 0);
  const source = preview();
  assert.equal(session.capture(source), true);
  source.image.pixels.fill(9); source.frame.frameNumber = 88;
  assert.deepEqual(session.snapshot?.image.pixels, [1, 2, 3]); assert.deepEqual(session.snapshot?.frame, frame());
  assert.equal(Object.isFrozen(session.snapshot?.frame), true);
  assert.equal(session.status, "identifying"); assert.equal(session.busy, true);
  assert.equal(session.capture(preview()), false);
  encoding.resolve(new Blob(["owned"])); await flush(); assert.equal(requests, 1);
  timer.advance(2000); output.resolve(prediction()); await flush();
  assert.equal(session.status, "identified"); assert.equal(session.busy, false); assert.ok(session.snapshot?.result);
  const captured = session.snapshot; timer.advance(60000); await flush();
  assert.equal(session.snapshot, captured); assert.equal(session.status, "identified"); assert.equal(requests, 1);
  assert.equal(session.capture(preview(timer.now(), 1)), false); assert.equal(encodes, 1);
  session.retake(); await flush();
  assert.equal(session.snapshot, null); assert.equal(session.status, "preview"); assert.equal(requests, 1); assert.ok(changes >= 4);
  session.dispose(); assert.equal(timer.pending, 0);
});

test("retake and model changes suppress aborted late results and retain uncooperative provider slots", async () => {
  const timer = clock(), old = deferred<unknown>(); let signal: AbortSignal | undefined, firstCalls = 0, nextCalls = 0, changes = 0;
  const session = createCameraCaptureSession({ ...timer, copyImage: (image: { pixels: number[] }) => structuredClone(image),
    encode: async () => new Blob(), onChange: () => { changes++; } });
  session.setIdentifier(async (_, __, value) => { firstCalls++; signal = value; return old.promise; });
  session.capture(preview()); await flush();
  session.retake(); assert.equal(signal?.aborted, true); assert.equal(session.snapshot, null); assert.equal(session.busy, true);
  const next = async (input: DisplayedFrame) => { nextCalls++; return prediction(input); };
  session.setIdentifier(next); await flush(); assert.equal(nextCalls, 0);
  assert.equal(session.capture(preview()), false); assert.equal(session.snapshot, null);
  const beforeSettlement = changes;
  old.resolve(prediction()); await flush();
  assert.equal(session.busy, false); assert.equal(session.snapshot, null); assert.ok(changes > beforeSettlement, "UI learns that the retired slot settled");
  assert.equal(nextCalls, 0); assert.equal(firstCalls, 1);
  assert.equal(session.capture(preview(timer.now(), 1)), true); await flush();
  assert.equal(nextCalls, 1); assert.equal(session.snapshot?.result?.frameNumber, 1);
  session.setIdentifier(next); assert.equal(session.status, "identified", "Same identifier does not discard a still");
  session.setIdentifier(undefined); assert.equal(session.snapshot, null); assert.equal(session.status, "preview");
  session.dispose();
});

test("expired encoding does not send any frame, and timeout never starts an automatic retry", async () => {
  const timer = clock(), encoding = deferred<Blob>(); let requests = 0;
  const session = createCameraCaptureSession({ ...timer, copyImage: (image: { pixels: number[] }) => structuredClone(image),
    encode: () => encoding.promise, onChange() {} });
  session.setIdentifier(async input => { requests++; return prediction(input); });
  session.capture(preview()); timer.advance(MAX_LIVE_FRAME_AGE_MS);
  assert.equal(session.status, "error"); assert.match(session.message!, /timed out/); assert.equal(session.busy, true);
  assert.ok(session.snapshot); assert.equal(session.snapshot?.result, undefined);
  session.retake(); assert.equal(session.busy, true); assert.equal(session.capture(preview(timer.now())), false);
  encoding.resolve(new Blob()); await flush();
  assert.equal(requests, 0); assert.equal(session.busy, false); assert.equal(session.snapshot, null);
  timer.advance(10000); await flush(); assert.equal(requests, 0); session.dispose();
});

test("late responses after a transport timeout cannot annotate the captured still", async () => {
  const timer = clock(), output = deferred<unknown>(); let requests = 0;
  const session = createCameraCaptureSession({ ...timer, copyImage: (image: { pixels: number[] }) => structuredClone(image),
    encode: async () => new Blob(), onChange() {} });
  session.setIdentifier(async () => { requests++; return output.promise; });
  session.capture(preview()); await flush(); timer.advance(MAX_LIVE_FRAME_AGE_MS);
  output.resolve(prediction()); await flush();
  assert.equal(session.status, "error"); assert.match(session.message!, /timed out/); assert.equal(session.snapshot?.result, undefined);
  assert.equal(session.busy, false); assert.equal(requests, 1); session.dispose();
});

test("stale, future and non-finite previews or copy errors never encode, and exception contents stay private", async () => {
  const timer = clock(); let encodes = 0, copies = 0;
  const session = createCameraCaptureSession({ ...timer, copyImage: (_image: { pixels: number[] }) => { copies++; throw new Error("secret file path"); },
    encode: async () => { encodes++; return new Blob(); }, onChange() {} });
  assert.equal(session.capture(preview()), false); assert.match(session.message!, /not ready/);
  session.setIdentifier(async input => prediction(input));
  for (const capturedAt of [timer.now() - MAX_CAMERA_PREVIEW_AGE_MS, timer.now() + 1, NaN, -Infinity]) {
    assert.equal(session.capture(preview(capturedAt)), false); assert.match(session.message!, /outdated/);
  }
  assert.equal(copies, 0); assert.equal(session.capture(preview()), false);
  assert.match(session.message!, /could not be captured/); assert.doesNotMatch(session.message!, /secret/);
  assert.equal(session.snapshot, null); assert.equal(session.busy, false); assert.equal(encodes, 0); session.dispose();
});

test("busy, invalid and encoding failures remain explicit errors without implicit requests", async () => {
  for (const failure of ["busy", "invalid", "encoding"] as const) {
    const timer = clock(); let requests = 0;
    const session = createCameraCaptureSession({ ...timer, copyImage: (image: { pixels: number[] }) => structuredClone(image),
      encode: async () => { if (failure === "encoding") throw new Error("private encoder detail"); return new Blob(); }, onChange() {} });
    session.setIdentifier(async () => { requests++; if (failure === "busy") throw new IdentificationBusyError(); return prediction(frame(99)); });
    session.capture(preview()); await flush();
    assert.equal(session.status, "error"); assert.equal(session.snapshot?.result, undefined);
    assert.match(session.message!, failure === "busy" ? /busy/ : failure === "invalid" ? /invalid or mismatched/ : /could not be prepared/);
    assert.doesNotMatch(session.message!, /private/); assert.equal(session.busy, failure === "busy");
    const expected = requests; timer.advance(60000); await flush(); assert.equal(requests, expected);
    session.retake(); await flush(); assert.equal(requests, expected); session.dispose();
  }
});

test("dispose aborts work and prevents late change notifications or future capture", async () => {
  const timer = clock(), output = deferred<unknown>(); let changes = 0, signal: AbortSignal | undefined;
  const session = createCameraCaptureSession({ ...timer, copyImage: (image: { pixels: number[] }) => structuredClone(image),
    encode: async () => new Blob(), onChange: () => { changes++; } });
  session.setIdentifier(async (_, __, value) => { signal = value; return output.promise; });
  session.capture(preview()); await flush(); session.dispose(); session.dispose();
  assert.equal(signal?.aborted, true); assert.equal(session.snapshot, null); assert.equal(timer.pending, 0);
  const prior = changes; output.resolve(prediction()); await flush(); timer.advance(60000);
  assert.equal(changes, prior); assert.equal(session.capture(preview(timer.now())), false);
});
