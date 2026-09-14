import assert from "node:assert/strict";
import test from "node:test";
import { createLiveFramePipeline, MAX_LIVE_FRAME_AGE_MS } from "../src/input/liveFramePipeline";
import { IdentificationBusyError } from "../src/input/liveIdentification";
import type { DisplayedFrame } from "../src/overlays/selectFrame";
import type { FrameResult } from "@holospex/contracts";

function deferred<T>() {
  let resolve!: (value: T) => void, reject!: (cause: unknown) => void;
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}
const flush = () => new Promise<void>(resolve => setImmediate(resolve));
const frame = (frameNumber = 0): DisplayedFrame => ({ mediaId: "live-camera-1", frameNumber, timestampMs: frameNumber * 100,
  width: 640, height: 480 });
const prediction = (input = frame()) => ({ schemaVersion: "1.0.0", ...input, coordinateSpace: "original_pixels",
  source: "ml_prediction", status: "ok", model: { id: "test-model", version: "test-v1" },
  structures: [{ instanceId: "gallbladder-0", structureId: "gallbladder", confidence: 0.9, visibility: "visible",
    polygon: [[10, 10], [50, 10], [50, 50], [10, 50]] }] });
function clock() {
  let time = 100;
  const timers = new Map<object, { due: number; callback: () => void }>();
  return {
    now: () => time,
    scheduleTimeout: (callback: () => void, delayMs: number) => { const id = {}; timers.set(id, { due: time + delayMs, callback }); return id; },
    cancelTimeout: (id: unknown) => { timers.delete(id as object); },
    advance: (milliseconds: number, runTimers = true) => {
      time += milliseconds;
      if (runTimers) for (const [id, timer] of timers) if (timer.due <= time) { timers.delete(id); timer.callback(); }
    },
    get pending() { return timers.size; },
  };
}

test("live pipeline drops overlap during encoding and identification, then publishes the exact owned frame", async () => {
  const timer = clock(), encoding = deferred<Blob>(), analyzing = deferred<unknown>();
  const results: { frame: DisplayedFrame; result: FrameResult; capturedAt: number }[] = [], unavailable: string[] = [];
  let analyses = 0, droppedEncodes = 0;
  const pipeline = createLiveFramePipeline({ ...timer, identify: async () => { analyses++; return analyzing.promise; },
    onResult: (frame, result, capturedAt) => results.push({ frame, result, capturedAt }), onUnavailable: message => unavailable.push(message) });
  const source = frame();
  assert.equal(pipeline.submit(source, () => encoding.promise, 100), true);
  source.frameNumber = 99; // Consumer mutation cannot change the submitted identity.
  assert.equal(pipeline.busy, true);
  assert.equal(pipeline.submit(frame(1), async () => { droppedEncodes++; return new Blob(); }, 100), false);
  encoding.resolve(new Blob(["jpeg"])); await flush();
  assert.equal(analyses, 1);
  assert.equal(pipeline.submit(frame(2), async () => { droppedEncodes++; return new Blob(); }, 100), false);
  analyzing.resolve(prediction()); await flush();
  assert.equal(droppedEncodes, 0); assert.equal(pipeline.busy, false); assert.equal(timer.pending, 0);
  assert.equal(results.length, 1); assert.deepEqual(results[0].frame, frame()); assert.equal(results[0].capturedAt, 100);
  assert.equal(results[0].result.frameNumber, 0); assert.deepEqual(unavailable, []);
  pipeline.dispose();
});

test("two-second CPU identification keeps the captured image and result frame paired", async () => {
  const timer = clock(), analyzing = deferred<unknown>(), image = new Blob(["captured jpeg"]);
  const capturedFrame = frame(), capturedAt = timer.now();
  const results: { frame: DisplayedFrame; result: FrameResult; capturedAt: number }[] = [], unavailable: string[] = [];
  let signal: AbortSignal | undefined;
  const pipeline = createLiveFramePipeline({ ...timer, identify: async (input, blob, nextSignal) => {
    assert.deepEqual(input, capturedFrame); assert.equal(blob, image); signal = nextSignal; return analyzing.promise;
  }, onResult: (frame, result, capturedAt) => results.push({ frame, result, capturedAt }),
  onUnavailable: message => unavailable.push(message) });
  pipeline.submit(capturedFrame, async () => image, capturedAt); await flush();
  timer.advance(2000);
  assert.equal(signal?.aborted, false); assert.equal(pipeline.busy, true); assert.deepEqual(unavailable, []);
  analyzing.resolve(prediction(capturedFrame)); await flush();
  assert.equal(results.length, 1); assert.deepEqual(results[0].frame, capturedFrame);
  assert.equal(results[0].capturedAt, capturedAt); assert.equal(results[0].result.frameNumber, capturedFrame.frameNumber);
  assert.equal(pipeline.busy, false); assert.equal(timer.pending, 0); assert.deepEqual(unavailable, []);
  pipeline.dispose();
});

test("timeout includes capture and encoding age and keeps the slot until ignored-abort identification settles", async () => {
  const timer = clock(), encoding = deferred<Blob>(), analyzing = deferred<unknown>();
  const unavailable: string[] = [], results: unknown[] = [];
  let signal: AbortSignal | undefined, calls = 0;
  const pipeline = createLiveFramePipeline({ ...timer, identify: async (_, __, nextSignal) => { calls++; signal = nextSignal; return analyzing.promise; },
    onResult: value => results.push(value), onUnavailable: message => unavailable.push(message) });
  timer.advance(200);
  assert.equal(pipeline.submit(frame(), () => encoding.promise, 100), true);
  timer.advance(300); encoding.resolve(new Blob()); await flush();
  assert.equal(calls, 1); assert.equal(signal?.aborted, false);
  timer.advance(MAX_LIVE_FRAME_AGE_MS - 501);
  assert.equal(signal?.aborted, false); assert.deepEqual(unavailable, []);
  timer.advance(1);
  assert.equal(signal?.aborted, true); assert.equal(pipeline.busy, true);
  assert.deepEqual(unavailable, ["Unable to identify. Identification timed out."]);
  assert.equal(pipeline.submit(frame(1), async () => new Blob(), timer.now()), false);
  analyzing.resolve(prediction()); await flush();
  assert.equal(pipeline.busy, false); assert.deepEqual(results, []); assert.equal(unavailable.length, 1);
  assert.equal(pipeline.submit(frame(), async () => new Blob(), timer.now()), true);
  await flush(); assert.equal(calls, 2); assert.equal(results.length, 1);
  pipeline.dispose();
});

test("expired encoding never starts identification, and late results fail even when timer delivery is delayed", async () => {
  const timer = clock(), encoding = deferred<Blob>(), analyzing = deferred<unknown>();
  const unavailable: string[] = []; let calls = 0, results = 0;
  const pipeline = createLiveFramePipeline({ ...timer, identify: async () => { calls++; return analyzing.promise; },
    onResult: () => { results++; }, onUnavailable: message => unavailable.push(message) });
  pipeline.submit(frame(), () => encoding.promise, timer.now());
  timer.advance(MAX_LIVE_FRAME_AGE_MS);
  assert.equal(pipeline.busy, true);
  encoding.resolve(new Blob()); await flush();
  assert.equal(calls, 0); assert.equal(pipeline.busy, false);
  pipeline.submit(frame(), async () => new Blob(), timer.now()); await flush();
  timer.advance(MAX_LIVE_FRAME_AGE_MS, false);
  analyzing.resolve(prediction()); await flush();
  assert.equal(results, 0); assert.equal(unavailable.length, 2); assert.equal(timer.pending, 0);
  pipeline.dispose();
});

test("disposal aborts old work and suppresses every late callback", async () => {
  const timer = clock(), pending = deferred<unknown>(); let signal: AbortSignal | undefined, callbacks = 0;
  const pipeline = createLiveFramePipeline({ ...timer, identify: async (_, __, nextSignal) => { signal = nextSignal; return pending.promise; },
    onResult: () => { callbacks++; }, onUnavailable: () => { callbacks++; } });
  pipeline.submit(frame(), async () => new Blob(), 100); await flush();
  pipeline.dispose(); pipeline.dispose();
  assert.equal(signal?.aborted, true); assert.equal(timer.pending, 0);
  pending.resolve(prediction()); await flush(); timer.advance(MAX_LIVE_FRAME_AGE_MS + 1);
  assert.equal(callbacks, 0); assert.equal(pipeline.submit(frame(), async () => new Blob(), timer.now()), false);
});

test("busy identification backs off without encoding or queuing stale frames, and disposal releases cooldown", async () => {
  const timer = clock(), unavailable: string[] = [], results: DisplayedFrame[] = [];
  let calls = 0, encodes = 0;
  const pipeline = createLiveFramePipeline({ ...timer, identify: async input => {
    calls++;
    if (calls !== 2) throw new IdentificationBusyError();
    return prediction(input);
  }, onResult: value => results.push(value), onUnavailable: message => unavailable.push(message) });
  const encode = async () => { encodes++; return new Blob(); };
  assert.equal(pipeline.submit(frame(), encode, timer.now()), true); await flush();
  assert.deepEqual(unavailable, ["Identification is busy."]);
  assert.equal(pipeline.busy, true); assert.equal(timer.pending, 0);
  timer.advance(1999);
  assert.equal(pipeline.submit(frame(1), encode, timer.now()), false);
  assert.equal(encodes, 1); assert.equal(calls, 1); assert.deepEqual(results, []);
  timer.advance(1);
  assert.equal(pipeline.busy, false);
  assert.equal(pipeline.submit(frame(2), encode, timer.now()), true); await flush();
  assert.deepEqual(results, [frame(2)]); assert.equal(encodes, 2);
  assert.equal(pipeline.submit(frame(3), encode, timer.now()), true); await flush();
  assert.equal(pipeline.busy, true);
  pipeline.dispose();
  assert.equal(pipeline.busy, false); assert.equal(timer.pending, 0);
  timer.advance(10000);
  assert.equal(calls, 3); assert.equal(pipeline.submit(frame(4), encode, timer.now()), false);
});

test("invalid results, encoding errors and identification failures are withheld while releasing the slot", async () => {
  const timer = clock(), unavailable: string[] = []; let results = 0, calls = 0;
  let output: unknown = prediction(frame(1));
  const pipeline = createLiveFramePipeline({ ...timer, identify: async () => { calls++; if (output instanceof Error) throw output; return output; },
    onResult: () => { results++; }, onUnavailable: message => unavailable.push(message) });
  for (const invalid of [prediction(frame(1)), { ...prediction(), width: 320 }, { ...prediction(), source: "synthetic_mock" }, null]) {
    output = invalid; assert.equal(pipeline.submit(frame(), async () => new Blob(), 100), true); await flush();
    assert.equal(pipeline.busy, false); assert.match(unavailable.at(-1)!, /invalid or mismatched/);
  }
  pipeline.submit(frame(), async () => { throw new Error("encoder failed"); }, 100); await flush();
  assert.equal(calls, 4); assert.match(unavailable.at(-1)!, /could not be prepared/);
  output = new Error("provider failed"); pipeline.submit(frame(), async () => new Blob(), 100); await flush();
  assert.match(unavailable.at(-1)!, /Identification failed/); assert.equal(pipeline.busy, false); assert.equal(results, 0);
  for (const capturedAt of [100 - MAX_LIVE_FRAME_AGE_MS, 101, NaN])
    assert.equal(pipeline.submit(frame(), async () => { throw new Error("should not encode"); }, capturedAt), false);
  pipeline.dispose();
});
