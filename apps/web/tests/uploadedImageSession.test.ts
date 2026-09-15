import assert from "node:assert/strict";
import test from "node:test";
import { createUploadedImageSession, type UploadedImage } from "../src/camera/uploadedImageSession";
import { MAX_LIVE_FRAME_AGE_MS } from "../src/input/liveFramePipeline";
import type { DisplayedFrame } from "../src/overlays/selectFrame";

const flush = () => new Promise<void>(resolve => setImmediate(resolve));
function deferred<T>() {
  let resolve!: (value: T) => void, reject!: (cause: unknown) => void;
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}
type Pixels = { values: number[] };
const image = (value = 1): UploadedImage<Pixels> => ({ image: { values: [value] }, width: 640, height: 480 });
const file = (name = "image.jpg") => new File(["local image"], name);
const prediction = (frame: DisplayedFrame) => ({ schemaVersion: "1.0.0", ...frame, coordinateSpace: "original_pixels",
  source: "ml_prediction", status: "ok", model: { id: "test-model", version: "v1" }, structures: [] });
function setup(decode = async (_file: File, _signal: AbortSignal) => image()) {
  let now = 100, changes = 0;
  const timers = new Map<object, { at: number; run: () => void }>();
  const session = createUploadedImageSession({ decode, copyImage: (source: Pixels) => structuredClone(source),
    encode: async source => new Blob([JSON.stringify(source.values)]), now: () => now,
    scheduleTimeout: (run, delay) => { const handle = {}; timers.set(handle, { at: now + delay, run }); return handle; },
    cancelTimeout: handle => { timers.delete(handle as object); }, onChange: () => changes++,
  });
  return { session, get changes() { return changes; }, advance(duration: number) {
    now += duration; for (const [key, timer] of timers) if (timer.at <= now) { timers.delete(key); timer.run(); }
  } };
}

test("local selection waits for explicit Identify and pairs owned pixels with a stable zero-time image identity", async () => {
  const decoded = image(), context = setup(async () => decoded), { session } = context;
  const output = deferred<unknown>(); let requests = 0, captured: DisplayedFrame | undefined;
  session.setIdentifier(async (frame, blob) => { captured = frame; requests++; assert.equal(await blob.text(), "[1]"); return output.promise; });
  session.setFile(file()); await flush(); assert.equal(session.status, "ready"); assert.equal(requests, 0);
  const identity = session.image!.frame;
  assert.match(identity.mediaId, /^image:/); assert.equal(identity.frameNumber, 0); assert.equal(identity.timestampMs, 0);
  context.advance(60000); assert.equal(session.identify(), true); assert.equal(session.identify(), false);
  decoded.image.values[0] = 9; await flush();
  assert.equal(requests, 1); assert.deepEqual(captured, identity); assert.deepEqual(session.image!.image.values, [1]);
  output.resolve(prediction(identity)); await flush(); context.advance(60000);
  assert.equal(session.status, "identified"); assert.ok(session.result); assert.equal(requests, 1);
  session.clearResult(); await flush(); assert.equal(session.result, null); assert.equal(session.image?.image, decoded.image);
  assert.equal(session.status, "ready"); assert.equal(requests, 1); session.dispose();
});

test("replacing/removing a file suppresses late decoders without starting identification", async () => {
  const pending: { signal: AbortSignal; task: ReturnType<typeof deferred<UploadedImage<Pixels>>> }[] = [];
  const { session } = setup(async (_, signal) => { const task = deferred<UploadedImage<Pixels>>(); pending.push({ signal, task }); return task.promise; });
  let requests = 0; session.setIdentifier(async frame => { requests++; return prediction(frame); });
  session.setFile(file("a.jpg")); session.setFile(file("b.jpg"));
  assert.equal(pending[0].signal.aborted, true); assert.equal(session.image, null);
  pending[1].task.resolve(image(2)); await flush(); const identity = session.image!.frame.mediaId;
  pending[0].task.resolve(image(1)); await flush(); assert.deepEqual(session.image!.image.values, [2]);
  session.setFile(file("c.jpg")); assert.equal(session.image, null);
  session.setFile(null); assert.equal(pending[2].signal.aborted, true);
  pending[2].task.resolve(image(3)); await flush(); assert.equal(session.status, "empty"); assert.equal(session.image, null);
  session.setFile(file("d.jpg")); pending[3].task.resolve(image(4)); await flush();
  assert.notEqual(session.image!.frame.mediaId, identity); assert.equal(requests, 0); session.dispose();
});

test("model/cutoff changes retain the image, abort old results and keep their request slot until settlement", async () => {
  const context = setup(), { session } = context, output = deferred<unknown>();
  let signal: AbortSignal | undefined, calls = 0;
  session.setIdentifier(async (_frame, _blob, value) => { signal = value; calls++; return output.promise; });
  session.setFile(file()); await flush(); const identity = session.image!.frame;
  session.identify(); await flush();
  const next = async (frame: DisplayedFrame) => { calls++; return prediction(frame); };
  session.setIdentifier(next); session.clearResult();
  assert.equal(signal?.aborted, true); assert.equal(session.result, null); assert.equal(session.image?.frame, identity);
  assert.equal(session.busy, true); assert.equal(session.identify(), false); assert.equal(calls, 1);
  output.resolve(prediction(identity)); await flush(); assert.equal(session.busy, false); assert.equal(session.result, null); assert.equal(calls, 1);
  assert.equal(session.identify(), true); await flush(); assert.equal(calls, 2); assert.ok(session.result);
  session.setFile(file("new.jpg")); await flush(); assert.equal(session.result, null); assert.notEqual(session.image!.frame.mediaId, identity.mediaId);
  session.dispose();
});

test("timeouts and mismatched results stay withheld until a manual retry; disposal ignores late work", async () => {
  const context = setup(), { session } = context, output = deferred<unknown>(); let calls = 0;
  session.setIdentifier(async () => { calls++; return output.promise; }); session.setFile(file()); await flush();
  const identity = session.image!.frame; session.identify(); await flush(); context.advance(MAX_LIVE_FRAME_AGE_MS);
  assert.equal(session.status, "error"); assert.match(session.message!, /timed out/); assert.equal(session.result, null);
  output.resolve(prediction(identity)); await flush(); context.advance(60000); assert.equal(calls, 1); assert.equal(session.result, null);
  session.setIdentifier(async frame => prediction({ ...frame, mediaId: "different-image" })); await flush();
  assert.equal(session.status, "ready"); session.identify(); await flush(); assert.equal(session.status, "error"); assert.equal(session.result, null);
  const late = deferred<UploadedImage<Pixels>>(), next = setup(async () => late.promise);
  next.session.setFile(file()); next.session.dispose(); const count = next.changes;
  late.resolve(image()); await flush(); assert.equal(next.changes, count); assert.equal(next.session.image, null);
  session.dispose();
});

test("file replacement and unmount abort providers which ignore cancellation without repainting their results", async () => {
  const context = setup(), { session } = context;
  const pending: { frame: DisplayedFrame; signal: AbortSignal; task: ReturnType<typeof deferred<unknown>> }[] = [];
  session.setIdentifier(async (frame, _blob, signal) => { const task = deferred<unknown>(); pending.push({ frame, signal, task }); return task.promise; });
  session.setFile(file("first.jpg")); await flush(); session.identify(); await flush();
  session.setFile(file("second.jpg")); await flush();
  assert.equal(pending[0].signal.aborted, true); assert.equal(session.busy, true); assert.equal(session.identify(), false);
  const nextIdentity = session.image!.frame;
  pending[0].task.resolve(prediction(pending[0].frame)); await flush();
  assert.equal(session.result, null); assert.equal(session.image!.frame, nextIdentity); assert.equal(pending.length, 1);
  session.identify(); await flush(); session.dispose(); const notifications = context.changes;
  assert.equal(pending[1].signal.aborted, true);
  pending[1].task.resolve(prediction(pending[1].frame)); await flush();
  assert.equal(context.changes, notifications); assert.equal(session.image, null); assert.equal(session.result, null);
});
