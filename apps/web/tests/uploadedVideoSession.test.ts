import assert from "node:assert/strict";
import test from "node:test";
import { createUploadedVideoSession, videoCaptureSize, type UploadedVideoCapture } from "../src/camera/uploadedVideoSession";
import type { DisplayedFrame } from "../src/overlays/selectFrame";

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>(yes => { resolve = yes; });
  return { promise, resolve };
}
const flush = () => new Promise<void>(resolve => setImmediate(resolve));
const capture = (frameNumber: number, timestampMs: number, mediaId = "upload:clip-a") => ({
  image: { pixels: `${mediaId}:${frameNumber}` }, capturedAt: performance.now(),
  frame: { mediaId, frameNumber, timestampMs, width: 1280, height: 720 },
});
const prediction = (frame: DisplayedFrame) => ({ schemaVersion: "1.0.0", ...frame, coordinateSpace: "original_pixels",
  source: "ml_prediction", status: "ok", model: { id: "test-model", version: "test-v1" },
  structures: [{ instanceId: "gallbladder-0", structureId: "gallbladder", confidence: 0.9, visibility: "visible",
    polygon: [[10, 10], [50, 10], [50, 50], [10, 50]] }] });

test("4K landscape, portrait and square uploads stay within the capture budget without cropping", () => {
  for (const [width, height, expected] of [
    [3840, 2160, { width: 1280, height: 720 }], [2160, 3840, { width: 405, height: 720 }],
    [2160, 2160, { width: 720, height: 720 }], [854, 480, { width: 854, height: 480 }],
  ] as const) {
    const size = videoCaptureSize(width, height)!;
    assert.deepEqual(size, expected);
    assert.ok(Math.abs(size.width / size.height - width / height) < 0.002);
    assert.ok(size.width <= 1280 && size.height <= 720);
  }
  for (const [width, height] of [[0, 720], [1280, 0], [-1, 720], [Infinity, 720], [NaN, 480]])
    assert.equal(videoCaptureSize(width, height), null);
});

test("a pending video request retains its owned pixels while the source video advances", async () => {
  const pending = deferred<unknown>(), original = capture(1, 3200);
  const displayed: UploadedVideoCapture<{ pixels: string }>[] = [];
  const session = createUploadedVideoSession<{ pixels: string }>({
    onResult: (image, result) => { assert.deepEqual(result.timestampMs, 3200); displayed.push(image); },
    onUnavailable: message => assert.fail(message),
  });
  session.setIdentifier(async () => pending.promise);
  assert.equal(session.submit(original, async () => new Blob(["first frame"])), true);
  await flush();
  original.frame.timestampMs = 6500;
  assert.equal(session.submit(capture(2, 6500), async () => new Blob()), false);
  pending.resolve(prediction({ ...original.frame, timestampMs: 3200 })); await flush();
  assert.equal(displayed.length, 1); assert.equal(displayed[0].image, original.image);
  assert.equal(displayed[0].frame.timestampMs, 3200);
  session.dispose();
});

test("seeking forward/back, pausing and replacing a clip reject old results and keep a single request slot", async () => {
  const requests: { frame: DisplayedFrame; signal: AbortSignal; task: ReturnType<typeof deferred<unknown>> }[] = [];
  const displayed: DisplayedFrame[] = [];
  const session = createUploadedVideoSession<{ pixels: string }>({ onResult: image => displayed.push(image.frame),
    onUnavailable: message => assert.fail(message) });
  session.setIdentifier(async (frame, _, signal) => {
    const task = deferred<unknown>(); requests.push({ frame, signal, task }); return task.promise;
  });
  const positions = [capture(0, 0), capture(1, 8000), capture(2, 1500), capture(3, 1700), capture(0, 0, "upload:clip-b")];
  for (let index = 0; index < positions.length - 1; index++) {
    assert.equal(session.submit(positions[index], async () => new Blob()), true); await flush();
    session.invalidate();
    assert.equal(requests[index].signal.aborted, true);
    // Ignore-abort providers must settle before another selected frame is submitted.
    assert.equal(session.busy, true);
    assert.equal(session.submit(positions[index + 1], async () => new Blob()), false);
    requests[index].task.resolve(prediction(requests[index].frame)); await flush();
    assert.equal(session.busy, false); assert.equal(displayed.length, 0);
  }
  assert.equal(session.submit(positions.at(-1)!, async () => new Blob()), true); await flush();
  requests.at(-1)!.task.resolve(prediction(requests.at(-1)!.frame)); await flush();
  assert.deepEqual(displayed, [positions.at(-1)!.frame]); session.dispose();
});

test("model replacement and hidden/source disposal cancel pending encoding before any pixels are sent", async () => {
  const encoding = deferred<Blob>(); let oldCalls = 0, newCalls = 0, results = 0;
  const session = createUploadedVideoSession<{ pixels: string }>({ onResult: () => results++, onUnavailable: message => assert.fail(message) });
  session.setIdentifier(async frame => { oldCalls++; return prediction(frame); });
  session.submit(capture(1, 10), () => encoding.promise);
  session.setIdentifier(async frame => { newCalls++; return prediction(frame); });
  assert.equal(session.submit(capture(2, 20), async () => new Blob()), false);
  encoding.resolve(new Blob()); await flush();
  assert.equal(oldCalls, 0); assert.equal(newCalls, 0);
  session.submit(capture(2, 20), async () => new Blob()); await flush();
  assert.equal(newCalls, 1); assert.equal(results, 1);
  const hiddenEncoding = deferred<Blob>(); session.submit(capture(3, 30), () => hiddenEncoding.promise);
  session.invalidate(); session.dispose(); hiddenEncoding.resolve(new Blob()); await flush();
  assert.equal(newCalls, 1); assert.equal(results, 1);
  assert.equal(session.submit(capture(4, 40), async () => new Blob()), false);
});
