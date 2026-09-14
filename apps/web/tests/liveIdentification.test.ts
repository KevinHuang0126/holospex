import assert from "node:assert/strict";
import test from "node:test";
import { IdentificationBusyError, identifyLiveFrame, loadIdentificationModel, validateLiveResult, type LiveFrame } from "../src/input/liveIdentification";

const frame: LiveFrame = { mediaId: "live-session-839c", frameNumber: 7, timestampMs: 1234.56789, width: 1280, height: 720 };
const jpeg = new Blob([new Uint8Array([255, 216, 255, 224])], { type: "image/jpeg" });
const flush = () => new Promise<void>(resolve => setImmediate(resolve));
function prediction() {
  return { schemaVersion: "1.0.0", ...frame, coordinateSpace: "original_pixels", source: "ml_prediction", status: "ok",
    model: { id: "holospex-segmentation", version: "test-v1" }, structures: [{ instanceId: "duct-0", structureId: "cystic_duct",
      polygon: [[420, 280], [450, 275], [465, 320]], confidence: 0.82, visibility: "visible" }] };
}

test("live identification sends an exact frame/JPEG JSON pair only to the fixed same-origin endpoint", async t => {
  const calls: { url: unknown; init?: RequestInit }[] = [];
  t.mock.method(globalThis, "fetch", async (url: unknown, init?: RequestInit) => { calls.push({ url, init }); return Response.json(prediction()); });
  assert.equal(calls.length, 0);
  const controller = new AbortController();
  const captured = { ...frame, ignoredExtra: "not part of the wire format" };
  const result = await identifyLiveFrame(captured, jpeg, controller.signal);
  assert.deepEqual(result, prediction()); assert.equal(calls.length, 1);
  const { url, init } = calls[0];
  assert.equal(url, "/api/identify");
  assert.equal(init?.method, "POST"); assert.equal(init?.credentials, "same-origin"); assert.equal(init?.redirect, "error");
  assert.equal(init?.signal, controller.signal); assert.equal(init?.referrerPolicy, "no-referrer");
  assert.deepEqual(init?.headers, { Accept: "application/json", "Content-Type": "application/json" });
  const body = JSON.parse(init?.body as string);
  assert.deepEqual(Object.keys(body), ["frame", "imageBase64"]);
  assert.deepEqual(body.frame, frame);
  assert.deepEqual(new Uint8Array(Buffer.from(body.imageBase64, "base64")), new Uint8Array(await jpeg.arrayBuffer()));
  assert.ok(!body.imageBase64.startsWith("data:"));
});

test("live result identity is exact across session, counter, fractional timestamp and original dimensions", () => {
  assert.deepEqual(validateLiveResult(prediction(), frame), prediction());
  for (const mismatch of [{ mediaId: "previous-session" }, { frameNumber: 6 }, { timestampMs: frame.timestampMs + 0.000001 },
    { timestampMs: Math.round(frame.timestampMs) }, { width: 640 }, { height: 360 }])
    assert.throws(() => validateLiveResult({ ...prediction(), ...mismatch }, frame), /does not match/);
  for (const malformed of [null, {}, [prediction()], { result: prediction() }, { ...prediction(), coordinateSpace: "normalized" },
    { ...prediction(), model: undefined }, { ...prediction(), structures: [{ ...prediction().structures[0], confidence: undefined }] },
    { ...prediction(), structures: [{ ...prediction().structures[0], polygon: [[0, 0], [1281, 10], [10, 20]] }] }])
    assert.throws(() => validateLiveResult(malformed, frame));
  for (const source of ["synthetic_mock", "reviewed_annotation"]) assert.throws(() => validateLiveResult({ ...prediction(), source }, frame), /predictions only/);
  assert.throws(() => validateLiveResult(prediction(), { ...frame, timestampMs: NaN }), /capture-session/);
});

test("propagated provenance and explicit unavailable output retain their canonical meaning", () => {
  const propagated = { ...prediction(), source: "propagated_prediction", propagatedFromTimestampMs: 1200 };
  assert.equal(validateLiveResult(propagated, frame).source, "propagated_prediction");
  assert.throws(() => validateLiveResult({ ...propagated, propagatedFromTimestampMs: frame.timestampMs }, frame));
  for (const status of ["unsupported", "missing", "error"]) {
    const unavailable = { ...prediction(), status, statusReason: "No supported anatomy for this capture.", structures: [] };
    assert.deepEqual(validateLiveResult(unavailable, frame), unavailable);
    assert.throws(() => validateLiveResult({ ...unavailable, structures: prediction().structures }, frame));
  }
  assert.deepEqual(validateLiveResult({ ...prediction(), structures: [] }, frame).structures, []);
});

test("model readiness supplies the trained model and confidence threshold without sending a frame", async t => {
  const ready = { status: "ready", model: { id: "holospex-deeplabv3-mobilenetv3", version: "person1-trained-v1" }, minimumConfidence: 0.72, dataset: "Endoscapes-Seg50" };
  const calls: { url: unknown; init?: RequestInit }[] = [];
  let value: unknown = ready;
  t.mock.method(globalThis, "fetch", async (url: unknown, init?: RequestInit) => { calls.push({ url, init }); return Response.json(value); });
  const signal = new AbortController().signal;
  assert.deepEqual(await loadIdentificationModel(signal), ready);
  assert.equal(calls[0].url, "/api/identify"); assert.equal(calls[0].init?.method, "GET");
  assert.equal(calls[0].init?.body, undefined); assert.equal(calls[0].init?.credentials, "same-origin");
  assert.equal(calls[0].init?.redirect, "error"); assert.ok(calls[0].init?.signal instanceof AbortSignal);
  for (const malformed of [null, {}, { ...ready, status: "unavailable" }, { ...ready, dataset: "generic-images" },
    { ...ready, model: { id: "", version: "test-v1" } }, { ...ready, model: undefined },
    { ...ready, minimumConfidence: undefined }, { ...ready, minimumConfidence: -1 }, { ...ready, minimumConfidence: 1.1 },
    { ...ready, minimumConfidence: "0.72" }]) {
    value = malformed;
    await assert.rejects(loadIdentificationModel(signal), /valid readiness/);
  }
  const aborted = new AbortController(); aborted.abort(); const before = calls.length;
  await assert.rejects(loadIdentificationModel(aborted.signal), { name: "AbortError" }); assert.equal(calls.length, before);
});

test("model readiness times out at five seconds and cleans up deadlines and cancellation listeners", async t => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  let requestSignal: AbortSignal | undefined;
  let resolve!: (response: Response) => void;
  t.mock.method(globalThis, "fetch", (_url: unknown, init?: RequestInit) => {
    requestSignal = init?.signal ?? undefined;
    return new Promise<Response>(done => { resolve = done; });
  });
  const pending = loadIdentificationModel(new AbortController().signal);
  const rejected = assert.rejects(pending, /readiness check timed out/);
  t.mock.timers.tick(5000);
  await rejected;
  assert.equal(requestSignal?.aborted, true);
  let lateBodyCancelled = false;
  resolve(new Response(new ReadableStream({ cancel() { lateBodyCancelled = true; } })));
  await flush();
  assert.equal(lateBodyCancelled, true);

  const ready = { status: "ready", model: { id: "person1-model", version: "trained-v1" }, minimumConfidence: 0.72, dataset: "Endoscapes-Seg50" };
  t.mock.method(globalThis, "fetch", async (_url: unknown, init?: RequestInit) => { requestSignal = init?.signal ?? undefined; return Response.json(ready); });
  const caller = new AbortController();
  await loadIdentificationModel(caller.signal);
  caller.abort(); t.mock.timers.tick(6000);
  assert.equal(requestSignal?.aborted, false, "Settled readiness must release both its deadline and caller abort listener");
});

test("in-flight requests retain their captured identity even if the caller reuses a mutable frame object", async t => {
  let resolve!: (value: Response) => void;
  t.mock.method(globalThis, "fetch", () => new Promise<Response>(done => { resolve = done; }));
  const mutable = { ...frame };
  const pending = identifyLiveFrame(mutable, jpeg, new AbortController().signal);
  mutable.frameNumber++; mutable.timestampMs += 100;
  await flush();
  resolve(Response.json(prediction()));
  assert.deepEqual(await pending, prediction());
});

test("HTTP failures, redirects and malformed JSON cannot produce live overlays", async t => {
  const replies = [new Response("Internal details", { status: 503 }), new Response("<html>Not an identification result</html>"), Response.json([prediction()])];
  t.mock.method(globalThis, "fetch", async () => replies.shift()!);
  const signal = new AbortController().signal;
  await assert.rejects(identifyLiveFrame(frame, jpeg, signal), /HTTP 503/);
  await assert.rejects(identifyLiveFrame(frame, jpeg, signal), /single JSON response/);
  await assert.rejects(identifyLiveFrame(frame, jpeg, signal));
  const redirected = Response.json(prediction()); Object.defineProperty(redirected, "redirected", { value: true }); replies.push(redirected);
  await assert.rejects(identifyLiveFrame(frame, jpeg, signal), /redirects are not allowed/);
  await assert.rejects(identifyLiveFrame(frame, new Blob(["png"], { type: "image/png" }), signal), /JPEG capture/);
  replies.push(Response.json({ status: "error", message: "Person 1's trained checkpoint is unavailable." }, { status: 503 }));
  await assert.rejects(loadIdentificationModel(signal), /trained checkpoint is unavailable/);
});

test("busy responses are typed and sanitized without reading an upstream error body", async t => {
  let cancelled = false;
  t.mock.method(globalThis, "fetch", async () => new Response(new ReadableStream({ cancel() { cancelled = true; } }),
    { status: 429, headers: { "Retry-After": "999999" } }));
  await assert.rejects(identifyLiveFrame(frame, jpeg, new AbortController().signal), (error: unknown) => {
    assert.ok(error instanceof IdentificationBusyError);
    assert.equal(error.message, "Identification is busy.");
    assert.equal(error.retryAfterMs, 2000);
    return true;
  });
  assert.equal(cancelled, true);
});

test("response limits cancel oversized declared and streamed bodies before parsing", async t => {
  let cancelled = 0;
  const declared = new Response(new ReadableStream({ cancel() { cancelled++; } }), { headers: { "Content-Length": String(2 * 1024 * 1024 + 1) } });
  let chunks = 0;
  const streamed = new Response(new ReadableStream({ pull(controller) { controller.enqueue(new Uint8Array(1024 * 1024)); chunks++; }, cancel() { cancelled++; } }));
  const replies = [declared, streamed];
  t.mock.method(globalThis, "fetch", async () => replies.shift()!);
  const signal = new AbortController().signal;
  await assert.rejects(identifyLiveFrame(frame, jpeg, signal), /2 MiB limit/);
  await assert.rejects(identifyLiveFrame(frame, jpeg, signal), /2 MiB limit/);
  assert.equal(cancelled, 2); assert.ok(chunks <= 4);
});

test("cancellation blocks dispatch and prevents late fetch/body completion from publishing a result", async t => {
  let calls = 0, resolve!: (value: Response) => void;
  t.mock.method(globalThis, "fetch", () => { calls++; return new Promise<Response>(done => { resolve = done; }); });
  const before = new AbortController(); before.abort();
  await assert.rejects(identifyLiveFrame(frame, jpeg, before.signal), { name: "AbortError" }); assert.equal(calls, 0);
  const during = new AbortController(), pending = identifyLiveFrame(frame, jpeg, during.signal);
  await flush();
  let lateBodyCancelled = false;
  during.abort(); resolve(new Response(new ReadableStream({ cancel() { lateBodyCancelled = true; } })));
  await assert.rejects(pending, { name: "AbortError" }); assert.equal(lateBodyCancelled, true);
  let bodyCancelled = false;
  t.mock.method(globalThis, "fetch", async () => new Response(new ReadableStream({ cancel() { bodyCancelled = true; } })));
  const reading = new AbortController(), readingResult = identifyLiveFrame(frame, jpeg, reading.signal);
  await new Promise(done => setTimeout(done, 0)); reading.abort();
  await assert.rejects(readingResult, { name: "AbortError" }); assert.equal(bodyCancelled, true);
});

test("oversized captures and cancellation during JPEG reading never dispatch identification", async t => {
  let calls = 0;
  t.mock.method(globalThis, "fetch", async () => { calls++; return Response.json(prediction()); });
  const signal = new AbortController().signal;
  await assert.rejects(identifyLiveFrame(frame, new Blob([new Uint8Array(3 * 1024 * 1024 + 1)], { type: "image/jpeg" }), signal), /3 MiB/);
  await assert.rejects(identifyLiveFrame({ ...frame, width: 4097 }, jpeg, signal), /capture-session/);
  await assert.rejects(identifyLiveFrame({ ...frame, width: 3840, height: 2160 }, jpeg, signal), /capture-session/);
  await assert.rejects(identifyLiveFrame(frame, new Blob(["not a JPEG"], { type: "image/jpeg" }), signal), /valid JPEG/);
  const reading = new AbortController();
  let resolve!: (value: ArrayBuffer) => void;
  const pendingImage = new Blob([new Uint8Array([255, 216, 255])], { type: "image/jpeg" });
  t.mock.method(pendingImage, "arrayBuffer", () => new Promise<ArrayBuffer>(done => { resolve = done; }));
  const pending = identifyLiveFrame(frame, pendingImage, reading.signal);
  reading.abort(); resolve(new Uint8Array([255, 216, 255]).buffer);
  await assert.rejects(pending, { name: "AbortError" }); assert.equal(calls, 0);
});

test("the largest accepted JPEG produces a request below the hosting platform's 4.5 MB limit", async t => {
  let body = "";
  t.mock.method(globalThis, "fetch", async (_url: unknown, init?: RequestInit) => {
    body = init?.body as string; return Response.json(prediction());
  });
  const bytes = new Uint8Array(3 * 1024 * 1024);
  bytes.set([255, 216, 255]);
  await identifyLiveFrame(frame, new Blob([bytes], { type: "image/jpeg" }), new AbortController().signal);
  assert.ok(Buffer.byteLength(body) < 4_500_000);
  assert.deepEqual(Buffer.from(JSON.parse(body).imageBase64, "base64"), Buffer.from(bytes));
});
