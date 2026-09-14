import assert from "node:assert/strict";
import test from "node:test";
import { EventEmitter } from "node:events";
import type { IncomingMessage, ServerResponse } from "node:http";
import { Readable } from "node:stream";
import { createIdentificationHandler, IDENTIFICATION_BUSY, IDENTIFICATION_UNAVAILABLE, MAX_IDENTIFICATION_BODY_BYTES, MAX_IDENTIFICATION_RESPONSE_BYTES } from "../../../api/identify";

const frame = { mediaId: "capture-unique-session", frameNumber: 7, timestampMs: 187.125, width: 1280, height: 720 };
const capture = () => ({ frame: { ...frame }, imageBase64: Buffer.from([255, 216, 255, 224, 0, 0, 255, 217]).toString("base64") });
const ready = () => ({ status: "ready", model: { id: "holospex-deeplabv3-mobilenetv3", version: "trained-v1" }, minimumConfidence: 0.7, dataset: "Endoscapes-Seg50" });
const prediction = () => ({ schemaVersion: "1.0.0", ...frame, coordinateSpace: "original_pixels", source: "ml_prediction", status: "ok", model: ready().model, structures: [] });
type Handler = ReturnType<typeof createIdentificationHandler>;
interface CallOptions { method?: string; headers?: Record<string, string | undefined>; raw?: string; parsed?: unknown; onRequest?: (request: IncomingMessage) => void }
async function call(handler: Handler, body: unknown = capture(), options: CallOptions = {}) {
  const request = Readable.from([options.raw ?? JSON.stringify(body)]) as unknown as IncomingMessage & { body?: unknown };
  request.method = options.method ?? "POST";
  request.headers = { host: "demo.vercel.app", origin: "https://demo.vercel.app", "content-type": "application/json", ...options.headers };
  if (options.parsed !== undefined) request.body = options.parsed;
  const response = Object.assign(new EventEmitter(), {
    statusCode: 0, writableEnded: false, destroyed: false,
    setHeader(name: string, value: string) { headers[name] = value; },
    end(value: string) { text = value; this.writableEnded = true; },
  });
  const headers: Record<string, string> = {}; let text = "";
  options.onRequest?.(request);
  await handler(request, response as unknown as ServerResponse);
  return { status: response.statusCode, headers, body: text ? JSON.parse(text) : null };
}
function handler(fetchImpl: typeof fetch, more: Parameters<typeof createIdentificationHandler>[0] = {}) {
  return createIdentificationHandler({ endpoint: "https://private-model.example/identify", fetchImpl, ...more });
}

test("identification readiness returns only the public model contract and uses server credentials", async () => {
  let calls = 0;
  const api = handler(async (url, options) => {
    calls++; assert.equal(url, "https://private-model.example/identify"); assert.equal(options?.method, "GET");
    assert.equal(options.body, undefined); assert.equal(options.redirect, "error"); assert.equal(options.credentials, "omit");
    assert.equal(new Headers(options.headers).get("Authorization"), "Bearer private-token");
    return Response.json({ ...ready(), privatePath: "/local/weights.pt", token: "private-token" });
  }, { token: "private-token" });
  const result = await call(api, undefined, { method: "GET", headers: { origin: undefined, "sec-fetch-site": "same-origin" } });
  assert.equal(calls, 1); assert.equal(result.status, 200); assert.deepEqual(result.body, ready());
  assert.equal(result.headers["Cache-Control"], "no-store"); assert.equal(result.headers["X-Content-Type-Options"], "nosniff");
});

test("identification POST forwards the validated JSON capture and returns exact model geometry", async () => {
  const result = prediction();
  const api = handler(async (url, options) => {
    assert.equal(url, "https://private-model.example/identify"); assert.equal(options?.method, "POST");
    assert.equal(new Headers(options.headers).get("Content-Type"), "application/json");
    assert.deepEqual(JSON.parse(String(options.body)), capture());
    return Response.json(result);
  });
  assert.deepEqual((await call(api)).body, result);
  assert.equal((await call(api, undefined, { parsed: capture() })).status, 200, "Vercel's parsed request bodies remain bounded and validated");
});

test("origin and method checks reject cross-origin GET/POST without calling the model", async () => {
  const api = handler(async () => { throw new Error("Unexpected upstream call"); });
  assert.equal((await call(api, undefined, { method: "PUT" })).status, 405);
  for (const method of ["GET", "POST"]) {
    for (const headers of [
      { origin: "https://other.example" }, { origin: "null" }, { origin: "http://demo.vercel.app" },
      { "sec-fetch-site": "cross-site" }, { "sec-fetch-site": "same-site" }, { origin: undefined },
    ]) assert.equal((await call(api, undefined, { method, headers })).status, 403);
  }
  assert.equal((await call(api, undefined, { headers: { "content-type": "text/plain" } })).status, 415);
  assert.equal((await call(api, undefined, { method: "OPTIONS" })).headers.Allow, "GET, POST");
});

test("unconfigured, unsafe or unreachable private model settings remain unavailable without exposing details", async () => {
  let calls = 0;
  const failedFetch: typeof fetch = async () => { calls++; throw new Error("secret-token /private/checkpoint.pt"); };
  for (const endpoint of ["", "file:///weights", "http://public.example/identify", "https://secret:password@model.example", "https://model.example/#secret"]) {
    const result = await call(createIdentificationHandler({ endpoint, fetchImpl: failedFetch }));
    assert.equal(result.status, 503); assert.deepEqual(result.body, { status: "unavailable", message: IDENTIFICATION_UNAVAILABLE });
  }
  assert.equal(calls, 0);
  const unavailable = await call(handler(failedFetch));
  assert.equal(unavailable.status, 503); assert.equal(unavailable.body.message, IDENTIFICATION_UNAVAILABLE);
  assert.equal((await call(handler(async () => Response.json(ready()), { endpoint: "http://127.0.0.1:8765/identify" }), undefined,
    { method: "GET", headers: { host: "127.0.0.1:5174", origin: "http://127.0.0.1:5174" } })).status, 200);
});

test("capture validation rejects endpoints, bad identity and non-JPEG payloads before upload", async () => {
  let calls = 0;
  const api = handler(async () => { calls++; return Response.json(prediction()); });
  for (const body of [
    { ...capture(), endpoint: "https://attacker.example" }, { ...capture(), token: "override" },
    { ...capture(), frame: { ...frame, frameNumber: -1 } }, { ...capture(), frame: { ...frame, timestampMs: null } },
    { ...capture(), frame: { ...frame, mediaId: "" } }, { ...capture(), frame: { ...frame, width: 0 } },
    { ...capture(), frame: { ...frame, width: 4097, height: 1 } }, { ...capture(), frame: { ...frame, width: 1, height: 4097 } },
    { ...capture(), frame: { ...frame, width: 4096, height: 1025 } },
    { ...capture(), frame: { ...frame, width: 100000, height: 100000 } }, { ...capture(), frame: { ...frame, extra: true } },
    { ...capture(), imageBase64: "data:image/jpeg;base64,/9j/" }, { ...capture(), imageBase64: "bad!" },
    { ...capture(), imageBase64: Buffer.from("not a JPEG").toString("base64") }, null,
  ]) assert.equal((await call(api, body)).status, 400);
  assert.equal((await call(api, undefined, { raw: "{" })).status, 400); assert.equal(calls, 0);
});

test("request bounds cover Content-Length, streamed and pre-parsed bodies", async () => {
  let calls = 0;
  const api = handler(async () => { calls++; return Response.json(prediction()); });
  for (const options of [
    { headers: { "content-length": String(MAX_IDENTIFICATION_BODY_BYTES + 1) } },
    { headers: { "content-length": "invalid" } },
    { raw: " ".repeat(MAX_IDENTIFICATION_BODY_BYTES + 1) },
    { parsed: { ...capture(), imageBase64: "A".repeat(MAX_IDENTIFICATION_BODY_BYTES) } },
  ]) assert.equal((await call(api, undefined, options)).status, 413);
  assert.equal(calls, 0);
});

test("the model result must match the original capture exactly and be a canonical prediction", async () => {
  for (const value of [
    { ...prediction(), mediaId: "another-session" }, { ...prediction(), frameNumber: 8 },
    { ...prediction(), timestampMs: frame.timestampMs + 0.001 }, { ...prediction(), width: 640 }, { ...prediction(), height: 480 },
    { ...prediction(), source: "synthetic_mock" }, { ...prediction(), source: "reviewed_annotation" },
    { ...prediction(), coordinateSpace: "normalized" }, { ...prediction(), status: "error", statusReason: "unavailable", structures: [{ unsupported: true }] },
  ]) assert.equal((await call(handler(async () => Response.json(value)))).status, 502);
  const missing = { ...prediction(), status: "missing", statusReason: "Unsupported view" };
  assert.deepEqual((await call(handler(async () => Response.json(missing)))).body, missing);
  const propagated = { ...prediction(), source: "propagated_prediction", propagatedFromTimestampMs: 0 };
  assert.deepEqual((await call(handler(async () => Response.json(propagated)))).body, propagated);
});

test("upstream errors, non-JSON, malformed readiness and oversized responses never become predictions", async () => {
  for (const provider of [
    () => new Response("private diagnostics", { status: 500 }),
    () => new Response("{}", { headers: { "content-type": "text/html" } }),
    () => new Response("{", { headers: { "content-type": "application/json" } }),
    () => new Response("{}", { headers: { "content-type": "application/json", "content-length": String(MAX_IDENTIFICATION_RESPONSE_BYTES + 1) } }),
    () => new Response(" ".repeat(MAX_IDENTIFICATION_RESPONSE_BYTES + 1), { headers: { "content-type": "application/json" } }),
  ]) assert.ok((await call(handler(async () => provider()))).status >= 500);
  for (const status of [{ ...ready(), minimumConfidence: null }, { ...ready(), minimumConfidence: 1.1 }, { ...ready(), dataset: "unrelated" }, { ...ready(), model: {} }, { status: "unavailable" }])
    assert.equal((await call(handler(async () => Response.json(status)), undefined, { method: "GET" })).status, 503);
});

test("upstream busy responses expose only a fixed retry delay and safe message", async () => {
  let cancelled = false;
  const upstream = new Response(new ReadableStream({
    start(controller) { controller.enqueue(new TextEncoder().encode("private-token /private/checkpoint.pt")); },
    cancel() { cancelled = true; },
  }), { status: 429, headers: { "content-type": "text/html", "retry-after": "9999", "x-private-token": "private-token" } });
  const result = await call(handler(async () => upstream));
  assert.equal(result.status, 429);
  assert.deepEqual(result.body, { status: "unavailable", message: IDENTIFICATION_BUSY });
  assert.equal(result.headers["Retry-After"], "2");
  assert.equal(result.headers["x-private-token"], undefined);
  assert.equal(cancelled, true, "discard the upstream diagnostics without reading them");

  const redirected = new Response("private diagnostics", { status: 429 });
  Object.defineProperty(redirected, "redirected", { value: true });
  const rejected = await call(handler(async () => redirected));
  assert.equal(rejected.status, 503);
  assert.deepEqual(rejected.body, { status: "unavailable", message: IDENTIFICATION_UNAVAILABLE });
  assert.equal(rejected.headers["Retry-After"], undefined);
});

test("deadline cancels a pending fetch and hides internal errors", async () => {
  let cancelled = false;
  const api = handler(async (_url, options) => new Promise((_resolve, reject) => {
    options?.signal?.addEventListener("abort", () => { cancelled = true; reject(new Error("private provider timeout")); });
  }), { timeoutMs: 10 });
  const result = await call(api);
  assert.equal(result.status, 504); assert.equal(result.body.message, IDENTIFICATION_UNAVAILABLE); assert.equal(cancelled, true);
});

test("client disconnection cancels an in-flight model request", async () => {
  let request: IncomingMessage; let cancelled = false;
  const api = handler(async (_url, options) => new Promise((_resolve, reject) => {
    options?.signal?.addEventListener("abort", () => { cancelled = true; reject(new Error("aborted")); });
    request.emit("aborted");
  }));
  await call(api, undefined, { onRequest: value => { request = value; } });
  assert.equal(cancelled, true);
});
