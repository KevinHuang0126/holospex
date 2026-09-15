import assert from "node:assert/strict";
import test from "node:test";
import { EventEmitter } from "node:events";
import type { IncomingMessage, ServerResponse } from "node:http";
import { Readable } from "node:stream";
import { identificationApiPlugin } from "../dev/identificationApi";
import { IDENTIFICATION_UNAVAILABLE } from "../../../api/identify";

type Hook = "configureServer" | "configurePreviewServer";
type Middleware = (request: IncomingMessage, response: ServerResponse, next: () => void) => void;
const readiness = { status: "ready", model: { id: "test-model", version: "trained-v1" }, minimumConfidence: 0.5,
  supportsMinimumConfidence: true, dataset: "Endoscapes-Seg50" };
const frame = { mediaId: "test-local-capture", frameNumber: 0, timestampMs: 0, width: 320, height: 180 };
const capture = { frame, imageBase64: Buffer.from([255, 216, 255, 224, 255, 217]).toString("base64"), minimumConfidence: 0.24 };
const prediction = { schemaVersion: "1.0.0", ...frame, coordinateSpace: "original_pixels", source: "ml_prediction",
  status: "ok", model: readiness.model, structures: [] };

function middleware(hook: Hook, environment: Parameters<typeof identificationApiPlugin>[0] = {}) {
  const plugin = identificationApiPlugin(environment), installed: Middleware[] = [];
  assert.equal(plugin.apply, "serve");
  assert.equal(plugin.config, undefined, "private settings do not become browser config/defines");
  const configure = plugin[hook];
  assert.equal(typeof configure, "function", `${hook} must install the API before Vite's static fallback`);
  (configure as (server: unknown) => void)({ middlewares: { use: (value: Middleware) => installed.push(value) } });
  assert.equal(installed.length, 1);
  return installed[0];
}

async function request(api: Middleware, url: string | undefined, options: { method?: string; body?: unknown; headers?: Record<string, string | undefined> } = {}) {
  const input = Readable.from([options.body === undefined ? "" : JSON.stringify(options.body)]) as unknown as IncomingMessage;
  input.url = url; input.method = options.method ?? "GET";
  input.headers = { host: "127.0.0.1:4174", "sec-fetch-site": "same-origin", ...options.headers };
  let finish!: (value: { next: boolean; status: number; body: unknown; headers: Record<string, string> }) => void;
  const done = new Promise<Parameters<typeof finish>[0]>(resolve => { finish = resolve; });
  const headers: Record<string, string> = {};
  const output = Object.assign(new EventEmitter(), {
    statusCode: 0, headersSent: false, writableEnded: false, destroyed: false,
    setHeader(name: string, value: string) { headers[name] = value; },
    end(body: string) {
      this.headersSent = true; this.writableEnded = true;
      finish({ next: false, status: this.statusCode, body: JSON.parse(body), headers });
    },
  });
  api(input, output as unknown as ServerResponse, () => finish({ next: true, status: output.statusCode, body: null, headers }));
  return done;
}

for (const hook of ["configureServer", "configurePreviewServer"] as const) {
  test(`${hook}: unrelated routes pass through without contacting the model`, async t => {
    let calls = 0;
    t.mock.method(globalThis, "fetch", async () => { calls++; throw new Error("Unexpected model request"); });
    const api = middleware(hook);
    for (const path of [undefined, "/mannequin", "/assets/app.js", "/api/placement", "/api/identify-extra", "/api/identify/frame"]) {
      const result = await request(api, path);
      assert.equal(result.next, true); assert.equal(result.status, 0); assert.deepEqual(result.headers, {});
    }
    assert.equal(calls, 0);
  });

  test(`${hook}: configured bridge authenticates readiness and captures without exposing credentials`, async t => {
    const endpoint = "https://private-model.example/identify", token = "server-only-test-token";
    let calls = 0;
    t.mock.method(globalThis, "fetch", async (url: unknown, init?: RequestInit) => {
      calls++; assert.equal(url, endpoint);
      assert.equal(new Headers(init?.headers).get("Authorization"), `Bearer ${token}`);
      assert.equal(init?.redirect, "error"); assert.equal(init?.credentials, "omit");
      if (init?.method === "GET") {
        assert.equal(init.body, undefined);
        return Response.json({ ...readiness, token, privateEndpoint: endpoint, privatePath: "/local/weights.pt" });
      }
      assert.equal(init?.method, "POST"); assert.deepEqual(JSON.parse(String(init.body)), capture);
      return Response.json(prediction);
    });
    const api = middleware(hook, { HOLOSPEX_IDENTIFICATION_URL: endpoint, HOLOSPEX_IDENTIFICATION_TOKEN: token });
    const ready = await request(api, "/api/identify?readiness=1");
    assert.equal(ready.next, false); assert.equal(ready.status, 200); assert.deepEqual(ready.body, readiness);
    assert.equal(ready.headers["Cache-Control"], "no-store");
    const result = await request(api, "/api/identify", { method: "POST", body: capture,
      headers: { origin: "http://127.0.0.1:4174", "content-type": "application/json" } });
    assert.equal(result.status, 200); assert.deepEqual(result.body, prediction);
    for (const output of [ready, result]) {
      const serialized = JSON.stringify(output);
      for (const secret of [token, endpoint, "/local/weights.pt", "Authorization"])
        assert.equal(serialized.includes(secret), false);
    }
    assert.equal(calls, 2);
  });

  test(`${hook}: loopback default retains same-origin authorization checks`, async t => {
    let calls = 0;
    t.mock.method(globalThis, "fetch", async (url: unknown, init?: RequestInit) => {
      calls++; assert.equal(url, "http://127.0.0.1:8765/identify");
      assert.equal(new Headers(init?.headers).has("Authorization"), false);
      return Response.json(readiness);
    });
    const api = middleware(hook, { HOLOSPEX_IDENTIFICATION_TOKEN: "" });
    assert.equal((await request(api, "/api/identify", { headers: { "sec-fetch-site": "cross-site" } })).status, 403);
    assert.equal((await request(api, "/api/identify", { method: "POST", body: capture,
      headers: { origin: "https://other.example", "content-type": "application/json" } })).status, 403);
    assert.equal(calls, 0);
    assert.equal((await request(api, "/api/identify")).status, 200);
    assert.equal(calls, 1);
  });

  test(`${hook}: unavailable private provider returns only the sanitized error`, async t => {
    t.mock.method(globalThis, "fetch", async () => { throw new Error("server-only-test-token /local/weights.pt"); });
    const result = await request(middleware(hook), "/api/identify");
    assert.equal(result.next, false); assert.equal(result.status, 503);
    assert.deepEqual(result.body, { status: "unavailable", message: IDENTIFICATION_UNAVAILABLE });
    assert.equal(result.headers["Cache-Control"], "no-store");
  });
}
