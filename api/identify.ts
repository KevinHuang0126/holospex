import type { IncomingMessage, ServerResponse } from "node:http";
import { parseFrameResult } from "../contracts/src/index";

export const MAX_IDENTIFICATION_BODY_BYTES = 6 * 1024 * 1024;
export const MAX_IDENTIFICATION_RESPONSE_BYTES = 2 * 1024 * 1024;
export const IDENTIFICATION_BUSY = "Identification is busy. Waiting for the model to be available.";
export const IDENTIFICATION_UNAVAILABLE = "Person 1’s trained identification model is not available.";
type IdentificationRequest = IncomingMessage & { body?: unknown };
interface FrameIdentity { mediaId: string; frameNumber: number; timestampMs: number; width: number; height: number }
export interface IdentificationHandlerOptions {
  endpoint?: string;
  token?: string;
  fetchImpl?: typeof fetch;
  timeoutMs?: number;
}
class IdentificationError extends Error {
  constructor(readonly status: number, message: string) { super(message); }
}
function header(request: IncomingMessage, name: string): string | null {
  const value = request.headers[name]; return typeof value === "string" ? value : null;
}
function sameOrigin(request: IncomingMessage): boolean {
  const origin = header(request, "origin"), host = header(request, "host"), site = header(request, "sec-fetch-site");
  if (!host || site === "cross-site" || site === "same-site") return false;
  // Same-origin GET fetches normally omit Origin. Fetch Metadata supplies the boundary.
  if (!origin) return request.method === "GET" && site === "same-origin";
  try {
    const url = new URL(origin), local = url.hostname === "localhost" || url.hostname === "[::1]" || /^127(?:\.\d{1,3}){3}$/.test(url.hostname);
    return url.origin === origin && !url.username && !url.password && url.host.toLowerCase() === host.toLowerCase()
      && (url.protocol === "https:" || url.protocol === "http:" && local);
  } catch { return false; }
}
function endpointUrl(value: string): string {
  try {
    const url = new URL(value), local = url.hostname === "localhost" || url.hostname === "[::1]" || /^127(?:\.\d{1,3}){3}$/.test(url.hostname);
    if ((url.protocol !== "https:" && !(url.protocol === "http:" && local)) || url.username || url.password || url.hash) throw new Error();
    return url.href;
  } catch { throw new IdentificationError(503, IDENTIFICATION_UNAVAILABLE); }
}
function record(value: unknown): value is Record<string, unknown> { return !!value && typeof value === "object" && !Array.isArray(value); }
function parseInput(value: unknown): { frame: FrameIdentity; imageBase64: string } {
  if (!record(value) || Object.keys(value).some(key => key !== "frame" && key !== "imageBase64") || !record(value.frame))
    throw new IdentificationError(400, "Send a captured frame and its JPEG image.");
  const frame = value.frame;
  if (Object.keys(frame).some(key => !["mediaId", "frameNumber", "timestampMs", "width", "height"].includes(key))
    || typeof frame.mediaId !== "string" || !frame.mediaId.trim() || frame.mediaId.length > 256
    || !Number.isSafeInteger(frame.frameNumber) || (frame.frameNumber as number) < 0
    || typeof frame.timestampMs !== "number" || !Number.isFinite(frame.timestampMs) || frame.timestampMs < 0
    || !Number.isSafeInteger(frame.width) || (frame.width as number) <= 0 || (frame.width as number) > 4096
    || !Number.isSafeInteger(frame.height) || (frame.height as number) <= 0 || (frame.height as number) > 4096
    || (frame.width as number) * (frame.height as number) > 4_194_304)
    throw new IdentificationError(400, "The captured frame identity or dimensions are invalid.");
  const encoded = value.imageBase64;
  if (typeof encoded !== "string" || !encoded.length || encoded.length % 4 !== 0 || !/^[A-Za-z0-9+/]*={0,2}$/.test(encoded))
    throw new IdentificationError(400, "The capture must contain a base64 JPEG image.");
  const image = Buffer.from(encoded, "base64");
  if (image.length < 4 || image[0] !== 255 || image[1] !== 216 || image[2] !== 255 || image.toString("base64") !== encoded)
    throw new IdentificationError(400, "The capture must contain a base64 JPEG image.");
  return { frame: frame as unknown as FrameIdentity, imageBase64: encoded };
}
function aborted(signal: AbortSignal) { if (signal.aborted) throw new IdentificationError(504, IDENTIFICATION_UNAVAILABLE); }
async function cancellable<T>(operation: Promise<T>, signal: AbortSignal): Promise<T> {
  if (signal.aborted) {
    // The operation may reject synchronously while constructing the promise.
    // Observe it even when cancellation won before the race could be installed.
    void operation.catch(() => {}); aborted(signal);
  }
  let rejectAbort: () => void = () => {};
  const cancelled = new Promise<never>((_resolve, reject) => { rejectAbort = () => reject(new IdentificationError(504, IDENTIFICATION_UNAVAILABLE)); });
  signal.addEventListener("abort", rejectAbort, { once: true });
  try { return await Promise.race([operation, cancelled]); }
  finally { signal.removeEventListener("abort", rejectAbort); }
}
async function readBody(request: IdentificationRequest, signal: AbortSignal): Promise<unknown> {
  const declared = header(request, "content-length");
  if (declared && (!/^\d+$/.test(declared) || Number(declared) > MAX_IDENTIFICATION_BODY_BYTES))
    throw new IdentificationError(413, "Captured requests must be smaller than 6 MiB.");
  let text: string;
  if (request.body !== undefined) {
    try { text = Buffer.isBuffer(request.body) ? request.body.toString("utf8") : typeof request.body === "string" ? request.body : JSON.stringify(request.body); }
    catch { throw new IdentificationError(400, "The capture request is invalid JSON."); }
  } else {
    const chunks: Buffer[] = []; let length = 0;
    const iterator = request[Symbol.asyncIterator]();
    while (true) {
      const next = await cancellable(iterator.next(), signal);
      if (next.done) break;
      const bytes = Buffer.isBuffer(next.value) ? next.value : Buffer.from(next.value);
      length += bytes.length;
      if (length > MAX_IDENTIFICATION_BODY_BYTES) throw new IdentificationError(413, "Captured requests must be smaller than 6 MiB.");
      chunks.push(bytes);
    }
    text = Buffer.concat(chunks).toString("utf8");
  }
  if (typeof text !== "string" || Buffer.byteLength(text) > MAX_IDENTIFICATION_BODY_BYTES)
    throw new IdentificationError(413, "Captured requests must be smaller than 6 MiB.");
  try { return JSON.parse(text); } catch { throw new IdentificationError(400, "The capture request is invalid JSON."); }
}
async function readResponse(response: Response, signal: AbortSignal): Promise<unknown> {
  if (!response.redirected && response.status === 429) {
    await response.body?.cancel().catch(() => {});
    throw new IdentificationError(429, IDENTIFICATION_BUSY);
  }
  if (response.redirected || !response.ok || response.headers.get("content-type")?.split(";")[0].trim().toLowerCase() !== "application/json") {
    await response.body?.cancel(); throw new IdentificationError(503, IDENTIFICATION_UNAVAILABLE);
  }
  const length = response.headers.get("content-length");
  if (length !== null && (!/^\d+$/.test(length) || Number(length) > MAX_IDENTIFICATION_RESPONSE_BYTES)) {
    await response.body?.cancel(); throw new IdentificationError(502, "The identification model returned an invalid response.");
  }
  if (!response.body) throw new IdentificationError(502, "The identification model returned an empty response.");
  const reader = response.body.getReader(), chunks: Uint8Array[] = []; let received = 0;
  const cancel = () => { void reader.cancel().catch(() => {}); };
  signal.addEventListener("abort", cancel, { once: true });
  try {
    while (true) {
      const next = await cancellable(reader.read(), signal); aborted(signal);
      if (next.done) break;
      received += next.value.byteLength;
      if (received > MAX_IDENTIFICATION_RESPONSE_BYTES) throw new IdentificationError(502, "The identification model returned an oversized response.");
      chunks.push(next.value);
    }
    try { return JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(Buffer.concat(chunks))); }
    catch { throw new IdentificationError(502, "The identification model returned invalid JSON."); }
  } finally {
    signal.removeEventListener("abort", cancel);
    await reader.cancel().catch(() => {}); reader.releaseLock();
  }
}
function readyStatus(value: unknown) {
  if (!record(value) || value.status !== "ready" || value.dataset !== "Endoscapes-Seg50" || !record(value.model)
    || typeof value.model.id !== "string" || !value.model.id.trim() || value.model.id.length > 256
    || typeof value.model.version !== "string" || !value.model.version.trim() || value.model.version.length > 256
    || typeof value.minimumConfidence !== "number" || !Number.isFinite(value.minimumConfidence) || value.minimumConfidence < 0 || value.minimumConfidence > 1)
    throw new IdentificationError(503, IDENTIFICATION_UNAVAILABLE);
  // Copy the public contract only; private service diagnostics never reach the browser.
  return { status: "ready", model: { id: value.model.id, version: value.model.version }, minimumConfidence: value.minimumConfidence, dataset: value.dataset };
}

/** A fixed, server-configured model bridge. Captures and credentials are never stored. */
export function createIdentificationHandler(options: IdentificationHandlerOptions = {}) {
  return async (request: IdentificationRequest, response: ServerResponse): Promise<void> => {
    response.setHeader("Cache-Control", "no-store"); response.setHeader("Content-Type", "application/json; charset=utf-8");
    response.setHeader("X-Content-Type-Options", "nosniff");
    const send = (status: number, value: unknown) => {
      if (response.writableEnded || response.destroyed) return;
      response.statusCode = status; response.end(JSON.stringify(value));
    };
    const error = (status: number, message: string) => send(status, { status: "unavailable", message });
    if (request.method !== "GET" && request.method !== "POST") { response.setHeader("Allow", "GET, POST"); error(405, "Use GET for model status or POST for identification."); return; }
    if (!sameOrigin(request)) { error(403, "Open identification from this app’s own URL."); return; }
    if (request.method === "POST" && header(request, "content-type")?.split(";")[0].trim().toLowerCase() !== "application/json") {
      error(415, "Send the capture as application/json."); return;
    }
    const controller = new AbortController();
    const cancel = () => controller.abort();
    const closed = () => { if (!response.writableEnded) cancel(); };
    request.once("aborted", cancel); response.once("close", closed);
    const timeout = setTimeout(cancel, options.timeoutMs ?? 10000);
    try {
      const endpoint = options.endpoint ?? process.env.HOLOSPEX_IDENTIFICATION_URL;
      if (!endpoint?.trim()) throw new IdentificationError(503, IDENTIFICATION_UNAVAILABLE);
      const url = endpointUrl(endpoint), token = options.token ?? process.env.HOLOSPEX_IDENTIFICATION_TOKEN;
      const input = request.method === "POST" ? parseInput(await readBody(request, controller.signal)) : undefined;
      aborted(controller.signal);
      const upstream = await cancellable((options.fetchImpl ?? fetch)(url, {
        method: request.method, headers: { Accept: "application/json", ...(input ? { "Content-Type": "application/json" } : {}), ...(token ? { Authorization: `Bearer ${token}` } : {}) },
        ...(input ? { body: JSON.stringify(input) } : {}), signal: controller.signal, redirect: "error", credentials: "omit", cache: "no-store",
      }), controller.signal);
      const value = await readResponse(upstream, controller.signal);
      if (!input) { send(200, readyStatus(value)); return; }
      try {
        const result = parseFrameResult(value), frame = input.frame;
        if ((result.source !== "ml_prediction" && result.source !== "propagated_prediction") || result.coordinateSpace !== "original_pixels"
          || result.mediaId !== frame.mediaId || result.frameNumber !== frame.frameNumber || result.timestampMs !== frame.timestampMs
          || result.width !== frame.width || result.height !== frame.height) throw new Error();
        send(200, result);
      } catch { throw new IdentificationError(502, "The identification model returned an invalid or mismatched frame result."); }
    } catch (cause) {
      if (cause instanceof IdentificationError) {
        if (cause.status === 429) response.setHeader("Retry-After", "2");
        error(cause.status, cause.message);
      }
      else error(controller.signal.aborted ? 504 : 503, IDENTIFICATION_UNAVAILABLE);
    } finally {
      clearTimeout(timeout); request.removeListener("aborted", cancel); response.removeListener("close", closed);
      controller.abort();
    }
  };
}

export default createIdentificationHandler();
