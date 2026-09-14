import { parseFrameResult, type FrameResult } from "@holospex/contracts";
import { matchesFrameIdentity, type DisplayedFrame } from "../overlays/selectFrame";

export type LiveFrame = DisplayedFrame;
export type LiveFrameIdentifier = (frame: DisplayedFrame, image: Blob, signal: AbortSignal) => Promise<unknown>;
/** A busy host is distinct from a failed prediction; do not retry at video rate. */
export class IdentificationBusyError extends Error {
  readonly retryAfterMs = 2000;
  constructor() { super("Identification is busy."); this.name = "IdentificationBusyError"; }
}
export interface IdentificationModel {
  status: "ready";
  model: { id: string; version: string };
  minimumConfidence: number;
  dataset: "Endoscapes-Seg50";
}
const ENDPOINT = "/api/identify";
const MAX_RESPONSE_BYTES = 2 * 1024 * 1024;
// A 3 MiB JPEG leaves room for base64 and frame metadata below Vercel's 4.5 MB body limit.
const MAX_CAPTURE_BYTES = 3 * 1024 * 1024;

function abortIfRequested(signal: AbortSignal) {
  if (signal.aborted) throw new DOMException("Live identification was cancelled.", "AbortError");
}

/** Capture identity is immutable for the entire JPEG request/response pair. */
function captureIdentity(frame: DisplayedFrame): DisplayedFrame {
  if (!frame || typeof frame.mediaId !== "string" || !frame.mediaId.trim() || frame.mediaId.length > 160
    || !Number.isSafeInteger(frame.frameNumber) || frame.frameNumber < 0
    || !Number.isFinite(frame.timestampMs) || frame.timestampMs < 0
    || !Number.isSafeInteger(frame.width) || frame.width <= 0 || frame.width > 4096
    || !Number.isSafeInteger(frame.height) || frame.height <= 0 || frame.height > 4096 || frame.width * frame.height > 4_194_304)
    throw new Error("Live identification requires a valid capture-session and frame identity, up to 4096 pixels per side and 4 megapixels.");
  return { mediaId: frame.mediaId, frameNumber: frame.frameNumber, timestampMs: frame.timestampMs, width: frame.width, height: frame.height };
}

/** No timestamp rounding, stale-frame tolerance, or substitution of reference annotations. */
export function validateLiveResult(value: unknown, frame: DisplayedFrame): FrameResult {
  const expected = captureIdentity(frame), result = parseFrameResult(value);
  if (result.source !== "ml_prediction" && result.source !== "propagated_prediction")
    throw new Error("Live identification accepts ML or propagated predictions only.");
  if (!matchesFrameIdentity(result, expected))
    throw new Error("Live identification result does not match the captured session, frame, timestamp and dimensions.");
  return result;
}

async function readJson(response: Response, signal: AbortSignal): Promise<unknown> {
  const declared = response.headers.get("Content-Length");
  if (declared !== null && Number(declared) > MAX_RESPONSE_BYTES) {
    await response.body?.cancel();
    throw new Error("Identification response exceeds the 2 MiB limit.");
  }
  if (!response.body) throw new Error("Identification returned an empty response.");
  const reader = response.body.getReader();
  const cancel = () => { void reader.cancel().catch(() => {}); };
  signal.addEventListener("abort", cancel, { once: true });
  const chunks: Uint8Array[] = [];
  let received = 0;
  try {
    abortIfRequested(signal);
    while (true) {
      const next = await reader.read();
      abortIfRequested(signal);
      if (next.done) break;
      if (next.value.byteLength > MAX_RESPONSE_BYTES - received) throw new Error("Identification response exceeds the 2 MiB limit.");
      chunks.push(next.value); received += next.value.byteLength;
    }
    const bytes = new Uint8Array(received);
    let offset = 0;
    for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength; }
    try { return JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes)); }
    catch { throw new Error("Identification must return a single JSON response."); }
  } finally {
    signal.removeEventListener("abort", cancel);
    try { await reader.cancel(); } catch { /* Preserve validation or cancellation errors. */ }
    reader.releaseLock();
  }
}

async function identificationResponse(response: Response, signal: AbortSignal): Promise<unknown> {
  if (signal.aborted) {
    try { await response.body?.cancel(); } catch { /* Fetch may already have closed its body. */ }
  }
  abortIfRequested(signal);
  if (response.redirected) {
    await response.body?.cancel();
    throw new Error("Identification redirects are not allowed.");
  }
  if (response.status === 429) {
    try { await response.body?.cancel(); } catch { /* The response may already be closed. */ }
    abortIfRequested(signal);
    throw new IdentificationBusyError();
  }
  if (!response.ok) {
    let value: unknown;
    try { value = await readJson(response, signal); } catch { abortIfRequested(signal); }
    const error = value as { status?: unknown; message?: unknown } | undefined;
    const message = error?.status === "error" || error?.status === "unavailable" ? error.message : undefined;
    throw new Error(typeof message === "string" && message.trim() && message.length <= 300 && !/[\u0000-\u001f]/.test(message)
      ? message : `The trained identification model is unavailable (HTTP ${response.status}).`);
  }
  const value = await readJson(response, signal);
  abortIfRequested(signal);
  return value;
}

async function readModelReadiness(signal: AbortSignal): Promise<IdentificationModel> {
  abortIfRequested(signal);
  const response = await fetch(ENDPOINT, { method: "GET", headers: { Accept: "application/json" },
    credentials: "same-origin", redirect: "error", cache: "no-store", referrerPolicy: "no-referrer", signal });
  const value = await identificationResponse(response, signal) as Partial<IdentificationModel> | null;
  const text = (value: unknown): value is string => typeof value === "string" && !!value.trim() && value.length <= 160 && !/[\u0000-\u001f]/.test(value);
  if (!value || value.status !== "ready" || value.dataset !== "Endoscapes-Seg50" || !value.model || !text(value.model.id) || !text(value.model.version)
    || typeof value.minimumConfidence !== "number" || !Number.isFinite(value.minimumConfidence) || value.minimumConfidence < 0 || value.minimumConfidence > 1)
    throw new Error("The identification model has not supplied valid readiness and confidence settings.");
  return { status: "ready", model: { id: value.model.id, version: value.model.version }, minimumConfidence: value.minimumConfidence, dataset: "Endoscapes-Seg50" };
}

/** Checks Person 1's deployed model without sending camera pixels; bounded to five seconds. */
export async function loadIdentificationModel(signal: AbortSignal): Promise<IdentificationModel> {
  abortIfRequested(signal);
  const controller = new AbortController();
  let timer: ReturnType<typeof setTimeout> | undefined;
  let abort: () => void = () => {};
  const cancelled = new Promise<never>((_, reject) => {
    abort = () => {
      controller.abort();
      reject(new DOMException("Model readiness was cancelled.", "AbortError"));
    };
    signal.addEventListener("abort", abort, { once: true });
    timer = setTimeout(() => {
      controller.abort();
      reject(new Error("The identification model readiness check timed out. Try again."));
    }, 5000);
  });
  try { return await Promise.race([readModelReadiness(controller.signal), cancelled]); }
  finally { if (timer !== undefined) clearTimeout(timer); signal.removeEventListener("abort", abort); }
}

/** Sends one captured JPEG to the app's own trained-model endpoint. */
export async function identifyLiveFrame(frame: DisplayedFrame, image: Blob, signal: AbortSignal): Promise<FrameResult> {
  abortIfRequested(signal);
  const expected = captureIdentity(frame);
  if (image.type !== "image/jpeg" || image.size <= 0) throw new Error("Live identification requires a nonempty JPEG capture.");
  if (image.size > MAX_CAPTURE_BYTES) throw new Error("The JPEG capture exceeds the 3 MiB identification limit.");
  const bytes = new Uint8Array(await image.arrayBuffer());
  abortIfRequested(signal);
  if (bytes[0] !== 255 || bytes[1] !== 216 || bytes[2] !== 255) throw new Error("Live identification requires a valid JPEG capture.");
  const chunks: string[] = [];
  for (let offset = 0; offset < bytes.length; offset += 16_384) chunks.push(String.fromCharCode(...bytes.subarray(offset, offset + 16_384)));
  const imageBase64 = btoa(chunks.join(""));
  abortIfRequested(signal);
  const response = await fetch(ENDPOINT, {
    method: "POST", headers: { Accept: "application/json", "Content-Type": "application/json" }, body: JSON.stringify({ frame: expected, imageBase64 }),
    credentials: "same-origin", redirect: "error", cache: "no-store", referrerPolicy: "no-referrer", signal,
  });
  return validateLiveResult(await identificationResponse(response, signal), expected);
}
