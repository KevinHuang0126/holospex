import type { IncomingMessage, ServerResponse } from "node:http";
import { parsePlacementRequest, PlacementValidationError, type PlacementApiErrorBody } from "../shared/placement";
import { suggestPlacement, PlacementServiceError, type PlacementServiceOptions } from "../server/placement";

export const MAX_PLACEMENT_BODY_BYTES = 16 * 1024;
type PlacementHttpRequest = IncomingMessage & { body?: unknown };
export interface PlacementHandlerOptions extends PlacementServiceOptions { now?: () => number; limitPerHour?: number }
export function placementApiKey(environment: { OPENAI_API_KEY?: string; OPEN_AI_KEY?: string }) {
  return environment.OPENAI_API_KEY ?? environment.OPEN_AI_KEY;
}

function singleHeader(request: IncomingMessage, name: string) {
  const value = request.headers[name]; return typeof value === "string" ? value : null;
}
export function hasSamePlacementOrigin(request: IncomingMessage): boolean {
  const origin = singleHeader(request, "origin"), host = singleHeader(request, "host");
  if (!origin || !host || singleHeader(request, "sec-fetch-site") === "cross-site") return false;
  try {
    const parsed = new URL(origin);
    const local = ["localhost", "127.0.0.1", "[::1]"].includes(parsed.hostname);
    return parsed.origin === origin && !parsed.username && !parsed.password && parsed.host.toLowerCase() === host.toLowerCase()
      && (parsed.protocol === "https:" || parsed.protocol === "http:" && local);
  } catch { return false; }
}

async function readBody(request: PlacementHttpRequest): Promise<unknown> {
  const declared = singleHeader(request, "content-length");
  if (declared && (!/^\d+$/.test(declared) || Number(declared) > MAX_PLACEMENT_BODY_BYTES))
    throw new PlacementServiceError(413, "request_too_large", "Placement requests must be smaller than 16 KiB.");
  let text: string;
  if (request.body !== undefined) {
    try { text = Buffer.isBuffer(request.body) ? request.body.toString("utf8") : typeof request.body === "string" ? request.body : JSON.stringify(request.body); }
    catch { throw new PlacementValidationError("Invalid request JSON."); }
  } else {
    const chunks: Buffer[] = []; let length = 0;
    for await (const chunk of request) {
      const bytes = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
      length += bytes.length;
      if (length > MAX_PLACEMENT_BODY_BYTES) throw new PlacementServiceError(413, "request_too_large", "Placement requests must be smaller than 16 KiB.");
      chunks.push(bytes);
    }
    text = Buffer.concat(chunks).toString("utf8");
  }
  if (typeof text !== "string" || Buffer.byteLength(text, "utf8") > MAX_PLACEMENT_BODY_BYTES)
    throw new PlacementServiceError(413, "request_too_large", "Placement requests must be smaller than 16 KiB.");
  try { return JSON.parse(text); } catch { throw new PlacementValidationError("Invalid request JSON."); }
}

/** Demo budget is per warm server instance, not a global account spending limit. */
export function createPlacementHandler(options: PlacementHandlerOptions = {}) {
  let windowStarted = 0, requests = 0, inFlight = false;
  const now = options.now ?? Date.now, limit = Math.max(1, Math.min(20, options.limitPerHour ?? 20));
  return async (request: PlacementHttpRequest, response: ServerResponse): Promise<void> => {
    response.setHeader("Cache-Control", "no-store"); response.setHeader("Content-Type", "application/json; charset=utf-8");
    response.setHeader("X-Content-Type-Options", "nosniff");
    const send = (status: number, value: unknown) => { response.statusCode = status; response.end(JSON.stringify(value)); };
    const error = (status: number, code: string, message: string) => send(status, { error: { code, message } } satisfies PlacementApiErrorBody);
    if (request.method !== "POST") { response.setHeader("Allow", "POST"); error(405, "method_not_allowed", "Use POST to request placement."); return; }
    if (!hasSamePlacementOrigin(request)) { error(403, "origin_rejected", "Open this app on its own secure URL before requesting placement."); return; }
    if (singleHeader(request, "content-type")?.split(";")[0].trim().toLowerCase() !== "application/json") { error(415, "json_required", "Send placement metadata as application/json."); return; }
    let acquired = false;
    try {
      const input = parsePlacementRequest(await readBody(request));
      const apiKey = options.apiKey ?? placementApiKey(process.env), model = options.model ?? process.env.OPENAI_PLACEMENT_MODEL;
      if (!apiKey?.trim()) throw new PlacementServiceError(503, "placement_unconfigured", "AI placement is not configured. Set OPENAI_API_KEY or OPEN_AI_KEY on the server and redeploy.");
      const timestamp = now();
      if (timestamp - windowStarted >= 60 * 60 * 1000 || timestamp < windowStarted) { windowStarted = timestamp; requests = 0; }
      if (inFlight || requests >= limit) {
        response.setHeader("Retry-After", inFlight ? "5" : String(Math.max(1, Math.ceil((windowStarted + 3600000 - timestamp) / 1000))));
        throw new PlacementServiceError(429, "placement_rate_limited", inFlight ? "A placement request is already running. Wait for it to finish." : "This demo has reached its hourly placement budget. Try again later.");
      }
      requests++; inFlight = true; acquired = true;
      send(200, await suggestPlacement(input, { ...options, apiKey, model }));
    } catch (cause) {
      if (cause instanceof PlacementServiceError) error(cause.status, cause.code, cause.message);
      else if (cause instanceof PlacementValidationError) error(400, "invalid_request", cause.message);
      else error(500, "placement_failed", "Placement could not be completed. Try again later.");
    } finally { if (acquired) inFlight = false; }
  };
}

export default createPlacementHandler();
