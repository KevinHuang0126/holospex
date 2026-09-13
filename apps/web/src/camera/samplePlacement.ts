import { parsePlacementRequest, parsePlacementResponse, TEMPLATE_ID, type PlacementRequest, type PlacementResponse } from "../../../../shared/placement";
import type { ImageOverlayAsset } from "./imageOverlayAsset";
import type { MannequinSamplePlacement } from "./mannequinComposite";

/** Only class names and exact-mask measurements leave the browser. */
export function placementRequestFor(asset: ImageOverlayAsset): PlacementRequest {
  return parsePlacementRequest({ schemaVersion: 1, templateId: TEMPLATE_ID, sampleId: asset.id,
    width: asset.width, height: asset.height, regions: asset.regions ?? [] });
}

/** Place the labeled organ's centroid, rather than the surgical frame's center. */
export function placementFromSuggestion(request: PlacementRequest, suggestion: PlacementResponse,
  mannequinWidth: number, mannequinHeight: number): MannequinSamplePlacement {
  const result = parsePlacementResponse(suggestion, request);
  if (![mannequinWidth, mannequinHeight].every(value => Number.isFinite(value) && value > 0))
    throw new Error("Mannequin dimensions are unavailable.");
  const focus = request.regions.find(region => region.structureId === result.focusStructureId)!;
  const x = result.centerX * mannequinWidth, y = result.centerY * mannequinHeight;
  const fx = focus.centroid.x * request.width, fy = focus.centroid.y * request.height;
  // Keep the entire source frame inside the reference without shifting its
  // anatomical focus. This also handles unusual aspect ratios and edge targets.
  const scale = Math.min(
    result.focusWidthFraction * mannequinWidth / (focus.bounds.width * request.width),
    x / fx, (mannequinWidth - x) / (request.width - fx),
    y / fy, (mannequinHeight - y) / (request.height - fy),
    mannequinWidth * 0.5 / request.width,
  );
  if (!Number.isFinite(scale) || scale <= 0) throw new Error("The suggested anatomy cannot fit this mannequin.");
  return {
    centerX: (x + (request.width / 2 - fx) * scale) / mannequinWidth,
    centerY: (y + (request.height / 2 - fy) * scale) / mannequinHeight,
    widthFraction: request.width * scale / mannequinWidth,
  };
}

export async function requestSamplePlacement(request: PlacementRequest, signal: AbortSignal,
  fetcher: typeof fetch = fetch): Promise<PlacementResponse> {
  const response = await fetcher("/api/placement", {
    method: "POST", headers: { "Content-Type": "application/json" }, credentials: "same-origin",
    body: JSON.stringify(request), signal,
  });
  const raw = await response.text();
  if (raw.length > 16_384) throw new Error("The placement service returned an oversized response.");
  let value: unknown;
  try { value = JSON.parse(raw); }
  catch { throw new Error("The placement API is unavailable on this host. Use the deployed phone app or the development server."); }
  if (!response.ok) {
    const error = value && typeof value === "object" && "error" in value ? value.error : null;
    const message = typeof error === "string" ? error : error && typeof error === "object" && "message" in error ? error.message : null;
    throw new Error(typeof message === "string" && message.length < 500 ? message : "The placement service could not suggest a position. Try again or adjust manually.");
  }
  return parsePlacementResponse(value, request);
}
