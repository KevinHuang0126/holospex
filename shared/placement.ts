/** Browser-safe metadata contract. No patient imagery or camera frames cross this API. */
export const TEMPLATE_ID = "training-mannequin-v1" as const;
export const TEMPLATE_WIDTH = 894;
export const TEMPLATE_HEIGHT = 569;
export const TARGET_REGIONS = ["head", "right_upper_abdomen", "left_upper_abdomen", "chest", "lower_abdomen"] as const;
export type TargetRegion = typeof TARGET_REGIONS[number];
export interface TemplateLandmark {
  centerX: number; centerY: number;
  minCenterX: number; maxCenterX: number; minCenterY: number; maxCenterY: number;
  minWidthFraction: number; maxWidthFraction: number;
}
/** Authored image landmarks, not measured patient coordinates. Anterior body, head
 * left and feet right: the patient's right is toward the BOTTOM of this image. */
export const TEMPLATE_LANDMARKS: Record<TargetRegion, TemplateLandmark> = {
  head: { centerX: 0.085, centerY: 0.515, minCenterX: 0.045, maxCenterX: 0.135, minCenterY: 0.44, maxCenterY: 0.59, minWidthFraction: 0.05, maxWidthFraction: 0.12 },
  right_upper_abdomen: { centerX: 0.345, centerY: 0.545, minCenterX: 0.32, maxCenterX: 0.37, minCenterY: 0.515, maxCenterY: 0.565, minWidthFraction: 0.04, maxWidthFraction: 0.075 },
  left_upper_abdomen: { centerX: 0.345, centerY: 0.415, minCenterX: 0.32, maxCenterX: 0.37, minCenterY: 0.385, maxCenterY: 0.455, minWidthFraction: 0.04, maxWidthFraction: 0.075 },
  chest: { centerX: 0.255, centerY: 0.485, minCenterX: 0.21, maxCenterX: 0.29, minCenterY: 0.42, maxCenterY: 0.55, minWidthFraction: 0.04, maxWidthFraction: 0.14 },
  lower_abdomen: { centerX: 0.425, centerY: 0.485, minCenterX: 0.38, maxCenterX: 0.47, minCenterY: 0.42, maxCenterY: 0.55, minWidthFraction: 0.035, maxWidthFraction: 0.14 },
};
export const STRUCTURE_TARGETS: Readonly<Record<string, TargetRegion>> = {
  gallbladder: "right_upper_abdomen", cystic_duct: "right_upper_abdomen", cystic_artery: "right_upper_abdomen",
  cystic_plate: "right_upper_abdomen", hepatocystic_triangle_dissection: "right_upper_abdomen", liver: "right_upper_abdomen",
  brain: "head", brain_mri: "head", cerebrum: "head", cerebellum: "head",
  heart: "chest", lung: "chest", lungs: "chest", left_lung: "chest", right_lung: "chest",
  stomach: "left_upper_abdomen", spleen: "left_upper_abdomen", bladder: "lower_abdomen", urinary_bladder: "lower_abdomen", small_intestine: "lower_abdomen",
};
export interface PlacementRequest {
  schemaVersion: 1; sampleId: string; templateId: typeof TEMPLATE_ID; width: number; height: number;
  regions: { structureId: string; bounds: { x: number; y: number; width: number; height: number }; centroid: { x: number; y: number }; pixelCount: number }[];
}
export interface PlacementResponse {
  schemaVersion: 1; sampleId: string; templateId: typeof TEMPLATE_ID; source: "ai_suggested";
  focusStructureId: string; targetRegion: TargetRegion; centerX: number; centerY: number; focusWidthFraction: number; reason: string;
}
export interface PlacementApiErrorBody { error: { code: string; message: string } }
export class PlacementValidationError extends Error {}
function fail(message: string): never { throw new PlacementValidationError(message); }
function record(value: unknown, expected: readonly string[]): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) fail("Expected a placement object.");
  const object = value as Record<string, unknown>, keys = Object.keys(object);
  if (keys.length !== expected.length || expected.some(key => !Object.hasOwn(object, key))) fail("Unexpected placement fields.");
  return object;
}
function finite(value: unknown, min: number, max: number): value is number { return typeof value === "number" && Number.isFinite(value) && value >= min && value <= max; }
function identifier(value: unknown): value is string { return typeof value === "string" && /^[a-z][a-z0-9_]{0,63}$/.test(value); }
export function parsePlacementRequest(value: unknown): PlacementRequest {
  const request = record(value, ["schemaVersion", "sampleId", "templateId", "width", "height", "regions"]);
  if (request.schemaVersion !== 1 || request.templateId !== TEMPLATE_ID
    || typeof request.sampleId !== "string" || !/^[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}$/.test(request.sampleId)) fail("Invalid placement version or sample identity.");
  if (!finite(request.width, 1, 8192) || !Number.isInteger(request.width) || !finite(request.height, 1, 8192) || !Number.isInteger(request.height)
    || request.width * request.height > 16_000_000) fail("Invalid source image dimensions.");
  if (!Array.isArray(request.regions) || !request.regions.length || request.regions.length > 32) fail("Supply one to 32 labeled regions.");
  const seen = new Set<string>();
  const regions = request.regions.map(input => {
    const region = record(input, ["structureId", "bounds", "centroid", "pixelCount"]);
    if (!identifier(region.structureId) || seen.has(region.structureId)) fail("Region IDs must be unique anatomy identifiers.");
    seen.add(region.structureId);
    const bounds = record(region.bounds, ["x", "y", "width", "height"]), centroid = record(region.centroid, ["x", "y"]);
    if (!finite(bounds.x, 0, 1) || !finite(bounds.y, 0, 1) || !finite(bounds.width, Number.EPSILON, 1) || !finite(bounds.height, Number.EPSILON, 1)
      || bounds.x + bounds.width > 1 + 1e-9 || bounds.y + bounds.height > 1 + 1e-9) fail("Region bounds must remain inside the source image.");
    if (!finite(centroid.x, bounds.x, Math.min(1, bounds.x + bounds.width)) || !finite(centroid.y, bounds.y, Math.min(1, bounds.y + bounds.height))) fail("Region centroids must lie within their bounds.");
    if (!finite(region.pixelCount, 1, request.width as number * (request.height as number)) || !Number.isInteger(region.pixelCount)
      || region.pixelCount > Math.ceil(bounds.width * bounds.height * (request.width as number) * (request.height as number) + 1e-6)) fail("Invalid region pixel count.");
    return { structureId: region.structureId, bounds: { x: bounds.x, y: bounds.y, width: bounds.width, height: bounds.height },
      centroid: { x: centroid.x, y: centroid.y }, pixelCount: region.pixelCount };
  });
  return { schemaVersion: 1, sampleId: request.sampleId, templateId: TEMPLATE_ID, width: request.width, height: request.height, regions };
}

/** A result is accepted only for the same source and an authored, compatible body region. */
export function parsePlacementResponse(value: unknown, sourceRequest: PlacementRequest): PlacementResponse {
  const request = parsePlacementRequest(sourceRequest);
  const response = record(value, ["schemaVersion", "sampleId", "templateId", "source", "focusStructureId", "targetRegion", "centerX", "centerY", "focusWidthFraction", "reason"]);
  if (response.schemaVersion !== 1 || response.sampleId !== request.sampleId || response.templateId !== request.templateId || response.source !== "ai_suggested") fail("Placement result does not match the selected sample and template.");
  if (!identifier(response.focusStructureId) || !request.regions.some(region => region.structureId === response.focusStructureId)
    || !Object.hasOwn(STRUCTURE_TARGETS, response.focusStructureId) || STRUCTURE_TARGETS[response.focusStructureId] !== response.targetRegion) fail("Placement focus is unsupported or contradicts the supplied anatomy.");
  const targets = new Set<TargetRegion>();
  for (const region of request.regions) {
    if (region.structureId === "tool" || region.structureId === "background") continue;
    if (!Object.hasOwn(STRUCTURE_TARGETS, region.structureId)) fail("The sample contains unsupported anatomy labels.");
    targets.add(STRUCTURE_TARGETS[region.structureId]);
  }
  if (targets.size !== 1) fail("The sample spans ambiguous body regions.");
  const targetRegion = response.targetRegion as TargetRegion, landmark = TEMPLATE_LANDMARKS[targetRegion];
  if (!finite(response.centerX, landmark.minCenterX, landmark.maxCenterX) || !finite(response.centerY, landmark.minCenterY, landmark.maxCenterY)
    || !finite(response.focusWidthFraction, landmark.minWidthFraction, landmark.maxWidthFraction)) fail("Placement lies outside the authored template region.");
  if (typeof response.reason !== "string" || !response.reason.trim() || response.reason.length > 240 || /[\u0000-\u001f]/.test(response.reason)) fail("Invalid placement explanation.");
  return { schemaVersion: 1, sampleId: request.sampleId, templateId: TEMPLATE_ID, source: "ai_suggested", focusStructureId: response.focusStructureId,
    targetRegion, centerX: response.centerX, centerY: response.centerY, focusWidthFraction: response.focusWidthFraction, reason: response.reason.trim() };
}
