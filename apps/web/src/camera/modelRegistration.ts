import { anatomy, type AnatomyId } from "@holospex/contracts";
import posit from "js-aruco2/src/posit1.js";
import type { HudAnchor } from "../overlays/drawHud";
const { POS } = posit;

export interface ModelRegistration {
  modelId: string;
  provenance: "measured_model_locations" | "synthetic_mock";
  dictionary: "ARUCO_MIP_36h12";
  markerId: number;
  markerSizeMm: number;
  calibration: { width: number; height: number; fx: number; fy: number; cx: number; cy: number };
  anchors: { id: string; structureId: AnatomyId; positionMm: [number, number, number] }[];
}

export function parseModelRegistration(input: unknown): ModelRegistration {
  if (!input || typeof input !== "object") throw new Error("Model registration must be an object.");
  const value = input as ModelRegistration;
  if (typeof value.modelId !== "string" || !value.modelId.trim() || !["measured_model_locations", "synthetic_mock"].includes(value.provenance)
    || value.dictionary !== "ARUCO_MIP_36h12" || !Number.isInteger(value.markerId) || value.markerId < 0 || value.markerId >= 250
    || !Number.isFinite(value.markerSizeMm) || value.markerSizeMm <= 0) throw new Error("Invalid model/marker identity or size.");
  const c = value.calibration;
  if (!c || ![c.width, c.height, c.fx, c.fy].every(v => Number.isFinite(v) && v > 0)
    || !Number.isInteger(c.width) || !Number.isInteger(c.height)
    || !Number.isFinite(c.cx) || !Number.isFinite(c.cy) || c.cx < 0 || c.cy < 0 || c.cx > c.width || c.cy > c.height) throw new Error("Supply measured camera intrinsics and their image size.");
  if (!Array.isArray(value.anchors) || !value.anchors.length || value.anchors.some(a => !a || typeof a.id !== "string" || !a.id.trim() || !Object.hasOwn(anatomy, a.structureId)
    || !Array.isArray(a.positionMm) || a.positionMm.length !== 3 || !a.positionMm.every(Number.isFinite))) throw new Error("Supply named anatomical anchors in marker-relative millimeters.");
  if (new Set(value.anchors.map(a => a.id)).size !== value.anchors.length) throw new Error("Duplicate model anchor IDs.");
  return value;
}

export function projectPoint(position: number[], rotation: number[][], translation: number[], c: ModelRegistration["calibration"]): [number, number] | null {
  const point = rotation.map((row, i) => row.reduce((sum, v, j) => sum + v * position[j], translation[i]));
  if (point.length !== 3 || !point.every(Number.isFinite) || point[2] <= 0) return null;
  return [c.cx + c.fx * point[0] / point[2], c.cy - c.fy * point[1] / point[2]];
}

export interface ModelPose {
  rotation: number[][];
  translation: number[];
  calibration: ModelRegistration["calibration"];
  reprojectionError: number;
}

/** Marker/model positions never enter FrameResult. Camera lens distortion is not modeled. */
export function estimateModelPose(config: ModelRegistration, corners: { x: number; y: number }[], width: number, height: number): ModelPose | null {
  if (![width, height].every(v => Number.isFinite(v) && v > 0) || corners.length !== 4 || corners.some(p => !Number.isFinite(p.x) || !Number.isFinite(p.y))
    || Math.abs(width / height - config.calibration.width / config.calibration.height) > 0.01) return null;
  const scale = width / config.calibration.width;
  const c = { ...config.calibration, width, height, fx: config.calibration.fx * scale, fy: config.calibration.fy * scale, cx: config.calibration.cx * scale, cy: config.calibration.cy * scale };
  // POSIT stops when its rounded pixel error reaches zero. Solve the same
  // camera rays on a fixed, larger focal plane so downsampling cannot change
  // its stopping precision (or flip which planar pose wins).
  const solverFocal = 10000;
  const centered = corners.map(p => ({ x: (p.x - c.cx) / c.fx * solverFocal, y: (c.cy - p.y) / c.fy * solverFocal }));
  const pose = new POS.Posit(config.markerSizeMm, solverFocal).pose(centered);
  const half = config.markerSizeMm / 2;
  const square = [[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]];
  const candidates = [
    { error: pose.bestError, r: pose.bestRotation, t: pose.bestTranslation },
    { error: pose.alternativeError, r: pose.alternativeRotation, t: pose.alternativeTranslation },
  ].filter(p => Number.isFinite(p.error) && p.error >= 0 && p.r.length === 3 && p.r.every(row => row.length === 3) && p.t.length === 3)
    .map(p => ({ ...p, reprojection: square.reduce((max, point, i) => {
      const projected = projectPoint(point, p.r, p.t, c);
      return Math.max(max, projected ? Math.hypot(projected[0] - corners[i].x, projected[1] - corners[i].y) : Infinity);
    }, 0) })).sort((a, b) => a.reprojection - b.reprojection);
  const best = candidates[0];
  if (!best || best.reprojection > 4) return null;
  return { rotation: best.r, translation: best.t, calibration: c, reprojectionError: best.reprojection };
}

export function projectModelAnchors(config: ModelRegistration, pose: ModelPose): HudAnchor[] {
  const { calibration: c, rotation, translation } = pose;
  const anchors: HudAnchor[] = [];
  for (const anchor of config.anchors) {
    const point = projectPoint(anchor.positionMm, rotation, translation, c);
    if (!point || point[0] < 0 || point[1] < 0 || point[0] > c.width || point[1] > c.height) continue;
    anchors.push({ id: anchor.id, structureId: anchor.structureId, x: point[0], y: point[1] });
  }
  return anchors;
}
