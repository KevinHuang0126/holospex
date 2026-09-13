import { parseFrameResult, type FrameResult } from "@holospex/contracts";
import type { HudMode } from "./hudControls";
import type { ResultSource } from "./frameInput";

export function parseResultSet(input: unknown): FrameResult[] {
  const values = Array.isArray(input) ? input : [input];
  const results = values.map(parseFrameResult);
  const seen = new Set<string>();
  const frameTimes = new Map<number, number>(), frameNumbers = new Map<number, number>();
  const media = results[0];
  for (const result of results) {
    if (result.mediaId !== media.mediaId || result.width !== media.width || result.height !== media.height) throw new Error("Result file must describe one media ID and original size.");
    const key = `${result.source}:${result.timestampMs}`;
    if (seen.has(key)) throw new Error("Duplicate source/timestamp result.");
    seen.add(key);
    if ((frameTimes.has(result.frameNumber) && frameTimes.get(result.frameNumber) !== result.timestampMs)
      || (frameNumbers.has(result.timestampMs) && frameNumbers.get(result.timestampMs) !== result.frameNumber)) throw new Error("Sources disagree about original frame identity.");
    frameTimes.set(result.frameNumber, result.timestampMs); frameNumbers.set(result.timestampMs, result.frameNumber);
  }
  return results;
}

/** Only match an exported presentation time, allowing sub-ms serialization rounding.
 * Sparse annotation geometry is never held across intervening frames.
 */
export function findVideoResult(results: FrameResult[], mediaId: string, source: ResultSource, timestampMs: number, width: number, height: number): FrameResult | null {
  const candidates = results.filter(result => result.mediaId === mediaId && result.source === source && result.width === width && result.height === height && Math.abs(result.timestampMs - timestampMs) <= 0.5);
  return candidates.length === 1 ? candidates[0] : null;
}

export function visibleStructures(result: FrameResult | null, visible: boolean, mode: HudMode, threshold?: number): { structures: FrameResult["structures"]; warning: string | null } {
  if (!visible || mode === "identify" || mode === "assess") return { structures: [], warning: null };
  if (!result || result.status !== "ok") return { structures: [], warning: `Unable to assess. ${result?.statusReason ?? "No matching result for this frame."}` };
  if (mode === "feedback" && result.source !== "reviewed_annotation" && result.source !== "synthetic_mock") return { structures: [], warning: "Unable to assess. Select reviewed annotations for feedback." };
  if (!result.structures.length) return { structures: [], warning: null };
  const predicted = result.source === "ml_prediction" || result.source === "propagated_prediction";
  if (predicted && (threshold === undefined || !Number.isFinite(threshold) || threshold < 0 || threshold > 1)) return { structures: [], warning: "Unable to assess. Prediction threshold is not configured." };
  const structures = result.structures.filter(item => !predicted || item.confidence! >= threshold!);
  return { structures, warning: structures.length < result.structures.length ? "Unable to assess withheld regions. Low-confidence anatomy is hidden." : structures.some(item => item.visibility === "partial") ? "Partial visibility. Dashed boundaries indicate incomplete views." : null };
}
