import { anatomy, parseFrameResult, type FrameResult } from "@holospex/contracts";
import { matchesFrameIdentity, type DisplayedFrame } from "./selectFrame";

export { anatomy };
export type ResultSource = FrameResult["source"];
export const sourceLabels: Record<ResultSource, string> = {
  synthetic_mock: "Synthetic fixture",
  reviewed_annotation: "Reviewed annotation",
  ml_prediction: "ML prediction",
  propagated_prediction: "Propagated prediction",
};

export type HudFrameInput =
  | { status: "ready"; result: FrameResult; message: null }
  | { status: "missing" | "invalid" | "mismatch" | "unsupported" | "error"; result: null; message: string };

/** Consumer boundary only. No renaming, coordinate conversion, or confidence guessing. */
export function readHudFrame(value: unknown, displayed: DisplayedFrame | null, source: ResultSource): HudFrameInput {
  if (value == null) return { status: "missing", result: null, message: "Unable to assess. No result is available." };
  let result: FrameResult;
  try { result = parseFrameResult(value); }
  catch { return { status: "invalid", result: null, message: "Unable to assess. Result failed contract validation." }; }
  // Check identity before exposing even a technical status from another source/frame.
  if (!displayed || result.source !== source || !matchesFrameIdentity(result, displayed)) {
    return { status: "mismatch", result: null, message: "Unable to assess. No matching result for the displayed frame and source." };
  }
  if (result.status !== "ok") {
    return { status: result.status, result: null, message: `Unable to assess. ${result.statusReason}` };
  }
  return { status: "ready", result, message: null };
}
