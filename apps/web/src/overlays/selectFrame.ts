import type { FrameResult } from "@holospex/contracts";

export interface DisplayedFrame {
  mediaId: string;
  frameNumber: number;
  timestampMs: number;
  width: number;
  height: number;
}

/** An asynchronous result belongs to one exact source frame. On seek, clip
 * change, unavailable output, or coordinate mismatch, remove the old overlay.
 */
export function matchesDisplayedFrame(result: FrameResult | null, displayed: DisplayedFrame): boolean {
  return result?.status === "ok" && matchesFrameIdentity(result, displayed);
}

/** Availability does not change which original frame a record belongs to. */
export function matchesFrameIdentity(result: FrameResult | null, displayed: DisplayedFrame): boolean {
  return result !== null
    && result.coordinateSpace === "original_pixels"
    && result.mediaId === displayed.mediaId
    && result.frameNumber === displayed.frameNumber
    && result.timestampMs === displayed.timestampMs
    && result.width === displayed.width
    && result.height === displayed.height;
}
