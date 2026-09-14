import type { FrameResult } from "@holospex/contracts";
import { matchesFrameIdentity } from "../overlays/selectFrame";
import type { UploadedVideoCapture } from "./uploadedVideoSession";

export type InspectedVideoFrame<Image> = UploadedVideoCapture<Image> & { result?: FrameResult };

/** Freeze the pixels actually shown, even while the source video has advanced.
 * Preview canvases are mutable, so inspection always owns a separate image.
 */
export function freezeVideoFrame<Image>(displayed: UploadedVideoCapture<Image>,
  identified: InspectedVideoFrame<Image> | null, copyImage: (image: Image) => Image): InspectedVideoFrame<Image> {
  const result = identified?.result;
  return { ...displayed, image: copyImage(displayed.image), frame: { ...displayed.frame },
    result: result && identified.image === displayed.image && matchesFrameIdentity(result, displayed.frame) ? result : undefined };
}
