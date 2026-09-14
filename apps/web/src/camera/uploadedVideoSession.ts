import type { FrameResult } from "@holospex/contracts";
import { createLiveFramePipeline } from "../input/liveFramePipeline";
import type { LiveFrameIdentifier } from "../input/liveIdentification";
import type { DisplayedFrame } from "../overlays/selectFrame";

export interface UploadedVideoCapture<Image> { image: Image; frame: DisplayedFrame; capturedAt: number }

/** Bound phone uploads without changing the image aspect ratio or polygon coordinate space. */
export function videoCaptureSize(width: number, height: number) {
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) return null;
  const scale = Math.min(1, 1280 / width, 720 / height);
  return { width: Math.max(1, Math.round(width * scale)), height: Math.max(1, Math.round(height * scale)) };
}

/** Own the exact image paired with each response and invalidate it across playback discontinuities. */
export function createUploadedVideoSession<Image>(options: {
  onResult(capture: UploadedVideoCapture<Image>, result: FrameResult): void;
  onUnavailable(message: string): void;
}) {
  type Pipeline = ReturnType<typeof createLiveFramePipeline>;
  let identifier: LiveFrameIdentifier | undefined;
  let pipeline: Pipeline | null = null, captured: UploadedVideoCapture<Image> | null = null;
  let retired: Pipeline[] = [], disposed = false;
  const busy = () => {
    retired = retired.filter(previous => previous.busy);
    return !!pipeline?.busy || retired.length > 0;
  };
  const invalidate = () => {
    if (pipeline) {
      pipeline.dispose();
      // An aborted encoder or provider may still be running. Keep its slot until it settles.
      if (pipeline.busy) retired.push(pipeline);
    }
    pipeline = null; captured = null;
  };
  return {
    get busy() { return busy(); },
    setIdentifier(next: LiveFrameIdentifier | undefined) {
      if (identifier !== next) { invalidate(); identifier = next; }
    },
    invalidate,
    submit(capture: UploadedVideoCapture<Image>, encode: () => Promise<Blob>) {
      if (disposed || !identifier || busy()) return false;
      if (!pipeline) pipeline = createLiveFramePipeline({
        identify: identifier,
        onResult(frame, result) {
          if (!disposed && captured?.frame.mediaId === frame.mediaId && captured.frame.frameNumber === frame.frameNumber)
            options.onResult(captured, result);
        },
        onUnavailable: message => { if (!disposed) options.onUnavailable(message.replace(/camera/gi, "video")); },
      });
      captured = { ...capture, frame: { ...capture.frame } };
      return pipeline.submit(captured.frame, encode, captured.capturedAt);
    },
    dispose() { disposed = true; invalidate(); },
  };
}
