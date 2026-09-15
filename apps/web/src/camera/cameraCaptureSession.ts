import type { FrameResult } from "@holospex/contracts";
import { createLiveFramePipeline, type LiveFramePipelineOptions } from "../input/liveFramePipeline";
import type { LiveFrameIdentifier } from "../input/liveIdentification";
import type { DisplayedFrame } from "../overlays/selectFrame";

export interface CameraCapture<Image> { image: Image; frame: DisplayedFrame; capturedAt: number }
export interface CameraSnapshot<Image> extends CameraCapture<Image> { result?: FrameResult }
export type CameraCaptureStatus = "preview" | "identifying" | "identified" | "error";
export const MAX_CAMERA_PREVIEW_AGE_MS = 500;

/** Explicit capture owns its pixels until retake. A validated result remains on
 * that still; the live transport deadline does not expire an accepted photo.
 */
export function createCameraCaptureSession<Image>(options: {
  copyImage(image: Image): Image;
  encode(image: Image): Promise<Blob>;
  onChange(): void;
  now?: () => number;
  scheduleTimeout?: LiveFramePipelineOptions["scheduleTimeout"];
  cancelTimeout?: LiveFramePipelineOptions["cancelTimeout"];
}) {
  type Pipeline = ReturnType<typeof createLiveFramePipeline>;
  const now = options.now ?? (() => performance.now());
  let identifier: LiveFrameIdentifier | undefined;
  let pipeline: Pipeline | null = null, retired: Pipeline[] = [];
  let snapshot: CameraSnapshot<Image> | null = null, status: CameraCaptureStatus = "preview", message: string | null = null;
  let disposed = false, generation = 0;
  let notified: { snapshot: CameraSnapshot<Image> | null; status: CameraCaptureStatus; message: string | null; busy: boolean } | null = null;
  const busy = () => {
    retired = retired.filter(previous => previous.busy);
    return !!pipeline?.busy || retired.length > 0;
  };
  const notify = () => {
    if (disposed) return;
    const value = { snapshot, status, message, busy: busy() };
    if (!notified || value.snapshot !== notified.snapshot || value.status !== notified.status
      || value.message !== notified.message || value.busy !== notified.busy) {
      notified = value; options.onChange();
    }
  };
  // The pipeline clears its active slot after the encoder/provider promise has
  // settled. Notify after its continuation, including disposed providers which
  // ignored abort and therefore never invoke onResult/onUnavailable.
  const settled = () => { queueMicrotask(() => queueMicrotask(notify)); };
  const invalidate = () => {
    generation++;
    if (pipeline) {
      pipeline.dispose();
      if (pipeline.busy) retired.push(pipeline);
    }
    pipeline = null; snapshot = null; status = "preview"; message = null;
  };
  const fail = (reason: string) => { status = "error"; message = reason; notify(); };

  return {
    get snapshot() { return snapshot; },
    get status() { return status; },
    get message() { return message; },
    get busy() { return busy(); },
    setIdentifier(next: LiveFrameIdentifier | undefined) {
      if (disposed || identifier === next) return;
      invalidate(); identifier = next; notify();
    },
    capture(preview: CameraCapture<Image>): boolean {
      if (disposed || snapshot) return false;
      if (busy()) { fail("The previous capture is still finishing. Try again shortly."); return false; }
      if (!identifier) { fail("The anatomy identification model is not ready yet."); return false; }
      const capturedAt = preview.capturedAt, age = now() - capturedAt;
      if (!Number.isFinite(capturedAt) || !Number.isFinite(age) || age < 0 || age >= MAX_CAMERA_PREVIEW_AGE_MS) {
        fail("The camera preview is outdated. Wait for a fresh frame and capture again."); return false;
      }
      let owned: CameraSnapshot<Image>;
      try {
        const frame = Object.freeze({ ...preview.frame });
        owned = Object.freeze({ image: options.copyImage(preview.image), frame, capturedAt });
      } catch {
        fail("The camera frame could not be captured. Try again."); return false;
      }
      snapshot = owned; status = "identifying"; message = null;
      const epoch = generation, identify = identifier;
      pipeline = createLiveFramePipeline({
        now, scheduleTimeout: options.scheduleTimeout, cancelTimeout: options.cancelTimeout,
        identify: async (...args) => { try { return await identify(...args); } finally { settled(); } },
        onResult(_frame, result) {
          if (disposed || epoch !== generation || snapshot !== owned) return;
          snapshot = Object.freeze({ ...owned, result }); status = "identified"; message = null; notify();
        },
        onUnavailable(reason) {
          if (disposed || epoch !== generation || snapshot !== owned) return;
          fail(reason);
        },
      });
      const accepted = pipeline.submit(owned.frame, async () => {
        try { return await options.encode(owned.image); } finally { settled(); }
      }, capturedAt);
      notify();
      return accepted;
    },
    retake() { if (!disposed) { invalidate(); notify(); } },
    dispose() { if (!disposed) { disposed = true; invalidate(); } },
  };
}
