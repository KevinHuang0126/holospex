import type { LiveFrameIdentifier } from "../input/liveIdentification";
import type { LiveFramePipelineOptions } from "../input/liveFramePipeline";
import type { DisplayedFrame } from "../overlays/selectFrame";
import { createCameraCaptureSession } from "./cameraCaptureSession";

export interface UploadedImage<Image> { image: Image; width: number; height: number }

/** Local selection and explicit still identification have separate lifetimes. */
export function createUploadedImageSession<Image>(options: {
  decode(file: File, signal: AbortSignal): Promise<UploadedImage<Image>>;
  copyImage(image: Image): Image;
  encode(image: Image): Promise<Blob>;
  onChange(): void;
  now?: () => number;
  scheduleTimeout?: LiveFramePipelineOptions["scheduleTimeout"];
  cancelTimeout?: LiveFramePipelineOptions["cancelTimeout"];
}) {
  const now = options.now ?? (() => performance.now());
  let disposed = false, generation = 0;
  let controller: AbortController | null = null;
  let decoded: { image: Image; frame: DisplayedFrame } | null = null;
  let phase: "empty" | "loading" | "ready" | "error" = "empty", error: string | null = null;
  const notify = () => { if (!disposed) options.onChange(); };
  const capture = createCameraCaptureSession({ ...options, now, onChange: notify });
  return {
    get image() { return capture.snapshot ?? decoded; },
    get result() { return capture.snapshot?.result ?? null; },
    get status() { return phase !== "ready" ? phase : capture.status === "preview" ? "ready" : capture.status; },
    get message() { return error ?? capture.message?.replace(/camera frame/gi, "image") ?? null; },
    get busy() { return capture.busy; },
    setFile(file: File | null) {
      if (disposed) return;
      const epoch = ++generation;
      controller?.abort(); controller = null; decoded = null; error = null;
      phase = file ? "loading" : "empty";
      capture.retake(); notify();
      if (!file) return;
      const pending = new AbortController(); controller = pending;
      void (async () => {
        try {
          const result = await options.decode(file, pending.signal);
          if (disposed || epoch !== generation || pending.signal.aborted) return;
          decoded = { image: result.image, frame: Object.freeze({
            mediaId: `image:${globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(36).slice(2)}`}`,
            frameNumber: 0, timestampMs: 0, width: result.width, height: result.height,
          }) };
          phase = "ready";
        } catch (cause) {
          if (disposed || epoch !== generation || pending.signal.aborted) return;
          phase = "error";
          error = cause instanceof Error ? cause.message : "This image could not be decoded. Choose a JPEG, PNG or WebP image.";
        } finally { if (!disposed && epoch === generation) { controller = null; notify(); } }
      })();
    },
    setIdentifier(identifier: LiveFrameIdentifier | undefined) { capture.setIdentifier(identifier); },
    clearResult() { if (!disposed) capture.retake(); },
    identify() {
      if (disposed || phase !== "ready" || !decoded || capture.status === "identifying") return false;
      // Retake clears a previous result/cooldown, preserving any still-running retired slot.
      capture.retake();
      return capture.capture({ ...decoded, capturedAt: now() });
    },
    dispose() {
      if (disposed) return;
      disposed = true; generation++; controller?.abort(); controller = null;
      capture.dispose(); decoded = null;
    },
  };
}
