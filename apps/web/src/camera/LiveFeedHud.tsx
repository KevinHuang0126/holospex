import { useEffect, useLayoutEffect, useRef, useState } from "react";
import type { LiveFrameIdentifier } from "../input/liveIdentification";
import { drawHud, type HudAppearance } from "../overlays/drawHud";
import { sourceLabels } from "../overlays/frameInput";
import type { DisplayedFrame } from "../overlays/selectFrame";
import { visibleStructures } from "../overlays/videoResults";
import { createCameraCaptureSession, MAX_CAMERA_PREVIEW_AGE_MS, type CameraCapture } from "./cameraCaptureSession";
import { videoCaptureSize } from "./uploadedVideoSession";

export interface LiveFeedHudProps {
  stream: MediaStream | null;
  interrupted: boolean;
  visible: boolean;
  identify?: LiveFrameIdentifier;
  minimumConfidence?: number;
  appearance?: HudAppearance;
  onCanvas?: (canvas: HTMLCanvasElement | null) => void;
  onDisplayedFrame?: (frame: DisplayedFrame | null) => void;
}
const FEED_STALL_MS = MAX_CAMERA_PREVIEW_AGE_MS;

/** Preview stays local. Only an explicit capture sends a frame to the model.
 * The owned still and its result stay together until retake or a session reset.
 */
export function LiveFeedHud(props: LiveFeedHudProps) {
  const video = useRef<HTMLVideoElement>(null), canvas = useRef<HTMLCanvasElement>(null);
  const current = useRef(props); current.current = props;
  const latest = useRef<CameraCapture<HTMLCanvasElement> | null>(null);
  const session = useRef<ReturnType<typeof createCameraCaptureSession<HTMLCanvasElement>> | null>(null);
  const status = useRef("Feed stopped.");
  const [ui, setUi] = useState({ notice: status.current, canCapture: false, frozen: false, identifying: false });
  const paint = useRef(() => {});

  paint.current = () => {
    if (!canvas.current) return;
    const active = current.current, controller = session.current;
    const snapshot = active.stream && !active.interrupted ? controller?.snapshot : null;
    const display = active.stream && !active.interrupted ? snapshot ?? latest.current : null;
    const result = snapshot?.result;
    const overlay = visibleStructures(result ?? null, active.visible, "learn", active.minimumConfidence);
    const warning = overlay.warning?.replace(/Unable to assess/g, "Unable to identify") ?? null;
    const identifying = !!snapshot && controller?.status === "identifying";
    const failed = controller?.status === "error";
    const message = !active.stream ? "Feed stopped."
      : active.interrupted ? "Feed interrupted. Restart the camera if it does not recover."
      : !display ? status.current
      : snapshot ? identifying ? "Identifying captured frame…"
        : failed ? `${controller?.message ?? "Unable to identify this frame."} Retake to try another frame.`
        : !active.visible ? "Captured frame. Anatomy hidden. Retake to return to the camera."
        : warning ?? (overlay.structures.length ? "Anatomy identified in the captured frame. Retake to return to the camera." : "No supported anatomy identified in this frame. Retake to try another frame.")
      : !active.identify ? "Live preview. Connecting to the identification model; capture a frame once connected."
      : !active.visible ? "Live preview. Show overlays to capture and identify anatomy."
      : controller?.busy ? "Live preview. Waiting for the previous identification to stop."
      : controller?.message ?? "Live preview. Tap Capture & identify to freeze one frame.";
    const canCapture = !!(active.stream && !active.interrupted && active.visible && active.identify
      && latest.current && performance.now() - latest.current.capturedAt < FEED_STALL_MS
      && !document.hidden && controller && !controller.busy && !snapshot);
    setUi(previous => previous.notice === message && previous.canCapture === canCapture
      && previous.frozen === !!snapshot && previous.identifying === identifying
      ? previous : { notice: message, canCapture, frozen: !!snapshot, identifying });
    drawHud(canvas.current, display?.image ?? null, {
      width: display?.frame.width ?? 1280, height: display?.frame.height ?? 720,
      source: result?.source,
      sourceLabel: result ? sourceLabels[result.source] : snapshot ? "Captured camera frame" : "Live camera preview",
      statusLabel: display ? `Frame ${display.frame.frameNumber} · ${display.frame.timestampMs.toFixed(1)} ms${snapshot ? " · Captured still" : ""}` : "No live frame",
      structures: result ? overlay.structures : [], labelSlots: 6,
      appearance: active.appearance,
      warning: !display || active.interrupted || failed ? message : result ? warning : null,
    });
    active.onDisplayedFrame?.(display?.frame ?? null);
  };

  useLayoutEffect(() => {
    session.current = createCameraCaptureSession<HTMLCanvasElement>({
      copyImage(raw) {
        const image = document.createElement("canvas"); image.width = raw.width; image.height = raw.height;
        const pixels = image.getContext("2d");
        if (!pixels) throw new Error("Frame capture is unavailable.");
        pixels.drawImage(raw, 0, 0);
        return image;
      },
      encode: image => new Promise<Blob>((resolve, reject) => {
        image.toBlob(blob => blob ? resolve(blob) : reject(new Error("Frame encoding failed.")), "image/jpeg", 0.9);
      }),
      onChange: () => paint.current(),
    });
    return () => { session.current?.dispose(); session.current = null; };
  }, []);
  useLayoutEffect(() => { session.current?.setIdentifier(props.identify); paint.current(); }, [props.identify]);
  useLayoutEffect(() => { paint.current(); }, [props.visible, props.minimumConfidence, props.appearance]);
  useEffect(() => {
    props.onCanvas?.(canvas.current);
    const resize = new ResizeObserver(() => paint.current()); resize.observe(canvas.current!);
    return () => { resize.disconnect(); current.current.onCanvas?.(null); current.current.onDisplayedFrame?.(null); };
  }, []);

  useLayoutEffect(() => {
    const element = video.current!;
    let disposed = false, callback = 0, fallback = 0;
    let mediaId = "", frameNumber = 0, nativeSize = "", previousTime = -1;
    let lastFrameAt = performance.now(), stalled = false, wasBusy = false;
    const raw = document.createElement("canvas");
    const newIdentity = () => {
      mediaId = `live:${globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(36).slice(2)}`}`;
      frameNumber = 0;
    };
    const reset = (message: string) => {
      latest.current = null; status.current = message;
      newIdentity(); session.current?.retake(); paint.current();
    };
    const previewFrame = (timestampMs: number) => {
      if (disposed || document.hidden || props.interrupted || !props.stream) return;
      if (element.readyState < 2 || !element.videoWidth || !element.videoHeight || !Number.isFinite(timestampMs) || timestampMs < 0) return;
      try {
        const dimensions = videoCaptureSize(element.videoWidth, element.videoHeight);
        if (!dimensions) return;
        const now = performance.now(), { width, height } = dimensions;
        const size = `${element.videoWidth}x${element.videoHeight}`;
        if (stalled || size !== nativeSize || timestampMs < previousTime) {
          newIdentity(); nativeSize = size; stalled = false;
        }
        previousTime = timestampMs; lastFrameAt = now;
        // Monitor the camera while a still is displayed, without replacing its pixels or sending requests.
        if (session.current?.snapshot) return;
        if (raw.width !== width || raw.height !== height) { raw.width = width; raw.height = height; }
        const context = raw.getContext("2d");
        if (!context) throw new Error("Video rendering is unavailable.");
        context.drawImage(element, 0, 0, width, height);
        const frame: DisplayedFrame = { mediaId, frameNumber: frameNumber++, timestampMs, width, height };
        latest.current = { image: raw, frame, capturedAt: now };
        paint.current();
      } catch {
        reset("The camera frame could not be displayed. Stop and restart the camera.");
      }
    };
    const next = (_: number, metadata: VideoFrameCallbackMetadata) => {
      if (disposed) return;
      previewFrame(metadata.mediaTime * 1000);
      callback = element.requestVideoFrameCallback(next);
    };
    const preview = () => {
      if (disposed) return;
      const time = element.currentTime * 1000;
      if (time !== previousTime) previewFrame(time);
      fallback = requestAnimationFrame(preview);
    };
    const visibility = () => {
      reset(document.hidden ? "Camera paused while this page is hidden." : "Waiting for a live frame.");
      lastFrameAt = performance.now();
    };
    reset(props.interrupted ? "Feed interrupted." : props.stream ? "Waiting for a live frame." : "Feed stopped.");
    element.srcObject = props.stream;
    document.addEventListener("visibilitychange", visibility);
    if (props.stream && !props.interrupted) {
      if (typeof element.requestVideoFrameCallback === "function") callback = element.requestVideoFrameCallback(next);
      else fallback = requestAnimationFrame(preview);
      void element.play().catch(() => { if (!disposed) reset("Video playback could not start. Stop and restart the camera."); });
    }
    const watchdog = window.setInterval(() => {
      if (!props.stream || document.hidden || props.interrupted) return;
      if (!stalled && performance.now() - lastFrameAt > FEED_STALL_MS) {
        stalled = true; reset("Feed interrupted. Live frames stopped arriving.");
      }
      const busy = !!session.current?.busy;
      if (wasBusy !== busy) { wasBusy = busy; paint.current(); }
    }, 50);
    return () => {
      disposed = true; clearInterval(watchdog);
      document.removeEventListener("visibilitychange", visibility);
      if (callback) element.cancelVideoFrameCallback(callback);
      if (fallback) cancelAnimationFrame(fallback);
      element.pause(); element.srcObject = null;
      latest.current = null; session.current?.retake();
    };
  }, [props.stream, props.interrupted]);

  return <>
    <div className="camera-capture-controls">
      <button type="button" disabled={!ui.canCapture} onClick={() => {
        const active = current.current;
        if (active.stream && !active.interrupted && active.visible && !document.hidden && latest.current)
          session.current?.capture(latest.current);
        paint.current();
      }}>{ui.identifying ? "Identifying…" : "Capture & identify"}</button>
      <button type="button" className="secondary" disabled={!ui.frozen} onClick={() => {
        latest.current = null; status.current = "Waiting for a live frame.";
        session.current?.retake(); paint.current();
      }}>Retake</button>
      <span>{ui.frozen ? "Captured still" : "Live preview"}</span>
    </div>
    <video ref={video} autoPlay muted playsInline aria-hidden="true" style={{ position: "absolute", width: 1, height: 1, opacity: 0, pointerEvents: "none" }} />
    <canvas ref={canvas} style={{ width: "100%", display: "block" }} role="img" aria-label={ui.frozen ? "Captured camera frame with anatomy identification" : "Live camera preview"} />
    <p role="status" className="notice">{ui.notice}</p>
  </>;
}
