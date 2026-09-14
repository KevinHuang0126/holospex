import { useEffect, useLayoutEffect, useRef, useState } from "react";
import type { FrameResult } from "@holospex/contracts";
import { createLiveFramePipeline, MAX_LIVE_FRAME_AGE_MS } from "../input/liveFramePipeline";
import type { LiveFrameIdentifier } from "../input/liveIdentification";
import { drawHud, type HudAppearance } from "../overlays/drawHud";
import { sourceLabels } from "../overlays/frameInput";
import type { DisplayedFrame } from "../overlays/selectFrame";
import { visibleStructures } from "../overlays/videoResults";
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
interface CapturedFrame { image: HTMLCanvasElement; frame: DisplayedFrame; capturedAt: number }
interface IdentifiedFrame extends CapturedFrame { result: FrameResult }
const FRAME_INTERVAL_MS = 200;
const FEED_STALL_MS = 500;

/** Live pixels and their result are presented together, with a bounded processing delay.
 * Never reproject a prediction from an older frame onto a newer camera image.
 */
export function LiveFeedHud(props: LiveFeedHudProps) {
  const video = useRef<HTMLVideoElement>(null), canvas = useRef<HTMLCanvasElement>(null);
  const current = useRef(props); current.current = props;
  const latest = useRef<CapturedFrame | null>(null), identified = useRef<IdentifiedFrame | null>(null);
  const status = useRef("Feed stopped.");
  const [notice, setNotice] = useState(status.current);
  const paint = useRef(() => {});
  const identificationEnabled = !!props.identify && props.visible;

  paint.current = () => {
    if (!canvas.current) return;
    const active = current.current;
    const fresh = identified.current && performance.now() - identified.current.capturedAt < MAX_LIVE_FRAME_AGE_MS;
    const paired = active.stream && !active.interrupted && active.visible && active.identify && fresh ? identified.current : null;
    const display = active.stream && !active.interrupted ? paired ?? latest.current : null;
    const overlay = visibleStructures(paired?.result ?? null, active.visible, "learn", active.minimumConfidence);
    const warning = overlay.warning?.replace(/Unable to assess/g, "Unable to identify") ?? null;
    const message = !active.stream ? "Feed stopped."
      : active.interrupted ? "Feed interrupted. Restart the input if it does not recover."
      : !display ? status.current
      : !active.visible ? "Live preview. Anatomy hidden; identification paused."
      : !active.identify ? "Live preview. The anatomy identification model is not ready yet."
      : paired ? warning ?? (overlay.structures.length ? "Anatomy identified in the displayed capture." : "No supported anatomy identified in this frame.")
      : status.current;
    setNotice(message);
    drawHud(canvas.current, display?.image ?? null, {
      width: display?.frame.width ?? 1280, height: display?.frame.height ?? 720,
      source: paired?.result.source,
      sourceLabel: paired ? sourceLabels[paired.result.source] : "Live video feed",
      statusLabel: display ? `Frame ${display.frame.frameNumber} · ${display.frame.timestampMs.toFixed(1)} ms${paired ? ` · ${Math.round(performance.now() - paired.capturedAt)} ms behind capture` : ""}` : "No live frame",
      structures: paired ? overlay.structures : [], labelSlots: 6,
      appearance: active.appearance,
      warning: !display || active.interrupted || (active.identify && active.visible && !paired) ? message : paired ? warning : null,
    });
    active.onDisplayedFrame?.(display?.frame ?? null);
  };
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
    let lastFrameAt = performance.now(), lastSubmittedAt = -Infinity, stalled = false;
    let captured: CapturedFrame | null = null;
    let pipeline: ReturnType<typeof createLiveFramePipeline> | null = null;
    const raw = document.createElement("canvas");
    const synchronized = typeof element.requestVideoFrameCallback === "function";
    const reset = (message: string) => {
      pipeline?.dispose(); pipeline = null; captured = null;
      latest.current = null; identified.current = null;
      mediaId = `live:${globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(36).slice(2)}`}`; frameNumber = 0;
      status.current = message; paint.current();
    };
    const unavailable = (message: string) => {
      if (disposed) return;
      identified.current = null; status.current = message; paint.current();
    };
    const ensurePipeline = () => {
      if (!pipeline && identificationEnabled && props.identify && synchronized) {
        pipeline = createLiveFramePipeline({
          identify: props.identify,
          onUnavailable: unavailable,
          onResult(frame, result, capturedAt) {
            if (disposed || document.hidden || !captured || captured.frame.mediaId !== frame.mediaId || captured.frame.frameNumber !== frame.frameNumber) return;
            identified.current = { ...captured, result, capturedAt };
            status.current = "Unable to identify. Waiting for the next identified frame.";
            paint.current();
          },
        });
      }
    };
    const capture = (timestampMs: number) => {
      if (disposed || document.hidden || props.interrupted || !props.stream) return;
      if (element.readyState < 2 || !element.videoWidth || !element.videoHeight || !Number.isFinite(timestampMs) || timestampMs < 0) return;
      try {
        const dimensions = videoCaptureSize(element.videoWidth, element.videoHeight);
        if (!dimensions) return;
        const now = performance.now(), { width, height } = dimensions;
        const size = `${element.videoWidth}x${element.videoHeight}`;
        if (stalled || size !== nativeSize || timestampMs < previousTime) {
          reset("Unable to identify. Waiting for a result for this feed.");
          nativeSize = size; stalled = false;
        }
        previousTime = timestampMs; lastFrameAt = now;
        if (raw.width !== width || raw.height !== height) { raw.width = width; raw.height = height; }
        const context = raw.getContext("2d");
        if (!context) throw new Error("Video rendering is unavailable.");
        context.drawImage(element, 0, 0, width, height);
        const frame: DisplayedFrame = { mediaId, frameNumber: frameNumber++, timestampMs, width, height };
        latest.current = { image: raw, frame, capturedAt: now };
        ensurePipeline();
        if (pipeline && !pipeline.busy && now - lastSubmittedAt >= FRAME_INTERVAL_MS) {
          // Own the pixels while JPEG encoding and inference run asynchronously.
          const image = document.createElement("canvas"); image.width = width; image.height = height;
          const pixels = image.getContext("2d");
          if (!pixels) throw new Error("Frame capture is unavailable.");
          pixels.drawImage(raw, 0, 0);
          captured = { image, frame, capturedAt: now }; lastSubmittedAt = now;
          pipeline.submit(frame, () => new Promise<Blob>((resolve, reject) => {
            image.toBlob(blob => blob ? resolve(blob) : reject(new Error("Frame encoding failed.")), "image/jpeg", 0.9);
          }), now);
        }
        if (!synchronized && identificationEnabled) status.current = "Unable to identify. This browser supports preview only; video-frame callbacks are unavailable.";
        paint.current();
      } catch {
        reset("Unable to identify. The video frame could not be captured.");
      }
    };
    const next = (_: number, metadata: VideoFrameCallbackMetadata) => {
      if (disposed) return;
      capture(metadata.mediaTime * 1000);
      callback = element.requestVideoFrameCallback(next);
    };
    const preview = () => {
      if (disposed) return;
      const time = element.currentTime * 1000;
      if (time !== previousTime) capture(time);
      fallback = requestAnimationFrame(preview);
    };
    const visibility = () => {
      reset(document.hidden ? "Feed paused while this page is hidden." : "Waiting for a live frame.");
      lastFrameAt = performance.now();
    };
    reset(props.interrupted ? "Feed interrupted." : props.stream ? "Waiting for a live frame." : "Feed stopped.");
    element.srcObject = props.stream;
    document.addEventListener("visibilitychange", visibility);
    if (props.stream && !props.interrupted) {
      if (synchronized) callback = element.requestVideoFrameCallback(next);
      else fallback = requestAnimationFrame(preview);
      void element.play().catch(() => { if (!disposed) reset("Video playback could not start. Stop and restart the feed."); });
    }
    const watchdog = window.setInterval(() => {
      if (!props.stream || document.hidden || props.interrupted) return;
      if (!stalled && performance.now() - lastFrameAt > FEED_STALL_MS) {
        stalled = true; reset("Feed interrupted. Live frames stopped arriving.");
      } else if (identified.current && performance.now() - identified.current.capturedAt >= MAX_LIVE_FRAME_AGE_MS) {
        unavailable("Unable to identify. The identification result expired.");
      }
    }, 50);
    return () => {
      disposed = true; pipeline?.dispose(); clearInterval(watchdog);
      document.removeEventListener("visibilitychange", visibility);
      if (callback) element.cancelVideoFrameCallback(callback);
      if (fallback) cancelAnimationFrame(fallback);
      element.pause(); element.srcObject = null;
      latest.current = null; identified.current = null; captured = null;
    };
  }, [props.stream, props.interrupted, props.identify, identificationEnabled]);

  return <>
    <video ref={video} autoPlay muted playsInline aria-hidden="true" style={{ position: "absolute", width: 1, height: 1, opacity: 0, pointerEvents: "none" }} />
    <canvas ref={canvas} style={{ width: "100%", display: "block" }} role="img" aria-label="Live video feed with frame-matched anatomy identification" />
    <p role="status" className="notice">{notice}</p>
  </>;
}
