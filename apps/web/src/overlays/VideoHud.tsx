import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { anatomy, type FrameResult } from "@holospex/contracts";
import "./videoHud.css";
import { drawHud } from "./drawHud";
import { findVideoResult, visibleStructures } from "./videoResults";
import { sourceLabels, type ResultSource } from "./frameInput";
import type { HudMode } from "./hudControls";
import type { DisplayedFrame } from "./selectFrame";

export interface VideoHudProps {
  src: string;
  mediaId: string;
  results: FrameResult[];
  source: ResultSource;
  mode: HudMode;
  visible: boolean;
  minimumConfidence?: number;
  bindVideo?: (video: HTMLVideoElement | null) => void;
  onDisplayedFrame?: (frame: DisplayedFrame | null) => void;
  onCanvas?: (canvas: HTMLCanvasElement | null) => void;
  onReady?: () => void;
  onPlayback?: (playing: boolean) => void;
}

/** Canvas presents the captured video frame and its matched overlay atomically. */
export function VideoHud(props: VideoHudProps) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const video = useRef<HTMLVideoElement>(null);
  const snapshot = useRef<HTMLCanvasElement | null>(null);
  const presented = useRef<{ timestampMs: number; width: number; height: number } | null>(null);
  const current = useRef(props); current.current = props;
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const paint = useRef<() => void>(() => {});
  paint.current = () => {
    if (!canvas.current) return;
    const settings = current.current, frame = presented.current;
    const result = frame ? findVideoResult(settings.results, settings.mediaId, settings.source, frame.timestampMs, frame.width, frame.height) : null;
    const overlay = visibleStructures(result, settings.visible, settings.mode, settings.minimumConfidence);
    setNotice(overlay.warning);
    drawHud(canvas.current, frame ? snapshot.current : null, {
      width: frame?.width ?? (video.current?.videoWidth || 960), height: frame?.height ?? (video.current?.videoHeight || 540),
      sourceLabel: sourceLabels[settings.source], statusLabel: frame ? `${result ? `Frame ${result.frameNumber} · ` : ""}${frame.timestampMs.toFixed(1)} ms` : "Waiting for a video frame",
      // Reserve the full rail even when a frame has fewer or no visible labels.
      labelSlots: Object.keys(anatomy).length,
      structures: overlay.structures, warning: overlay.warning,
    });
    settings.onDisplayedFrame?.(result ? { mediaId: result.mediaId, frameNumber: result.frameNumber, timestampMs: result.timestampMs, width: result.width, height: result.height } : null);
  };
  useLayoutEffect(() => { paint.current(); }, [props.results, props.mediaId, props.source, props.mode, props.visible, props.minimumConfidence]);
  useEffect(() => {
    const element = video.current!;
    props.bindVideo?.(element); props.onCanvas?.(canvas.current);
    const resize = new ResizeObserver(() => paint.current()); resize.observe(canvas.current!);
    return () => { resize.disconnect(); element.pause(); current.current.bindVideo?.(null); current.current.onCanvas?.(null); current.current.onDisplayedFrame?.(null); };
  }, []);
  useEffect(() => {
    const element = video.current!;
    setError(null);
    snapshot.current = document.createElement("canvas");
    let callback = 0, disposed = false;
    const clear = () => { presented.current = null; paint.current(); };
    const next = (_: number, metadata: VideoFrameCallbackMetadata) => {
      if (disposed) return;
      const width = element.videoWidth, height = element.videoHeight;
      if (!element.seeking && width > 0 && height > 0) {
        const image = snapshot.current!;
        if (image.width !== width || image.height !== height) { image.width = width; image.height = height; }
        image.getContext("2d")?.drawImage(element, 0, 0, width, height);
        presented.current = { timestampMs: metadata.mediaTime * 1000, width, height };
        paint.current();
      } else clear();
      callback = element.requestVideoFrameCallback(next);
    };
    const names = ["seeking", "emptied", "loadstart", "error"];
    names.forEach(name => element.addEventListener(name, clear));
    clear();
    if (typeof element.requestVideoFrameCallback === "function") callback = element.requestVideoFrameCallback(next);
    else setError("This browser cannot synchronize video frames. Use a browser with video-frame callback support.");
    return () => { disposed = true; if (callback) element.cancelVideoFrameCallback(callback); names.forEach(name => element.removeEventListener(name, clear)); clear(); };
  }, [props.src]);
  return <>
    <video ref={video} src={props.src} playsInline muted preload="auto" style={{ position: "absolute", width: 1, height: 1, opacity: 0, pointerEvents: "none" }} aria-hidden="true"
      onLoadedData={props.onReady} onPlaying={() => props.onPlayback?.(true)} onPause={() => props.onPlayback?.(false)} onEnded={() => props.onPlayback?.(false)}
      onError={() => setError("Unable to play this clip. Check the file and browser codec support.")} />
    <canvas ref={canvas} style={{ width: "100%", display: "block", background: "#09181e" }} role="img" aria-label={`Video anatomy HUD. ${sourceLabels[props.source]}. ${props.visible ? props.mode : "Overlays hidden"}.`} />
    {error && <p role="alert" className="error">{error}</p>}
    <p role="status" className="notice video-hud-notice">{notice}</p>
  </>;
}
