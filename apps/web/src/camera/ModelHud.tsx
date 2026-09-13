import { useEffect, useLayoutEffect, useRef, useState } from "react";
import aruco from "js-aruco2";
import { drawHud, type HudAnchor } from "../overlays/drawHud";
import type { HudMode } from "../overlays/hudControls";
import { registerModel, type ModelRegistration } from "./modelRegistration";
const { AR } = aruco;

export function ModelHud(props: {
  stream: MediaStream | null; registration: ModelRegistration | null;
  visible: boolean; mode: HudMode; interrupted: boolean;
  onCanvas?: (canvas: HTMLCanvasElement | null) => void;
  onTracking?: (tracked: boolean) => void;
}) {
  const video = useRef<HTMLVideoElement>(null), canvas = useRef<HTMLCanvasElement>(null);
  const snapshot = useRef<HTMLCanvasElement | null>(null);
  const anchors = useRef<HudAnchor[] | null>(null);
  const current = useRef(props); current.current = props;
  const [message, setMessage] = useState("Camera stopped.");
  const status = useRef("Camera stopped.");
  const paint = useRef(() => {});
  paint.current = () => {
    if (!canvas.current) return;
    const active = current.current;
    const show = active.visible && active.mode !== "identify" && active.mode !== "assess" && !active.interrupted;
    drawHud(canvas.current, snapshot.current, {
      width: snapshot.current?.width || 640, height: snapshot.current?.height || 480,
      sourceLabel: active.registration?.provenance === "synthetic_mock" ? "Synthetic marker fixture" : "Known model locations",
      statusLabel: active.registration?.modelId ?? "No model configuration",
      structures: [], anchors: show ? anchors.current ?? [] : [],
      warning: anchors.current === null || active.interrupted ? status.current : null,
    });
  };
  useLayoutEffect(() => { if (props.interrupted) { anchors.current = null; status.current = "Tracking lost. Camera interrupted."; props.onTracking?.(false); } paint.current(); }, [props.visible, props.mode, props.interrupted]);
  useLayoutEffect(() => {
    anchors.current = null; snapshot.current = null;
    status.current = props.stream ? "Tracking unavailable. Waiting for the configured marker." : "Camera stopped.";
    props.onTracking?.(false); paint.current();
  }, [props.stream, props.registration]);
  useEffect(() => {
    props.onCanvas?.(canvas.current);
    const resize = new ResizeObserver(() => paint.current()); resize.observe(canvas.current!);
    return () => { resize.disconnect(); current.current.onCanvas?.(null); };
  }, []);
  useEffect(() => {
    const element = video.current!;
    element.srcObject = props.stream;
    const image = document.createElement("canvas"); snapshot.current = null;
    let callback = 0, lastFrameAt = 0, disposed = false;
    const detector = new AR.Detector({ dictionaryName: "ARUCO_MIP_36h12", maxHammingDistance: 0 });
    const update = (text: string, points: HudAnchor[] | null) => {
      status.current = text; anchors.current = points; setMessage(text); current.current.onTracking?.(points !== null); paint.current();
    };
    update(props.stream ? "Tracking lost. Bring the configured marker into view." : "Camera stopped.", null);
    const next = () => {
      if (disposed) return;
      lastFrameAt = performance.now();
      const settings = current.current;
      if (element.videoWidth && element.videoHeight && !settings.interrupted) {
        image.width = Math.min(640, element.videoWidth); image.height = Math.round(image.width * element.videoHeight / element.videoWidth);
        const context = image.getContext("2d", { willReadFrequently: true })!;
        context.drawImage(element, 0, 0, image.width, image.height); snapshot.current = image;
        if (!settings.registration) update("Tracking unavailable. Load the measured model configuration.", null);
        else {
          try {
            const markers = detector.detect(context.getImageData(0, 0, image.width, image.height)).filter(item => item.id === settings.registration!.markerId);
            const points = markers.length === 1 ? registerModel(settings.registration, markers[0].corners, image.width, image.height) : null;
            update(points ? "Marker tracked. Labels use known model locations." : "Tracking lost. Reacquire the marker with a clear view.", points);
          } catch { update("Tracking lost. Marker pose could not be assessed.", null); }
        }
      } else update("Tracking lost. Camera frames are unavailable.", null);
      callback = element.requestVideoFrameCallback(next);
    };
    if (props.stream && typeof element.requestVideoFrameCallback === "function") callback = element.requestVideoFrameCallback(next);
    else if (props.stream) update("Tracking unavailable. This browser cannot report video frames.", null);
    const watchdog = window.setInterval(() => {
      if (anchors.current !== null && performance.now() - lastFrameAt > 250) update("Tracking lost. Camera frames stopped arriving.", null);
    }, 100);
    return () => { disposed = true; clearInterval(watchdog); if (callback) element.cancelVideoFrameCallback(callback); element.srcObject = null; anchors.current = null; current.current.onTracking?.(false); };
  }, [props.stream, props.registration]);
  return <>
    <video ref={video} autoPlay muted playsInline aria-hidden="true" style={{ position: "absolute", width: 1, height: 1, opacity: 0, pointerEvents: "none" }} />
    <canvas ref={canvas} style={{ width: "100%", display: "block" }} role="img" aria-label="Camera with marker-registered model labels" />
    <p role="status">{message}</p>
  </>;
}
