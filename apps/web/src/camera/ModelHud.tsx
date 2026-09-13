import { useEffect, useLayoutEffect, useRef, useState } from "react";
import aruco from "js-aruco2";
import { drawHud, type HudAnchor } from "../overlays/drawHud";
import type { HudMode } from "../overlays/hudControls";
import { estimateModelPose, projectModelAnchors, type ModelPose, type ModelRegistration } from "./modelRegistration";
import { SurgicalRenderer } from "./SurgicalRenderer";
import { ImagePlaneRenderer, projectScreenAnchors, screenImagePlacement } from "./ImagePlaneRenderer";
import type { ImageOverlayAsset, ImageOverlayView } from "./imageOverlayAsset";
import { datasetOverlayState } from "../overlays/datasetRaster";
const { AR } = aruco;

export interface CameraImageOverlay {
  asset: ImageOverlayAsset;
  view: ImageOverlayView;
  placement: "screen" | "table";
  opacity: number;
  widthFraction: number;
  planeWidthMm: number;
}

export function ModelHud(props: {
  stream: MediaStream | null; registration: ModelRegistration | null;
  visible: boolean; mode: HudMode; interrupted: boolean;
  surgicalScene?: boolean;
  imageOverlay?: CameraImageOverlay | null;
  onCanvas?: (canvas: HTMLCanvasElement | null) => void;
  onTracking?: (tracked: boolean) => void;
  onCameraSize?: (size: { width: number; height: number }) => void;
}) {
  const video = useRef<HTMLVideoElement>(null), canvas = useRef<HTMLCanvasElement>(null);
  const snapshot = useRef<HTMLCanvasElement | null>(null);
  const anchors = useRef<HudAnchor[] | null>(null);
  const pose = useRef<ModelPose | null>(null);
  const renderer = useRef<SurgicalRenderer | null>(null);
  const rendererError = useRef<string | null>(null);
  const imageRenderer = useRef<ImagePlaneRenderer | null>(null);
  const imageRendererError = useRef<string | null>(null);
  const screenLayer = useRef<HTMLCanvasElement | null>(null);
  const current = useRef(props); current.current = props;
  const [message, setMessage] = useState("Camera stopped.");
  const status = useRef("Camera stopped.");
  const paint = useRef(() => {});
  paint.current = () => {
    if (!canvas.current) return;
    const active = current.current;
    const imageOverlay = active.imageOverlay;
    const imageState = datasetOverlayState(active.visible, active.mode, "raster");
    const show = (imageOverlay ? imageState.show : active.visible && active.mode !== "identify" && active.mode !== "assess") && !active.interrupted;
    let modelLayer: HTMLCanvasElement | undefined;
    let renderedAnchors: HudAnchor[] = [];
    let renderWarning = imageOverlay ? imageState.warning || imageRendererError.current : rendererError.current;
    if (show && imageOverlay && imageOverlay.opacity > 0 && snapshot.current && anchors.current !== null) {
      const image = imageOverlay.asset[imageOverlay.view];
      const width = snapshot.current.width, height = snapshot.current.height;
      try {
        if (imageOverlay.placement === "screen") {
          const layer = screenLayer.current ??= document.createElement("canvas");
          if (layer.width !== width || layer.height !== height) { layer.width = width; layer.height = height; }
          const context = layer.getContext("2d")!;
          context.clearRect(0, 0, width, height);
          const rect = screenImagePlacement(image.canvas.width, image.canvas.height, width, height, imageOverlay.widthFraction);
          context.globalAlpha = Math.max(0, Math.min(1, imageOverlay.opacity));
          context.drawImage(image.canvas, rect.x, rect.y, rect.width, rect.height); context.globalAlpha = 1;
          modelLayer = layer; renderedAnchors = projectScreenAnchors(image.anchors, rect);
        } else if (imageRenderer.current && pose.current && active.registration) {
          const result = imageRenderer.current.render(pose.current, width, height, imageOverlay.planeWidthMm, active.registration.markerSizeMm, imageOverlay.opacity);
          modelLayer = result.image; renderedAnchors = result.anchors;
        }
      } catch (cause) { renderWarning = cause instanceof Error ? cause.message : "Image overlay unavailable."; }
    } else if (show && active.surgicalScene && pose.current && snapshot.current && renderer.current) {
      try { modelLayer = renderer.current.render(pose.current, snapshot.current.width, snapshot.current.height); }
      catch (cause) { renderWarning = cause instanceof Error ? cause.message : "3D rendering unavailable."; }
    }
    drawHud(canvas.current, snapshot.current, {
      width: snapshot.current?.width || 640, height: snapshot.current?.height || 480,
      sourceLabel: imageOverlay?.asset.sourceLabel ?? (active.surgicalScene ? "Synthetic 3D scene" : !active.registration ? "Camera preview" : active.registration.provenance === "synthetic_mock" ? "Synthetic marker fixture" : "Known model locations"),
      statusLabel: imageOverlay ? `Case ${imageOverlay.asset.id} · ${imageOverlay.placement === "screen" ? "Screen placement" : "Table marker placement"}` : active.registration?.modelId ?? "No image or model selected",
      structures: [], anchors: imageOverlay ? renderedAnchors : show && (!active.surgicalScene || modelLayer) ? anchors.current ?? [] : [], modelLayer,
      labelSlots: imageOverlay ? imageOverlay.asset[imageOverlay.view].anchors.length : active.registration?.anchors.length ?? 3,
      warning: renderWarning || (anchors.current === null || active.interrupted ? status.current : null),
    });
  };
  useLayoutEffect(() => {
    if (props.interrupted) { anchors.current = null; pose.current = null; status.current = "Tracking lost. Camera interrupted."; setMessage(status.current); props.onTracking?.(false); }
    paint.current();
  }, [props.visible, props.mode, props.interrupted, props.imageOverlay?.opacity, props.imageOverlay?.widthFraction, props.imageOverlay?.planeWidthMm]);
  useLayoutEffect(() => {
    anchors.current = null; pose.current = null;
    status.current = props.stream ? "Tracking unavailable. Waiting for the configured marker." : "Camera stopped.";
    setMessage(status.current); props.onTracking?.(false); paint.current();
  }, [props.stream, props.registration, props.surgicalScene, props.imageOverlay?.asset, props.imageOverlay?.view, props.imageOverlay?.placement]);
  useLayoutEffect(() => { snapshot.current = null; paint.current(); }, [props.stream]);
  useEffect(() => {
    rendererError.current = null;
    if (props.surgicalScene) {
      try { renderer.current = new SurgicalRenderer(); }
      catch { rendererError.current = "3D rendering needs WebGL 2. Try a current phone browser."; }
    }
    paint.current();
    return () => { renderer.current?.dispose(); renderer.current = null; };
  }, [props.surgicalScene]);
  useEffect(() => {
    imageRendererError.current = null;
    if (props.imageOverlay?.placement === "table") {
      try { imageRenderer.current = new ImagePlaneRenderer(props.imageOverlay.asset[props.imageOverlay.view]); }
      catch { imageRendererError.current = "Table placement needs WebGL 2. Choose Camera screen to test the image overlay."; }
    }
    paint.current();
    return () => { imageRenderer.current?.dispose(); imageRenderer.current = null; };
  }, [props.imageOverlay?.asset, props.imageOverlay?.view, props.imageOverlay?.placement]);
  useEffect(() => {
    props.onCanvas?.(canvas.current);
    const resize = new ResizeObserver(() => paint.current()); resize.observe(canvas.current!);
    return () => { resize.disconnect(); current.current.onCanvas?.(null); };
  }, []);
  useEffect(() => {
    const element = video.current!;
    element.srcObject = props.stream;
    const image = document.createElement("canvas"), detection = document.createElement("canvas"); snapshot.current = null;
    let callback = 0, lastFrameAt = 0, disposed = false, nativeSize = "";
    const detector = new AR.Detector({ dictionaryName: "ARUCO_MIP_36h12", maxHammingDistance: 0 });
    const update = (text: string, points: HudAnchor[] | null, nextPose: ModelPose | null = null) => {
      const wasTracked = anchors.current !== null;
      status.current = text; anchors.current = points; pose.current = nextPose; setMessage(text);
      if (wasTracked !== (points !== null)) current.current.onTracking?.(points !== null && current.current.imageOverlay?.placement !== "screen");
      paint.current();
    };
    update(props.stream ? "Tracking lost. Bring the configured marker into view." : "Camera stopped.", null);
    const next = () => {
      if (disposed) return;
      lastFrameAt = performance.now();
      const settings = current.current;
      try {
        if (element.videoWidth && element.videoHeight && !settings.interrupted) {
          const key = `${element.videoWidth}x${element.videoHeight}`;
          if (key !== nativeSize) {
            nativeSize = key;
            settings.onCameraSize?.({ width: element.videoWidth, height: element.videoHeight });
          }
          const width = Math.min(1280, element.videoWidth), height = Math.round(width * element.videoHeight / element.videoWidth);
          if (image.width !== width || image.height !== height) { image.width = width; image.height = height; }
          const context = image.getContext("2d")!;
          context.drawImage(element, 0, 0, width, height); snapshot.current = image;
          const config = settings.registration;
          if (settings.imageOverlay?.placement === "screen") update("Image placed on the camera screen. This placement stays fixed as you move the phone.", []);
          else if (!config) update("Camera ready. Choose a labeled sample or a marker setup to display an overlay.", null);
          else if (Math.abs(element.videoWidth / element.videoHeight - config.calibration.width / config.calibration.height) > 0.01) {
            update(`Camera size ${element.videoWidth} × ${element.videoHeight} does not match calibration aspect ${config.calibration.width} × ${config.calibration.height}. Supply calibration for this camera view.`, null);
          } else {
            const detectionWidth = Math.min(640, width), detectionHeight = Math.round(detectionWidth * height / width);
            if (detection.width !== detectionWidth || detection.height !== detectionHeight) { detection.width = detectionWidth; detection.height = detectionHeight; }
            const pixels = detection.getContext("2d", { willReadFrequently: true })!;
            pixels.drawImage(image, 0, 0, detectionWidth, detectionHeight);
            const markers = detector.detect(pixels.getImageData(0, 0, detectionWidth, detectionHeight)).filter(item => item.id === config.markerId);
            const nextPose = markers.length === 1 ? estimateModelPose(config, markers[0].corners, detectionWidth, detectionHeight) : null;
            const points = nextPose ? projectModelAnchors(config, nextPose) : null;
            const projected = points?.map(point => ({ ...point, x: point.x * width / detectionWidth, y: point.y * height / detectionHeight,
              ...(config.provenance === "synthetic_mock" && !settings.surgicalScene && !settings.imageOverlay ? { label: `Test point ${config.anchors.findIndex(anchor => anchor.id === point.id) + 1}` } : {}),
            })) ?? null;
            const trackedText = settings.imageOverlay ? "Image placed on the table. Keep the marker fully visible." : settings.surgicalScene ? "Scene placed. Keep the marker in view as you move around the table." : config.provenance === "synthetic_mock" ? "Marker tracked. Synthetic test points; approximate calibration." : "Marker tracked. Labels use supplied model locations.";
            update(projected === null
              ? markers.length > 1 ? "Tracking lost. More than one matching marker is visible. Keep only one in view."
                : `Tracking lost. Keep marker ${config.markerId} fully visible with a clear view.`
              : projected.length ? trackedText : "Marker tracked. Move back to bring the scene and its locations into view.", projected, nextPose);
          }
        } else update("Tracking lost. Camera frames are unavailable.", null);
      } catch { update("Tracking lost. Camera frame or marker pose could not be assessed.", null); }
      callback = element.requestVideoFrameCallback(next);
    };
    if (props.stream && typeof element.requestVideoFrameCallback === "function") callback = element.requestVideoFrameCallback(next);
    else if (props.stream) update("Tracking unavailable. This browser cannot report video frames.", null);
    if (props.stream) void element.play().catch(() => { if (!disposed) update("Camera playback could not start. Stop and restart the camera.", null); });
    const watchdog = window.setInterval(() => {
      if (anchors.current !== null && performance.now() - lastFrameAt > 250) update("Tracking lost. Camera frames stopped arriving.", null);
    }, 100);
    return () => { disposed = true; clearInterval(watchdog); if (callback) element.cancelVideoFrameCallback(callback); element.srcObject = null; anchors.current = null; pose.current = null; current.current.onTracking?.(false); };
  }, [props.stream]);
  return <>
    <video ref={video} autoPlay muted playsInline aria-hidden="true" style={{ position: "absolute", width: 1, height: 1, opacity: 0, pointerEvents: "none" }} />
    <canvas ref={canvas} style={{ width: "100%", display: "block" }} role="img" aria-label={props.imageOverlay ? "Live camera with labeled surgical image overlay" : "Camera with marker-registered model labels"} />
    <p role="status">{message}</p>
  </>;
}
