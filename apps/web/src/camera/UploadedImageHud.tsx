import { useEffect, useLayoutEffect, useRef, useState } from "react";
import type { LiveFrameIdentifier } from "../input/liveIdentification";
import { drawHud, type HudAppearance } from "../overlays/drawHud";
import { sourceLabels } from "../overlays/frameInput";
import { visibleStructures } from "../overlays/videoResults";
import { decodeUploadedImage } from "./uploadedImage";
import { createUploadedImageSession } from "./uploadedImageSession";

export interface UploadedImageHudProps {
  file: File | null;
  visible: boolean;
  identify?: LiveFrameIdentifier;
  minimumConfidence?: number;
  appearance?: HudAppearance;
  onCanvas?: (canvas: HTMLCanvasElement | null) => void;
}

export function UploadedImageHud(props: UploadedImageHudProps) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const current = useRef(props); current.current = props;
  const session = useRef<ReturnType<typeof createUploadedImageSession<HTMLCanvasElement>> | null>(null);
  const paint = useRef(() => {});
  const [ui, setUi] = useState({ notice: "Choose an image to identify anatomy.", enabled: false, button: "Identify image" });
  paint.current = () => {
    if (!canvas.current) return;
    const active = current.current, controller = session.current;
    const display = controller?.image, result = controller?.result;
    const state = controller?.status;
    const overlay = visibleStructures(result ?? null, active.visible, "learn", active.minimumConfidence);
    const warning = overlay.warning?.replace(/Unable to assess/g, "Unable to identify") ?? null;
    const message = state === "loading" ? "Loading image locally..."
      : state === "error" ? controller?.message ?? "Unable to identify this image. Try again."
      : !display ? "Choose an image to identify anatomy."
      : state === "identifying" ? "Identifying this image..."
      : !active.visible ? "Image preview. Anatomy hidden."
      : result ? warning ?? (overlay.structures.length ? "Anatomy identified in this image." : "No supported anatomy identified in this image.")
      : !active.identify ? "Image preview. Connecting to the identification model; select Identify image once connected."
      : controller?.busy ? "The previous identification is still finishing. Try again shortly."
      : "Image preview. Tap Identify image to identify anatomy.";
    const enabled = !!(display && state !== "identifying" && active.identify && active.visible);
    const button = state === "identifying" ? "Identifying..." : state === "error" && display ? "Retry identification" : result ? "Identify again" : "Identify image";
    setUi(previous => previous.notice === message && previous.enabled === enabled && previous.button === button
      ? previous : { notice: message, enabled, button });
    drawHud(canvas.current, display?.image ?? null, {
      width: display?.frame.width ?? 1280, height: display?.frame.height ?? 720,
      source: result?.source, sourceLabel: result ? sourceLabels[result.source] : "Uploaded image",
      statusLabel: display ? `Image frame 0 | ${display.frame.width} x ${display.frame.height}` : "No image",
      structures: result ? overlay.structures : [], labelSlots: 6, appearance: active.appearance,
      warning: state === "error" ? message : result ? warning : null,
    });
  };
  useLayoutEffect(() => {
    session.current = createUploadedImageSession({
      decode: decodeUploadedImage,
      copyImage(source: HTMLCanvasElement) {
        const image = document.createElement("canvas"); image.width = source.width; image.height = source.height;
        const context = image.getContext("2d");
        if (!context) throw new Error("Image rendering is unavailable.");
        context.drawImage(source, 0, 0); return image;
      },
      encode: image => new Promise<Blob>((resolve, reject) => {
        image.toBlob(blob => blob ? resolve(blob) : reject(new Error("Image encoding failed.")), "image/jpeg", 0.9);
      }),
      onChange: () => paint.current(),
    });
    return () => { session.current?.dispose(); session.current = null; };
  }, []);
  useLayoutEffect(() => { session.current?.setFile(props.file); }, [props.file]);
  useLayoutEffect(() => {
    session.current?.setIdentifier(props.identify); session.current?.clearResult(); paint.current();
  }, [props.identify, props.minimumConfidence]);
  useLayoutEffect(() => { paint.current(); }, [props.visible, props.appearance]);
  useEffect(() => {
    props.onCanvas?.(canvas.current);
    const resize = new ResizeObserver(() => paint.current()); resize.observe(canvas.current!);
    return () => { resize.disconnect(); current.current.onCanvas?.(null); };
  }, []);
  return <>
    <div className="uploaded-image-controls">
      <button type="button" disabled={!ui.enabled} onClick={() => {
        if (current.current.visible && current.current.identify) session.current?.identify();
      }}>{ui.button}</button>
      <span>Still image</span>
    </div>
    <canvas ref={canvas} style={{ width: "100%", display: "block" }} role="img" aria-label="Uploaded image with anatomy identification" />
    <p role="status" className="notice">{ui.notice}</p>
  </>;
}
