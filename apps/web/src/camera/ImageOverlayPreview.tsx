import { useEffect, useLayoutEffect, useRef } from "react";
import { drawHud } from "../overlays/drawHud";
import { datasetOverlayState } from "../overlays/datasetRaster";
import type { HudMode } from "../overlays/hudControls";
import type { ImageOverlayAsset, ImageOverlayView } from "./imageOverlayAsset";

/** Preview the exact texture and label positions that will be placed over the camera. */
export function ImageOverlayPreview({ asset, view, visible, mode, onCanvas }: {
  asset: ImageOverlayAsset | null; view: ImageOverlayView; visible: boolean; mode: HudMode;
  onCanvas?: (canvas: HTMLCanvasElement | null) => void;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const current = useRef({ asset, view, visible, mode, onCanvas }); current.current = { asset, view, visible, mode, onCanvas };
  const paint = useRef(() => {});
  paint.current = () => {
    if (!canvas.current) return;
    const active = current.current, state = datasetOverlayState(active.visible, active.mode, "raster");
    const layer = active.asset?.[active.view];
    drawHud(canvas.current, state.show && layer ? layer.canvas : null, {
      width: active.asset?.width || 854, height: active.asset?.height || 480,
      sourceLabel: active.asset?.sourceLabel ?? "Labeled image overlay",
      statusLabel: "Image preview · camera is off", structures: [],
      anchors: state.show ? layer?.anchors ?? [] : [], labelSlots: layer?.anchors.length ?? 6,
      warning: state.warning ?? (!active.asset ? "Choose a labeled sample to preview its image and anatomy." : "Start the camera to place this image in your view."),
    });
  };
  useEffect(() => {
    current.current.onCanvas?.(canvas.current);
    const resize = new ResizeObserver(() => paint.current()); resize.observe(canvas.current!);
    return () => { resize.disconnect(); current.current.onCanvas?.(null); };
  }, []);
  useLayoutEffect(() => paint.current(), [asset, view, visible, mode]);
  return <canvas ref={canvas} style={{ width: "100%", display: "block" }} role="img" aria-label="Labeled surgical image preview; camera is off" />;
}
