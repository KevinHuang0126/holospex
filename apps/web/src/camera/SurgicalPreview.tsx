import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { drawHud } from "../overlays/drawHud";
import type { HudMode } from "../overlays/hudControls";
import { SurgicalRenderer } from "./SurgicalRenderer";

/** A clearly identified model preview, never presented as a tracked camera frame. */
export function SurgicalPreview({ visible, mode, onCanvas }: {
  visible: boolean; mode: HudMode; onCanvas?: (canvas: HTMLCanvasElement | null) => void;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const renderer = useRef<SurgicalRenderer | null>(null);
  const current = useRef({ visible, mode, onCanvas }); current.current = { visible, mode, onCanvas };
  const [error, setError] = useState<string | null>(null);
  const paint = useRef(() => {});
  paint.current = () => {
    if (!canvas.current || !renderer.current) return;
    try {
      const show = current.current.visible && !["identify", "assess"].includes(current.current.mode);
      const preview = renderer.current.preview(900, 800);
      drawHud(canvas.current, show ? preview.image : null, { width: 900, height: 800,
        sourceLabel: "Synthetic 3D scene", statusLabel: "Preview · camera is off", structures: [],
        anchors: show ? preview.anchors : [], labelSlots: 3,
        warning: "Start the camera and point it at the marker on your table to place this scene.",
      });
    } catch { setError("The 3D preview could not render. Open this page in a browser with WebGL 2 enabled."); }
  };
  useEffect(() => {
    try { renderer.current = new SurgicalRenderer(); paint.current(); }
    catch { setError("The 3D preview needs WebGL 2. Try a current phone browser."); }
    current.current.onCanvas?.(canvas.current);
    const resize = new ResizeObserver(() => paint.current()); resize.observe(canvas.current!);
    return () => { resize.disconnect(); renderer.current?.dispose(); renderer.current = null; current.current.onCanvas?.(null); };
  }, []);
  useLayoutEffect(() => paint.current(), [visible, mode]);
  return <><canvas ref={canvas} style={{ width: "100%", display: "block" }} role="img" aria-label="Preview of a 3D open-abdomen training model with gallbladder, cystic duct and artery labels" />
    {error && <p role="alert">{error}</p>}</>;
}
