import { useEffect, useLayoutEffect, useRef, useState } from "react";
import type { AnatomyId } from "@holospex/contracts";
import { drawHud, type HudAppearance } from "./drawHud";
import { buildDatasetRaster, datasetOverlayState, selectDatasetPixel, type DatasetView } from "./datasetRaster";
import { decodeIndexMask } from "./decodeIndexMask";
import { datasetSourceLabel, type LoadedDatasetSample } from "./datasetSamples";
import type { DisplayedFrame } from "./selectFrame";
import type { HudMode } from "./hudControls";

export interface DatasetSelection {
  source: "supplied_dataset_annotation";
  frame: DisplayedFrame;
  point: [number, number];
  structureId: AnatomyId | null;
  status: "annotated" | "background" | "ignored";
}
export interface DatasetSampleHudProps {
  sample: LoadedDatasetSample | null;
  mode: HudMode;
  visible: boolean;
  view?: DatasetView;
  appearance?: HudAppearance;
  onDisplayedFrame?: (frame: DisplayedFrame | null) => void;
  onSelection?: (selection: DatasetSelection) => void;
  onCanvas?: (canvas: HTMLCanvasElement | null) => void;
}
function pixelCanvas(width: number, height: number, pixels: Uint8ClampedArray) {
  const canvas = document.createElement("canvas"); canvas.width = width; canvas.height = height;
  const context = canvas.getContext("2d");
  if (!context) throw new Error("Canvas rendering is unavailable.");
  const data = context.createImageData(width, height); data.data.set(pixels); context.putImageData(data, 0, 0);
  return canvas;
}
type Prepared = ReturnType<typeof buildDatasetRaster> & {
  sample: LoadedDatasetSample; image: ImageBitmap; fillCanvas: HTMLCanvasElement; outlineCanvas: HTMLCanvasElement;
};

/** The sample, image and exact mask become visible together, only after validation. */
export function DatasetSampleHud(props: DatasetSampleHudProps) {
  const canvas = useRef<HTMLCanvasElement>(null), current = useRef(props); current.current = props;
  const prepared = useRef<Prepared | null>(null);
  const failure = useRef<string | null>(null);
  const layout = useRef<ReturnType<typeof drawHud>>(undefined);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const paint = useRef<() => void>(() => {});
  const overlay = datasetOverlayState(props.visible, props.mode, props.view ?? "raster");
  paint.current = () => {
    if (!canvas.current) return;
    const settings = current.current, data = prepared.current?.sample === settings.sample ? prepared.current : null;
    const state = datasetOverlayState(settings.visible, settings.mode, settings.view ?? "raster");
    const show = data !== null && state.show;
    layout.current = drawHud(canvas.current, data?.image ?? null, {
      width: data?.sample.labels.frame.width ?? 854, height: data?.sample.labels.frame.height ?? 480,
      sourceLabel: datasetSourceLabel,
      labelSlots: data ? Math.max(data.anchors.length, data.sample.labels.structures.length) : 6,
      statusLabel: data ? `Case ${data.sample.id} · Still frame 0` : "Waiting for a validated still",
      structures: show && settings.view === "polygons" ? data.sample.labels.structures : [],
      anchors: show && settings.view !== "polygons" ? data.anchors : undefined,
      raster: show && settings.view !== "polygons" ? { fill: data.fillCanvas, outline: data.outlineCanvas } : undefined,
      appearance: settings.appearance,
      warning: data ? state.warning : failure.current,
    });
  };
  useLayoutEffect(() => { paint.current(); }, [props.sample, props.visible, props.mode, props.view,
    props.appearance?.fillOpacity, props.appearance?.showBoundaries, props.appearance?.showLabels]);
  useEffect(() => {
    const resize = new ResizeObserver(() => paint.current()); resize.observe(canvas.current!);
    current.current.onCanvas?.(canvas.current);
    return () => { resize.disconnect(); current.current.onCanvas?.(null); };
  }, []);
  useEffect(() => {
    let disposed = false;
    const controller = new AbortController();
    const sample = props.sample;
    prepared.current = null; failure.current = null; setError(null); setLoading(!!sample);
    current.current.onDisplayedFrame?.(null); paint.current();
    let image: ImageBitmap | null = null;
    void (async () => {
      if (!sample) return;
      try {
        image = await createImageBitmap(sample.image);
        if (disposed || current.current.sample !== sample) { image.close(); return; }
        const { width, height } = sample.labels.frame;
        if (image.width !== width || image.height !== height) throw new Error("Original image dimensions disagree with the label record.");
        const mask = await decodeIndexMask(sample.indexMask, width, height, controller.signal);
        if (disposed || current.current.sample !== sample) return;
        const raster = buildDatasetRaster(sample.labels, mask);
        prepared.current = { ...raster, sample, image, fillCanvas: pixelCanvas(width, height, raster.fill), outlineCanvas: pixelCanvas(width, height, raster.outline) };
        setLoading(false); paint.current(); current.current.onDisplayedFrame?.({ ...sample.labels.frame });
      } catch (cause) {
        image?.close(); image = null;
        if (!disposed && current.current.sample === sample) {
          prepared.current = null; setLoading(false);
          failure.current = `Unable to assess: ${cause instanceof Error ? cause.message : "sample could not load."}`;
          setError(failure.current); paint.current();
        }
      }
    })();
    return () => { disposed = true; controller.abort(); image?.close(); prepared.current = null; current.current.onDisplayedFrame?.(null); };
  }, [props.sample]);
  return <>
    <canvas ref={canvas} style={{ width: "100%", display: "block" }} role="img"
      aria-label={`${datasetSourceLabel}. ${overlay.show ? (props.appearance?.showLabels === false ? "Anatomy overlay; text labels hidden" : "Anatomy with labels beside the image") : "Anatomy overlay hidden"}.`}
      onClick={event => {
        const data = prepared.current, position = layout.current;
        if (!data || data.sample !== props.sample || !position || !props.onSelection) return;
        const bounds = event.currentTarget.getBoundingClientRect();
        const point: [number, number] = [
          ((event.clientX - bounds.left) * position.width / bounds.width - position.image.x) / position.image.scale,
          ((event.clientY - bounds.top) * position.height / bounds.height - position.image.y) / position.image.scale,
        ];
        const selected = selectDatasetPixel(data.sample.labels, data.pixels, point);
        if (selected) props.onSelection({ source: "supplied_dataset_annotation", frame: { ...data.sample.labels.frame }, point, ...selected });
      }} />
    {loading && <p role="status" className="notice">Loading and checking the original image and exact mask…</p>}
    {error && <p role="alert" className="error">{error}</p>}
    {overlay.warning && <p role="status" className="notice">{overlay.warning}</p>}
  </>;
}
