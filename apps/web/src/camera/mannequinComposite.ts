import type { HudAnchor } from "../overlays/drawHud";
import type { ImageOverlayAsset, ImageOverlayLayer, ImageOverlayView } from "./imageOverlayAsset";

export interface MannequinSamplePlacement {
  centerX: number;
  centerY: number;
  widthFraction: number;
}
export const DEFAULT_MANNEQUIN_SAMPLE_PLACEMENT: MannequinSamplePlacement = {
  centerX: 0.34, centerY: 0.54, widthFraction: 0.12,
};
export interface MannequinSampleLayout { x: number; y: number; width: number; height: number; scale: number }

function checkDimensions(width: number, height: number) {
  if (![width, height].every(value => Number.isInteger(value) && value > 0 && value <= 8192) || width * height > 16_000_000)
    throw new Error("Sample and mannequin dimensions must be positive pixels within the supported image limits.");
}

/** Preserve the full sample aspect ratio and keep every source pixel within the mannequin photo. */
export function mannequinSampleLayout(sampleWidth: number, sampleHeight: number, mannequinWidth: number, mannequinHeight: number,
  placement: MannequinSamplePlacement = DEFAULT_MANNEQUIN_SAMPLE_PLACEMENT): MannequinSampleLayout {
  checkDimensions(sampleWidth, sampleHeight); checkDimensions(mannequinWidth, mannequinHeight);
  if (![placement.centerX, placement.centerY, placement.widthFraction].every(Number.isFinite)
    || placement.centerX < 0 || placement.centerX > 1 || placement.centerY < 0 || placement.centerY > 1
    || placement.widthFraction <= 0 || placement.widthFraction > 1)
    throw new Error("Mannequin placement requires normalized centers and an image width fraction greater than zero and at most one.");
  const scale = Math.min(mannequinWidth * placement.widthFraction / sampleWidth, mannequinHeight / sampleHeight);
  const width = sampleWidth * scale, height = sampleHeight * scale;
  return {
    x: Math.max(0, Math.min(mannequinWidth - width, mannequinWidth * placement.centerX - width / 2)),
    y: Math.max(0, Math.min(mannequinHeight - height, mannequinHeight * placement.centerY - height / 2)),
    width, height, scale,
  };
}

/** The same sample rectangle used by drawImage maps its label anchors into mannequin pixels. */
export function projectMannequinAnchors(anchors: HudAnchor[], layout: MannequinSampleLayout): HudAnchor[] {
  return anchors.flatMap(anchor => {
    const x = anchor.x * layout.scale, y = anchor.y * layout.scale;
    return Number.isFinite(x) && Number.isFinite(y) && x >= 0 && x <= layout.width && y >= 0 && y <= layout.height
      ? [{ ...anchor, x: layout.x + x, y: layout.y + y }] : [];
  });
}

/**
 * A flat photo composite for the existing marker plane. The prepared mannequin's
 * transparent background stays transparent beneath the sample.
 * Inputs remain caller-owned; disposal releases only these two new canvases.
 */
export function composeMannequinAsset(sample: ImageOverlayAsset, mannequin: CanvasImageSource, width: number, height: number,
  placement: MannequinSamplePlacement = DEFAULT_MANNEQUIN_SAMPLE_PLACEMENT): ImageOverlayAsset {
  const layout = mannequinSampleLayout(sample.width, sample.height, width, height, placement);
  for (const view of ["scene", "anatomy"] as const) {
    if (sample[view].canvas.width !== sample.width || sample[view].canvas.height !== sample.height)
      throw new Error("Sample texture dimensions disagree with its original image.");
  }
  const created: HTMLCanvasElement[] = [];
  const dispose = () => { for (const canvas of created) { canvas.width = 1; canvas.height = 1; } };
  const compose = (view: ImageOverlayView): ImageOverlayLayer => {
    const canvas = document.createElement("canvas"); created.push(canvas);
    canvas.width = width; canvas.height = height;
    const context = canvas.getContext("2d");
    if (!context) throw new Error("Mannequin image composition needs canvas rendering.");
    context.drawImage(mannequin, 0, 0, width, height);
    context.drawImage(sample[view].canvas, layout.x, layout.y, layout.width, layout.height);
    return { canvas, anchors: projectMannequinAnchors(sample[view].anchors, layout) };
  };
  try {
    return { id: sample.id, width, height, sourceLabel: `Mannequin composite · ${sample.sourceLabel}`,
      credit: sample.credit, scene: compose("scene"), anatomy: compose("anatomy"), dispose };
  } catch (cause) { dispose(); throw cause; }
}
