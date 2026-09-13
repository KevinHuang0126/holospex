import { anatomy, type AnatomyId } from "@holospex/contracts";
import type { HudAnchor } from "./drawHud";
import type { HudMode } from "./hudControls";
import type { DatasetSample } from "./datasetSamples";

export type DatasetView = "raster" | "polygons";
export function datasetOverlayState(visible: boolean, mode: HudMode, view: DatasetView) {
  if (!visible || mode === "identify" || mode === "assess") return { show: false, warning: null };
  if (mode === "feedback") return { show: false, warning: "Unable to assess: reviewed lesson feedback has not been supplied for these samples." };
  return { show: true, warning: view === "polygons" ? "Approximate polygons omit some regions and holes. Use the exact mask for all supplied labels." : null };
}

/** Validate native decoded pixels before generating either labels or colored geometry. */
export function buildDatasetRaster(sample: DatasetSample, rgba: Uint8ClampedArray) {
  const { width, height } = sample.frame, size = width * height;
  if (rgba.length !== size * 4) throw new Error("Decoded mask dimensions do not match the still.");
  const pixels = new Uint8Array(size), fill = new Uint8ClampedArray(size * 4), outline = new Uint8ClampedArray(size * 4);
  const classes = new Map(sample.classes.map(item => [item.index, item.structureId]));
  const counts = new Map<number, number>(), sums = new Map<number, [number, number]>();
  const colors = new Map(sample.classes.filter(item => item.structureId !== "background").map(item => {
    const hex = anatomy[item.structureId as AnatomyId].color;
    return [item.index, [1, 3, 5].map(offset => Number.parseInt(hex.slice(offset, offset + 2), 16))];
  }));
  for (let p = 0; p < size; p++) {
    const offset = p * 4, value = rgba[offset];
    if (rgba[offset + 1] !== value || rgba[offset + 2] !== value || rgba[offset + 3] !== 255 || (!classes.has(value) && value !== 255))
      throw new Error("Index mask contains unsupported or altered pixel values.");
    pixels[p] = value; counts.set(value, (counts.get(value) ?? 0) + 1);
    const color = colors.get(value);
    if (!color) continue; // Background and ignore=255 remain transparent.
    fill.set([...color, 255], offset);
    const sum = sums.get(value) ?? [0, 0]; sum[0] += p % width; sum[1] += Math.floor(p / width); sums.set(value, sum);
  }
  for (const item of sample.classes) if (item.structureId !== "background" && (counts.get(item.index) ?? 0) !== sample.raster.pixelCounts[item.structureId])
    throw new Error(`Mask pixel counts disagree with supplied labels for ${item.structureId}.`);
  if ((counts.get(255) ?? 0) !== sample.raster.ignoredPixelCount) throw new Error("Mask ignored-pixel count disagrees with supplied labels.");
  const nearest = new Map<number, { distance: number; x: number; y: number }>();
  for (let p = 0; p < size; p++) {
    const value = pixels[p], color = colors.get(value);
    if (!color) continue;
    const x = p % width, y = Math.floor(p / width), count = counts.get(value)!, sum = sums.get(value)!;
    const distance = (x - sum[0] / count) ** 2 + (y - sum[1] / count) ** 2;
    if (distance < (nearest.get(value)?.distance ?? Infinity)) nearest.set(value, { distance, x: x + 0.5, y: y + 0.5 });
    // Keep the two-pixel contour inside supplied foreground pixels. It remains
    // readable when reduced, without coloring background or ignore=255.
    if (x < 2 || y < 2 || x >= width - 2 || y >= height - 2
      || [p - 1, p + 1, p - width, p + width, p - 2, p + 2, p - 2 * width, p + 2 * width].some(next => pixels[next] !== value))
      outline.set([...color, 255], p * 4);
  }
  const anchors: HudAnchor[] = sample.classes.flatMap(item => {
    const point = nearest.get(item.index);
    return point && item.structureId !== "background" ? [{ id: item.structureId, structureId: item.structureId, x: point.x, y: point.y }] : [];
  });
  anchors.sort((a, b) => a.y - b.y || a.x - b.x);
  return { pixels, fill, outline, anchors };
}

export function selectDatasetPixel(sample: DatasetSample, pixels: Uint8Array, point: [number, number]) {
  const [x, y] = point, { width, height } = sample.frame;
  if (!point.every(Number.isFinite) || x < 0 || y < 0 || x >= width || y >= height || pixels.length !== width * height) return null;
  const value = pixels[Math.floor(y) * width + Math.floor(x)];
  const structureId = sample.classes.find(item => item.index === value)?.structureId;
  return { structureId: structureId && structureId !== "background" ? structureId : null,
    status: value === 255 ? "ignored" as const : value === 0 ? "background" as const : "annotated" as const };
}
