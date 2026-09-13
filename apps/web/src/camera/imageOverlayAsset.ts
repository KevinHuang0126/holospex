import { buildDatasetRaster } from "../overlays/datasetRaster";
import { decodeIndexMask } from "../overlays/decodeIndexMask";
import { datasetCredit, datasetSourceLabel, type DatasetSample, type LoadedDatasetSample } from "../overlays/datasetSamples";
import type { HudAnchor } from "../overlays/drawHud";

export type ImageOverlayView = "scene" | "anatomy";
export interface ImageOverlayLayer {
  canvas: HTMLCanvasElement;
  /** Original-image pixel coordinates, before the camera placement transform. */
  anchors: HudAnchor[];
}
export interface ImageAnatomyRegion {
  structureId: string;
  /** Normalized original-image pixel boundaries, including the last occupied pixel's far edge. */
  bounds: { x: number; y: number; width: number; height: number };
  /** Mean of occupied pixel centers, normalized to the original image; may lie in a mask hole. */
  centroid: { x: number; y: number };
  pixelCount: number;
}
export interface ImageOverlayAsset {
  id: string;
  width: number;
  height: number;
  sourceLabel: string;
  credit: typeof datasetCredit;
  scene: ImageOverlayLayer;
  anatomy: ImageOverlayLayer;
  regions?: ImageAnatomyRegion[];
  dispose(): void;
}

/** Keep the photographed tissue visible. Masks supply tint and contours, never replacement anatomy. */
export function composeImageOverlayPixels(sample: DatasetSample, original: Uint8ClampedArray, mask: Uint8ClampedArray) {
  const { width, height } = sample.frame;
  if (original.length !== width * height * 4) throw new Error("Original image dimensions disagree with the label record.");
  const raster = buildDatasetRaster(sample, mask);
  const regionStats = new Map(sample.classes.filter(item => item.structureId !== "background" && item.structureId !== "tool")
    .map(item => [item.index, { structureId: item.structureId, minX: width, minY: height, maxX: -1, maxY: -1,
      sumX: 0, sumY: 0, pixelCount: 0 }]));
  const scene = new Uint8ClampedArray(original.length), cutout = new Uint8ClampedArray(original.length);
  for (let pixel = 0; pixel < raster.pixels.length; pixel++) {
    const offset = pixel * 4;
    if (original[offset + 3] !== 255) throw new Error("The original surgical JPEG must decode as an opaque image.");
    const opacity = raster.outline[offset + 3] ? 0.85 : raster.fill[offset + 3] ? 0.12 : 0;
    for (let channel = 0; channel < 3; channel++) {
      scene[offset + channel] = original[offset + channel] * (1 - opacity) + raster.fill[offset + channel] * opacity;
    }
    scene[offset + 3] = 255;
    const region = regionStats.get(raster.pixels[pixel]);
    if (region) {
      cutout.set(scene.subarray(offset, offset + 4), offset);
      const x = pixel % width, y = Math.floor(pixel / width);
      region.minX = Math.min(region.minX, x); region.maxX = Math.max(region.maxX, x);
      region.minY = Math.min(region.minY, y); region.maxY = Math.max(region.maxY, y);
      region.sumX += x; region.sumY += y; region.pixelCount++;
    }
  }
  const regions: ImageAnatomyRegion[] = [...regionStats.values()].filter(region => region.pixelCount > 0).map(region => ({
    structureId: region.structureId,
    bounds: { x: region.minX / width, y: region.minY / height,
      width: (region.maxX - region.minX + 1) / width, height: (region.maxY - region.minY + 1) / height },
    centroid: { x: (region.sumX / region.pixelCount + 0.5) / width, y: (region.sumY / region.pixelCount + 0.5) / height },
    pixelCount: region.pixelCount,
  }));
  return {
    regions,
    scene: { rgba: scene, anchors: raster.anchors },
    anatomy: { rgba: cutout, anchors: raster.anchors.filter(anchor => anchor.structureId !== "tool") },
  };
}

function abortIfRequested(signal?: AbortSignal) {
  if (signal?.aborted) throw new DOMException("Image preparation was cancelled.", "AbortError");
}

/** Decode photographic color at native resolution. Label IDs use decodeIndexMask instead. */
async function decodeOriginalPixels(blob: Blob, width: number, height: number, signal?: AbortSignal) {
  abortIfRequested(signal);
  let source: ImageBitmap | HTMLImageElement | null = null;
  let objectUrl: string | null = null;
  let canvas: HTMLCanvasElement | null = null;
  try {
    if (typeof createImageBitmap === "function") {
      source = await createImageBitmap(blob);
    } else {
      const image = new Image();
      source = image;
      objectUrl = URL.createObjectURL(blob);
      await new Promise<void>((resolve, reject) => {
        image.onload = () => resolve();
        image.onerror = () => reject(new Error("Could not decode the original image."));
        image.src = objectUrl!;
      });
    }
    abortIfRequested(signal);
    const actualWidth = "naturalWidth" in source ? source.naturalWidth : source.width;
    const actualHeight = "naturalHeight" in source ? source.naturalHeight : source.height;
    if (actualWidth !== width || actualHeight !== height) {
      throw new Error("Original image dimensions disagree with the label record.");
    }
    canvas = document.createElement("canvas"); canvas.width = width; canvas.height = height;
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (!context) throw new Error("Canvas rendering is unavailable.");
    context.drawImage(source, 0, 0);
    return context.getImageData(0, 0, width, height).data;
  } finally {
    if (source && "close" in source) source.close();
    if (source && "onload" in source) { source.onload = null; source.onerror = null; source.src = ""; }
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    if (canvas) { canvas.width = 1; canvas.height = 1; }
  }
}

function pixelCanvas(width: number, height: number, rgba: Uint8ClampedArray) {
  const canvas = document.createElement("canvas"); canvas.width = width; canvas.height = height;
  const context = canvas.getContext("2d");
  if (!context) throw new Error("Canvas rendering is unavailable.");
  const data = context.createImageData(width, height); data.data.set(rgba); context.putImageData(data, 0, 0);
  return canvas;
}

/** Importer-validated JPEG + exact index PNG become one indivisible camera texture asset. */
export async function loadImageOverlayAsset(sample: LoadedDatasetSample, signal?: AbortSignal): Promise<ImageOverlayAsset> {
  const { width, height } = sample.labels.frame;
  const original = await decodeOriginalPixels(sample.image, width, height, signal);
  const mask = await decodeIndexMask(sample.indexMask, width, height, signal);
  abortIfRequested(signal);
  const pixels = composeImageOverlayPixels(sample.labels, original, mask);
  const canvases: HTMLCanvasElement[] = [];
  const dispose = () => { for (const canvas of canvases) { canvas.width = 1; canvas.height = 1; } };
  try {
    const scene = pixelCanvas(width, height, pixels.scene.rgba); canvases.push(scene);
    const anatomy = pixelCanvas(width, height, pixels.anatomy.rgba); canvases.push(anatomy);
    return { id: sample.id, width, height, sourceLabel: datasetSourceLabel, credit: datasetCredit, regions: pixels.regions,
      scene: { canvas: scene, anchors: pixels.scene.anchors },
      anatomy: { canvas: anatomy, anchors: pixels.anatomy.anchors }, dispose };
  } catch (error) { dispose(); throw error; }
}
