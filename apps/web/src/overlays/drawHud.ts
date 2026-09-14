import { anatomy, type AnatomyId, type FrameResult } from "@holospex/contracts";
import { polygonAnchor } from "./polygonAnchor";

export interface HudAnchor { id: string; structureId: AnatomyId; x: number; y: number; label?: string }
export interface HudAppearance {
  fillOpacity?: number;
  showBoundaries?: boolean;
  showLabels?: boolean;
  showConfidence?: boolean;
}
export interface HudScene {
  width: number; height: number;
  sourceLabel: string;
  source?: FrameResult["source"];
  statusLabel?: string;
  structures: FrameResult["structures"];
  anchors?: HudAnchor[];
  /** Reserve space so hiding answers does not zoom or move the underlying image. */
  labelSlots?: number;
  /** Native-resolution raster layers, already validated against their still. */
  raster?: { fill: CanvasImageSource; outline: CanvasImageSource };
  /** An image or 3D layer composed in this camera frame's original pixel space. */
  modelLayer?: CanvasImageSource;
  appearance?: HudAppearance;
  warning: string | null;
}
export function containedRect(width: number, height: number, areaWidth: number, areaHeight: number) {
  const scale = Math.min(areaWidth / width, areaHeight / height);
  return { x: (areaWidth - width * scale) / 2, y: (areaHeight - height * scale) / 2, width: width * scale, height: height * scale, scale };
}

function wrapText(context: CanvasRenderingContext2D, text: string, x: number, y: number, maxWidth: number, lineHeight: number) {
  let line = "";
  for (const word of text.split(" ")) {
    if (line && context.measureText(`${line} ${word}`).width > maxWidth) { context.fillText(line, x, y); y += lineHeight; line = word; }
    else line = line ? `${line} ${word}` : word;
  }
  context.fillText(line, x, y);
  return y + lineHeight;
}

function structureLabels(structures: HudScene["structures"]): HudAnchor[] {
  // Keep every region's mask, but use one pointer per anatomy class. Contour
  // fragments should not duplicate names, reorder the rail or shrink the video.
  const largest = new Map<AnatomyId, { item: HudScene["structures"][number]; area: number }>();
  for (const item of structures) {
    let twiceArea = 0;
    for (let i = 0, j = item.polygon.length - 1; i < item.polygon.length; j = i++) {
      const a = item.polygon[j], b = item.polygon[i]; twiceArea += a[0] * b[1] - b[0] * a[1];
    }
    const area = Math.abs(twiceArea);
    if (area > (largest.get(item.structureId)?.area ?? 0)) largest.set(item.structureId, { item, area });
  }
  return (Object.keys(anatomy) as AnatomyId[]).flatMap(structureId => {
    const item = largest.get(structureId)?.item;
    const point = item && polygonAnchor(item.polygon);
    return item && point ? [{ id: item.instanceId, structureId, ...point }] : [];
  });
}

/** Image + geometry are committed together; labels occupy a separate opaque rail. */
export function drawHud(canvas: HTMLCanvasElement, image: CanvasImageSource | null, scene: HudScene) {
  const allLabels: HudAnchor[] = (scene.anchors ?? structureLabels(scene.structures)).filter(label => Number.isFinite(label.x) && Number.isFinite(label.y)
    && label.x >= 0 && label.x <= scene.width && label.y >= 0 && label.y <= scene.height);
  const labels = scene.appearance?.showLabels === false ? [] : allLabels;
  const boundaries = scene.appearance?.showBoundaries !== false;
  const requestedOpacity = scene.appearance?.fillOpacity;
  const fillOpacity = typeof requestedOpacity === "number" && Number.isFinite(requestedOpacity)
    ? Math.max(0, Math.min(1, requestedOpacity)) : scene.raster ? 0.28 : 0.22;
  const displayWidth = Math.max(300, canvas.clientWidth || 960);
  const stacked = displayWidth < 760;
  const railWidth = stacked ? displayWidth : 300;
  const areaWidth = stacked ? displayWidth : displayWidth - railWidth;
  const layoutLabels = Math.max(allLabels.length, scene.labelSlots ?? 0);
  const areaHeight = stacked ? areaWidth * scene.height / scene.width : Math.max(440, layoutLabels * 62 + 300);
  const railX = stacked ? 0 : areaWidth, railY = stacked ? areaHeight : 0;
  // The phone layout gets a small numbered connection strip above its header.
  // Reserve it even when labels are hidden so the underlying image stays fixed.
  const connectionHeight = stacked ? Math.ceil(layoutLabels / Math.max(1, Math.floor(areaWidth / 28))) * 24 : 0;
  const labelTop = 136 + connectionHeight;
  const height = stacked ? areaHeight + Math.max(300, layoutLabels * 62 + 300) : areaHeight;
  const density = Math.min(2, globalThis.devicePixelRatio || 1);
  if (canvas.width !== Math.round(displayWidth * density) || canvas.height !== Math.round(height * density)) { canvas.width = Math.round(displayWidth * density); canvas.height = Math.round(height * density); }
  const context = canvas.getContext("2d");
  if (!context) return;
  context.setTransform(density, 0, 0, density, 0, 0);
  const rect = containedRect(scene.width, scene.height, areaWidth, areaHeight);
  context.fillStyle = "#09181e"; context.fillRect(0, 0, displayWidth, height);
  if (image) context.drawImage(image, rect.x, rect.y, rect.width, rect.height);
  context.save();
  context.beginPath(); context.rect(rect.x, rect.y, rect.width, rect.height); context.clip();
  if (scene.modelLayer) context.drawImage(scene.modelLayer, rect.x, rect.y, rect.width, rect.height);
  if (scene.raster) {
    context.imageSmoothingEnabled = false;
    if (fillOpacity > 0) {
      context.globalAlpha = fillOpacity; context.drawImage(scene.raster.fill, rect.x, rect.y, rect.width, rect.height); context.globalAlpha = 1;
    }
    if (boundaries) {
      context.save(); context.imageSmoothingEnabled = true;
      context.shadowColor = "#041116"; context.shadowBlur = 2;
      context.drawImage(scene.raster.outline, rect.x, rect.y, rect.width, rect.height); context.restore();
    }
  }
  for (const item of scene.structures) {
    context.beginPath();
    item.polygon.forEach(([x, y], index) => { if (index === 0) context.moveTo(rect.x + x * rect.scale, rect.y + y * rect.scale); else context.lineTo(rect.x + x * rect.scale, rect.y + y * rect.scale); });
    context.closePath();
    if (fillOpacity > 0) {
      context.fillStyle = anatomy[item.structureId].color; context.globalAlpha = fillOpacity; context.fill(); context.globalAlpha = 1;
    }
    if (boundaries) {
      context.strokeStyle = "#07181f"; context.lineWidth = 5; context.stroke();
      context.strokeStyle = anatomy[item.structureId].color; context.lineWidth = 2.5;
      context.setLineDash(item.visibility === "partial" ? [8, 5] : []); context.stroke(); context.setLineDash([]);
    }
  }
  context.restore();
  // Leaders cross the letterbox to reach the rail; clipping them to just the
  // image used to make them appear detached from their corresponding labels.
  context.save(); context.beginPath(); context.rect(0, 0, areaWidth, areaHeight); context.clip();
  const ordered = labels.map((label, index) => ({ label, index })).sort((a, b) => a.label.x - b.label.x || a.index - b.index);
  const columns = Math.max(1, Math.floor(areaWidth / 28));
  const connection = (index: number) => {
    const rank = ordered.findIndex(item => item.index === index), row = Math.floor(rank / columns);
    const count = Math.min(columns, labels.length - row * columns);
    return { x: (rank % columns + 0.5) * areaWidth / count, y: railY + row * 24 + 12 };
  };
  labels.forEach((label, index) => {
    const x = rect.x + label.x * rect.scale, y = rect.y + label.y * rect.scale;
    context.beginPath(); context.moveTo(x, y); context.lineTo(stacked ? connection(index).x : areaWidth, stacked ? areaHeight : labelTop - 6 + index * 62);
    context.strokeStyle = "#06151a"; context.lineWidth = 4; context.stroke();
    context.strokeStyle = anatomy[label.structureId].color; context.lineWidth = 1.5; context.stroke();
    context.beginPath(); context.arc(x, y, 4, 0, Math.PI * 2); context.fillStyle = anatomy[label.structureId].color; context.fill();
  });
  context.restore();
  context.fillStyle = "#10262e"; context.fillRect(railX, railY, railWidth, height - railY);
  const badge = (label: HudAnchor, index: number, x: number, y: number) => {
    context.beginPath(); context.arc(x, y, 9, 0, Math.PI * 2);
    context.fillStyle = "#06151a"; context.fill();
    context.lineWidth = 1; context.strokeStyle = anatomy[label.structureId].color; context.stroke();
    context.fillStyle = "#ffffff"; context.font = "bold 11px system-ui";
    const number = String(scene.anchors ? index + 1 : Object.keys(anatomy).indexOf(label.structureId) + 1);
    context.fillText(number, x - context.measureText(number).width / 2, y + 4);
  };
  if (stacked) labels.forEach((label, index) => {
    const point = connection(index);
    context.beginPath(); context.moveTo(point.x, railY); context.lineTo(point.x, point.y);
    context.strokeStyle = anatomy[label.structureId].color; context.lineWidth = 1.5; context.stroke();
    badge(label, index, point.x, point.y);
  });
  context.fillStyle = "#ffffff"; context.font = "bold 19px system-ui";
  let y = wrapText(context, scene.sourceLabel, railX + 18, railY + 30 + connectionHeight, railWidth - 36, 24);
  context.font = "14px system-ui"; context.fillStyle = "#bbd0d5";
  if (scene.statusLabel) wrapText(context, scene.statusLabel, railX + 18, y + 12, railWidth - 36, 20);
  labels.forEach((label, index) => {
    if (!stacked) {
      context.beginPath(); context.moveTo(railX, railY + labelTop - 6 + index * 62); context.lineTo(railX + 26, railY + labelTop - 6 + index * 62);
      context.strokeStyle = anatomy[label.structureId].color; context.lineWidth = 1.5; context.stroke();
    }
    badge(label, index, railX + 26, railY + labelTop - 6 + index * 62);
    context.fillStyle = anatomy[label.structureId].color; context.font = "bold 18px system-ui";
    const afterLabel = wrapText(context, label.label ?? anatomy[label.structureId].label, railX + 44, railY + labelTop + index * 62, railWidth - 62, 22);
    const predicted = scene.source === "ml_prediction" || scene.source === "propagated_prediction";
    const score = !scene.anchors && predicted ? scene.structures.find(item => item.instanceId === label.id)?.confidence : undefined;
    if (scene.appearance?.showConfidence && typeof score === "number" && Number.isFinite(score) && score >= 0 && score <= 1) {
      context.fillStyle = "#bbd0d5"; context.font = "12px system-ui";
      context.fillText(`Model score ${score.toFixed(2)}`, railX + 44, afterLabel - 6);
    }
  });
  if (scene.warning) {
    const warningY = railY + connectionHeight + Math.max(175, labels.length * 62 + 155);
    context.fillStyle = "#4d3410"; context.fillRect(railX + 10, warningY - 22, railWidth - 20, height - warningY + 12);
    context.fillStyle = "#ffe7a3"; context.font = "bold 17px system-ui";
    wrapText(context, `! ${scene.warning}`, railX + 20, warningY, railWidth - 40, 24);
  }
  return { image: rect, width: displayWidth, height };
}
