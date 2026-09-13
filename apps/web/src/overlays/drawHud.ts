import { anatomy, type AnatomyId, type FrameResult } from "@holospex/contracts";

export interface HudAnchor { id: string; structureId: AnatomyId; x: number; y: number; label?: string }
export interface HudAppearance {
  fillOpacity?: number;
  showBoundaries?: boolean;
  showLabels?: boolean;
}
export interface HudScene {
  width: number; height: number;
  sourceLabel: string;
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

/** Image + geometry are committed together; labels occupy a separate opaque rail. */
export function drawHud(canvas: HTMLCanvasElement, image: CanvasImageSource | null, scene: HudScene) {
  const allLabels: HudAnchor[] = scene.anchors ?? scene.structures.map(item => ({ id: item.instanceId, structureId: item.structureId,
    x: item.polygon.reduce((sum, point) => sum + point[0], 0) / item.polygon.length,
    y: item.polygon.reduce((sum, point) => sum + point[1], 0) / item.polygon.length }));
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
  labels.forEach((label, index) => {
    const x = rect.x + label.x * rect.scale, y = rect.y + label.y * rect.scale;
    context.beginPath(); context.moveTo(x, y); context.lineTo(stacked ? Math.min(areaWidth - 12, 24 + index * 28) : areaWidth, stacked ? areaHeight : 132 + index * 62);
    context.strokeStyle = "#06151a"; context.lineWidth = 4; context.stroke();
    context.strokeStyle = anatomy[label.structureId].color; context.lineWidth = 1.5; context.stroke();
    context.beginPath(); context.arc(x, y, 4, 0, Math.PI * 2); context.fillStyle = anatomy[label.structureId].color; context.fill();
  });
  context.restore();
  context.fillStyle = "#10262e"; context.fillRect(railX, railY, railWidth, height - railY);
  context.fillStyle = "#ffffff"; context.font = "bold 19px system-ui";
  let y = wrapText(context, scene.sourceLabel, railX + 18, railY + 30, railWidth - 36, 24);
  context.font = "14px system-ui"; context.fillStyle = "#bbd0d5";
  if (scene.statusLabel) wrapText(context, scene.statusLabel, railX + 18, y + 12, railWidth - 36, 20);
  labels.forEach((label, index) => {
    context.fillStyle = anatomy[label.structureId].color; context.font = "bold 18px system-ui";
    wrapText(context, label.label ?? anatomy[label.structureId].label, railX + 18, railY + 136 + index * 62, railWidth - 36, 22);
  });
  if (scene.warning) {
    const warningY = railY + Math.max(175, labels.length * 62 + 155);
    context.fillStyle = "#4d3410"; context.fillRect(railX + 10, warningY - 22, railWidth - 20, height - warningY + 12);
    context.fillStyle = "#ffe7a3"; context.font = "bold 17px system-ui";
    wrapText(context, `! ${scene.warning}`, railX + 20, warningY, railWidth - 40, 24);
  }
  return { image: rect, width: displayWidth, height };
}
