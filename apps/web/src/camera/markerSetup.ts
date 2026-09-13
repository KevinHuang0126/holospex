import aruco from "js-aruco2";
import { parseModelRegistration, type ModelRegistration } from "./modelRegistration";
const { AR } = aruco;

/** Approximate intrinsics and arbitrary points for checking the marker path only. */
export function createMarkerTestRegistration(width = 640, height = 480, markerSizeMm = 80): ModelRegistration {
  return parseModelRegistration({
    modelId: "marker-tracking-test", provenance: "synthetic_mock",
    dictionary: "ARUCO_MIP_36h12", markerId: 7, markerSizeMm,
    calibration: { width, height, fx: width, fy: width, cx: width / 2, cy: height / 2 },
    anchors: [
      { id: "test-center", structureId: "gallbladder", positionMm: [0, 0, 0] },
      { id: "test-right", structureId: "cystic_duct", positionMm: [markerSizeMm * 0.65, 0, 0] },
      { id: "test-up", structureId: "cystic_artery", positionMm: [0, markerSizeMm * 0.65, 0] },
    ],
  });
}

/** The dictionary SVG has eight cells of marker and one white cell on each side. */
export function printableMarkerSvg(config: ModelRegistration): string {
  const validated = parseModelRegistration(config);
  const extentMm = validated.markerSizeMm * 10 / 8;
  return new AR.Dictionary(validated.dictionary).generateSVG(validated.markerId)
    .replace("<svg ", `<svg width="${extentMm}mm" height="${extentMm}mm" shape-rendering="crispEdges" `);
}

export function downloadLocalFile(content: string, name: string, type: string) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const link = document.createElement("a"); link.href = url; link.download = name; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
