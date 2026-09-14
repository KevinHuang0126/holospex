import assert from "node:assert/strict";
import test from "node:test";
import { polygonAnchor } from "../src/overlays/polygonAnchor";
import { drawHud, type HudScene } from "../src/overlays/drawHud";

type Point = [number, number];
const concave: Point[] = [[0, 0], [100, 0], [100, 20], [20, 20], [20, 80], [100, 80], [100, 100], [0, 100]];
function inside(point: { x: number; y: number }, polygon: Point[]) {
  let result = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const a = polygon[j], b = polygon[i];
    if ((a[1] > point.y) !== (b[1] > point.y) && point.x < (b[0] - a[0]) * (point.y - a[1]) / (b[1] - a[1]) + a[0]) result = !result;
  }
  return result;
}

test("curved and concave anatomy pointers terminate inside their own contour, independent of vertex order", () => {
  const average = { x: concave.reduce((s, p) => s + p[0], 0) / concave.length, y: concave.reduce((s, p) => s + p[1], 0) / concave.length };
  assert.equal(inside(average, concave), false, "This contour reproduces the old background-targeting pointer");
  for (const polygon of [concave, [[0, 0], [120, 40], [118, 44], [0, 4]] as Point[], [[0, 0], [2, 0], [2, 1000], [0, 1000]] as Point[]]) {
    const anchor = polygonAnchor(polygon)!;
    assert.ok(inside(anchor, polygon));
    assert.deepEqual(polygonAnchor([...polygon].reverse()), anchor);
    assert.deepEqual(polygonAnchor([...polygon.slice(2), ...polygon.slice(0, 2)]), anchor);
    const resized = polygon.map(([x, y]): Point => [x * 4 + 31, y * 4 + 42]);
    const next = polygonAnchor(resized)!;
    assert.ok(inside(next, resized));
    assert.ok(Math.abs(next.x - (anchor.x * 4 + 31)) < 1e-8);
    assert.ok(Math.abs(next.y - (anchor.y * 4 + 42)) < 1e-8);
  }
});

test("degenerate or invalid contours cannot create a pointer", () => {
  for (const polygon of [[], [[0, 0], [2, 2]], [[0, 0], [1, 1], [2, 2]], [[0, NaN], [1, 1], [3, 2]]] as Point[][]) assert.equal(polygonAnchor(polygon), null);
});

function drawing(width: number) {
  const calls: { name: string; args: unknown[] }[] = [];
  const context = new Proxy({}, { get: (_, name: string) => (...args: unknown[]) => {
    calls.push({ name, args });
    if (name === "measureText") return { width: String(args[0]).length * 9 };
  } });
  const canvas = { width: 0, height: 0, clientWidth: width, getContext: () => context } as unknown as HTMLCanvasElement;
  return { canvas, calls };
}
const scene: HudScene = { width: 1280, height: 720, labelSlots: 6, sourceLabel: "ML prediction", source: "ml_prediction", warning: null,
  structures: [{ instanceId: "cystic_duct-0", structureId: "cystic_duct", polygon: concave, confidence: 0.82, visibility: "visible" }] };

test("leaders leave image clipping before crossing letterboxing and hiding labels keeps the image fixed", () => {
  for (const width of [360, 960]) {
    const { canvas, calls } = drawing(width);
    const shown = drawHud(canvas, {} as CanvasImageSource, { ...scene, appearance: { showBoundaries: false, fillOpacity: 0 } })!;
    const leader = calls.findIndex(call => call.name === "lineTo" && Number(call.args[0]) > 100);
    assert.ok(leader > calls.findIndex(call => call.name === "restore"));
    const anchor = polygonAnchor(concave)!;
    assert.ok(calls.some(call => call.name === "arc" && call.args[2] === 4 && call.args[0] === shown.image.x + anchor.x * shown.image.scale && call.args[1] === shown.image.y + anchor.y * shown.image.scale));
    calls.length = 0;
    assert.deepEqual(drawHud(canvas, {} as CanvasImageSource, { ...scene, appearance: { showLabels: false, showBoundaries: false, fillOpacity: 0 } }), shown);
    assert.ok(!calls.some(call => call.name === "arc" || (call.name === "fillText" && call.args[0] === "Cystic duct")));
  }
});

test("compact scores are opt-in and require prediction provenance; anchor and annotation labels get no invented confidence", () => {
  const { canvas, calls } = drawing(960);
  for (const source of [undefined, "reviewed_annotation", "synthetic_mock", "ml_prediction", "propagated_prediction"] as const) {
    for (const enabled of [false, true]) {
      calls.length = 0;
      drawHud(canvas, null, { ...scene, source, appearance: { showConfidence: enabled } });
      assert.equal(calls.some(call => call.name === "fillText" && call.args[0] === "Model score 0.82"), enabled && (source === "ml_prediction" || source === "propagated_prediction"));
    }
  }
  calls.length = 0;
  drawHud(canvas, null, { ...scene, anchors: [{ id: "cystic_duct-0", structureId: "cystic_duct", x: 20, y: 20 }], appearance: { showConfidence: true } });
  assert.ok(!calls.some(call => call.name === "fillText" && String(call.args[0]).startsWith("Model score")));
});

test("fragmented predictions retain every outline but only label the largest component of each class", () => {
  const { canvas, calls } = drawing(960);
  const small: Point[] = [[300, 300], [304, 300], [304, 304], [300, 304]];
  const structures = [
    { ...scene.structures[0], instanceId: "speck", polygon: small },
    ...scene.structures,
    { ...scene.structures[0], structureId: "gallbladder" as const, instanceId: "gallbladder-0", polygon: small },
  ];
  const layout = drawHud(canvas, null, { ...scene, structures })!;
  assert.equal(calls.filter(call => call.name === "closePath").length, 3, "All supplied contours are retained");
  const names = calls.filter(call => call.name === "fillText" && ["Gallbladder", "Cystic duct"].includes(String(call.args[0])));
  assert.deepEqual(names.map(call => call.args[0]), ["Gallbladder", "Cystic duct"]);
  const anchor = polygonAnchor(concave)!;
  assert.ok(calls.some(call => call.name === "arc" && call.args[2] === 4 && call.args[0] === layout.image.x + anchor.x * layout.image.scale && call.args[1] === layout.image.y + anchor.y * layout.image.scale));
  assert.deepEqual(drawHud(canvas, null, { ...scene, structures: [...structures, ...Array.from({ length: 20 }, (_, i) => ({ ...structures[0], instanceId: `speck-${i}` }))] }), layout);
});
