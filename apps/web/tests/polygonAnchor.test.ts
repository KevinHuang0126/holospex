import assert from "node:assert/strict";
import test from "node:test";
import { anatomy, type AnatomyId } from "@holospex/contracts";
import { polygonAnchor } from "../src/overlays/polygonAnchor";
import { drawHud, HUD_LABEL_CAPACITY, type HudScene } from "../src/overlays/drawHud";

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
    const leader = calls.findIndex(call => call.name === "lineTo" && (width < 760
      ? call.args[1] === shown.image.height : call.args[0] === shown.width - 300));
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

test("every combination of visible anatomy keeps the video, class rows and warning panel fixed", () => {
  const ids = Object.keys(anatomy) as AnatomyId[];
  const structures = ids.map(structureId => ({ ...scene.structures[0], structureId, instanceId: structureId }));
  for (const width of [360, 1200]) {
    const { canvas, calls } = drawing(width);
    const render = (items: HudScene["structures"], warning: string | null) => {
      calls.length = 0;
      const layout = drawHud(canvas, null, { ...scene, labelSlots: undefined, structures: items, warning });
      const text = calls.filter(call => call.name === "fillText");
      const warningStart = text.findIndex(call => String(call.args[0]).startsWith("!"));
      return { layout, size: [canvas.width, canvas.height],
        text: warningStart < 0 ? text : text.slice(0, warningStart),
        badges: calls.filter(call => call.name === "arc" && call.args[2] === 9),
        panel: calls.filter(call => call.name === "fillRect").at(-1) };
    };
    const full = render(structures, "Partial visibility.");
    for (let mask = 0; mask < 2 ** ids.length; mask++) {
      const visible = structures.filter((_, index) => mask & (1 << index));
      const sparse = render(visible, "Low-confidence anatomy is hidden.");
      assert.deepEqual(sparse.layout, full.layout);
      assert.deepEqual(sparse.size, full.size);
      assert.deepEqual(sparse.panel, full.panel, "Warning box must not follow the last visible label");
      for (const text of sparse.text.filter(call => !String(call.args[0]).startsWith("!"))) {
        assert.ok(full.text.some(call => JSON.stringify(call) === JSON.stringify(text)), "Visible labels keep their original coordinates");
      }
      for (const badge of sparse.badges) assert.ok(full.badges.some(call => JSON.stringify(call) === JSON.stringify(badge)));
      assert.deepEqual(render(visible, null).layout, full.layout);
    }
  }
});

test("the shared twelve-label rail never resizes for caller hints, source text, hidden labels or overflow", () => {
  const anchors = Array.from({ length: 40 }, (_, index) => ({ id: `anchor-${index}`, structureId: "cystic_duct" as const,
    label: `Anchor ${String(index + 1).padStart(2, "0")}`, x: 20 + index * 10, y: 40 }));
  for (const width of [300, 335, 336, 759, 760, 1200]) {
    const { canvas, calls } = drawing(width), image = {} as CanvasImageSource;
    const baseline = drawHud(canvas, image, { ...scene, structures: [], anchors: [], labelSlots: 0 })!;
    const dimensions = [canvas.width, canvas.height];
    for (const count of [0, 1, 12, 13, 40]) {
      calls.length = 0;
      const supplied = anchors.slice(0, count);
      const layout = drawHud(canvas, image, { ...scene, anchors: supplied, labelSlots: count + 100,
        sourceLabel: "A longer supplied source description with changing frame provenance",
        statusLabel: "Frame 100 at 1000 milliseconds", warning: count % 2 ? "Partial visibility." : null });
      assert.deepEqual(layout, baseline, "Labels and legacy caller hints cannot change the image or rail geometry");
      assert.deepEqual([canvas.width, canvas.height], dimensions);
      assert.deepEqual(calls.find(call => call.name === "drawImage")?.args,
        [image, baseline.image.x, baseline.image.y, baseline.image.width, baseline.image.height]);
      const names = calls.filter(call => call.name === "fillText" && /^Anchor \d+$/.test(String(call.args[0])));
      assert.deepEqual(names.map(call => call.args[0]), supplied.slice(0, HUD_LABEL_CAPACITY).map(item => item.label));
      assert.equal(calls.filter(call => call.name === "arc" && call.args[2] === 4).length, Math.min(count, HUD_LABEL_CAPACITY));
      assert.equal(calls.filter(call => call.name === "closePath").length, scene.structures.length, "Overflow never removes image contours");
      const overflow = calls.some(call => call.name === "fillText" && String(call.args[0]).includes("Showing 12 of"));
      assert.equal(overflow, count > HUD_LABEL_CAPACITY);
      if (names.length) {
        const lastRow = Number(names.at(-1)!.args[2]);
        assert.ok(lastRow < baseline.height - 140, "All twelve rows leave room for the reserved warning area");
      }
      calls.length = 0;
      assert.deepEqual(drawHud(canvas, image, { ...scene, anchors: supplied, labelSlots: 0,
        appearance: { showLabels: false }, warning: null }), baseline);
      assert.ok(!calls.some(call => call.name === "arc" || call.name === "fillText" && /Anchor|Showing/.test(String(call.args[0]))));
    }
    calls.length = 0;
    drawHud(canvas, image, { ...scene, anchors: [...anchors.slice(0, 12), { ...anchors[12], x: scene.width + 1 }] });
    assert.ok(!calls.some(call => call.name === "fillText" && String(call.args[0]).includes("Showing")), "Invalid anchors are excluded from the overflow count");
  }
});
