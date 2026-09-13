import assert from "node:assert/strict";
import test from "node:test";
import { composeMannequinAsset, DEFAULT_MANNEQUIN_SAMPLE_PLACEMENT, mannequinSampleLayout, projectMannequinAnchors } from "../src/camera/mannequinComposite";
import type { ImageOverlayAsset } from "../src/camera/imageOverlayAsset";
import { imagePlaneLayout, projectImagePlaneAnchors } from "../src/camera/ImagePlaneRenderer";
import { projectPoint, type ModelPose } from "../src/camera/modelRegistration";
import { datasetCredit } from "../src/overlays/datasetSamples";
import type { HudAnchor } from "../src/overlays/drawHud";

const close = (actual: number, expected: number) => assert.ok(Math.abs(actual - expected) < 0.000001, `${actual} != ${expected}`);
const anchors: HudAnchor[] = [
  { id: "duct", structureId: "cystic_duct", x: 427, y: 240 },
  { id: "artery", structureId: "cystic_artery", x: 650, y: 300 },
];

test("sample pixels and anchors remain registered through mannequin composition, marker tilt and camera resize", () => {
  const rect = mannequinSampleLayout(854, 480, 894, 569, { centerX: 0.385, centerY: 0.48, widthFraction: 0.20 });
  const transformed = projectMannequinAnchors(anchors, rect);
  close(rect.width / rect.height, 854 / 480);
  close(transformed[0].x, 344.19); close(transformed[0].y, 273.12);
  for (let index = 0; index < anchors.length; index++) {
    close((transformed[index].x - rect.x) / rect.scale, anchors[index].x);
    close((transformed[index].y - rect.y) / rect.scale, anchors[index].y);
    assert.equal(transformed[index].structureId, anchors[index].structureId);
  }
  const plane = imagePlaneLayout(894, 569, 320, 80);
  for (const angle of [-0.38, 0.19, 0.46]) {
    const pose: ModelPose = {
      rotation: [[1, 0, 0], [0, Math.cos(angle), -Math.sin(angle)], [0, Math.sin(angle), Math.cos(angle)]],
      translation: [16, -90, 800], reprojectionError: 0,
      calibration: { width: 1280, height: 720, fx: 1100, fy: 1030, cx: 613, cy: 341 },
    };
    // The middle sample pixel lies at the requested normalized mannequin center.
    // Convert that independently to marker millimeters, then the physical camera.
    const expected = projectPoint([-36.8, 50 + 0.52 * 320 * 569 / 894, -0.5], pose.rotation, pose.translation, pose.calibration)!;
    for (const [width, height] of [[1280, 720], [640, 360], [960, 540]]) {
      const projected = projectImagePlaneAnchors(transformed, 894, 569, pose, width, height, plane);
      assert.equal(projected.length, 2);
      close(projected[0].x, expected[0] * width / 1280); close(projected[0].y, expected[1] * height / 720);
    }
  }
});

test("edge placement preserves the entire sample for tall and wide aspect ratios", () => {
  for (const [sampleWidth, sampleHeight] of [[854, 480], [100, 2000], [2000, 100]]) {
    for (const [centerX, centerY] of [[0, 0], [1, 1], [0, 1], [1, 0], [0.5, 0.5]]) {
      const rect = mannequinSampleLayout(sampleWidth, sampleHeight, 894, 569, { centerX, centerY, widthFraction: 1 });
      assert.ok(rect.x >= 0 && rect.y >= 0 && rect.x + rect.width <= 894.000001 && rect.y + rect.height <= 569.000001);
      close(rect.width / rect.height, sampleWidth / sampleHeight);
      const corners = projectMannequinAnchors([
        { ...anchors[0], x: 0, y: 0 }, { ...anchors[1], x: sampleWidth, y: sampleHeight },
        { ...anchors[0], x: -1 }, { ...anchors[0], x: sampleWidth + 1 }, { ...anchors[0], y: NaN },
      ], rect);
      assert.equal(corners.length, 2);
      close(corners[0].x, rect.x); close(corners[0].y, rect.y);
      close(corners[1].x, rect.x + rect.width); close(corners[1].y, rect.y + rect.height);
    }
  }
  for (const [width, height] of [[0, 480], [854, NaN], [1.5, 480], [8193, 480], [4001, 4000]]) assert.throws(() => mannequinSampleLayout(width, height, 894, 569), /dimensions/);
  for (const placement of [{ centerX: NaN }, { centerY: -1 }, { centerX: 1.1 }, { widthFraction: 0 }, { widthFraction: 1.1 }])
    assert.throws(() => mannequinSampleLayout(854, 480, 894, 569, { ...DEFAULT_MANNEQUIN_SAMPLE_PLACEMENT, ...placement }), /normalized/);
});

test("both composites draw the original mannequin before their matching sample and own only output canvases", () => {
  const previous = Object.getOwnPropertyDescriptor(globalThis, "document");
  const created: { width: number; height: number; calls: unknown[][] }[] = [];
  let failSecond = false, sourceDisposed = false;
  const mannequin = { width: 894, height: 569, close: () => { throw new Error("Must not close the source mannequin."); } } as unknown as CanvasImageSource;
  const sample: ImageOverlayAsset = { id: "116_35325", width: 854, height: 480, sourceLabel: "Supplied dataset annotation", credit: datasetCredit,
    scene: { canvas: { width: 854, height: 480 } as HTMLCanvasElement, anchors },
    anatomy: { canvas: { width: 854, height: 480 } as HTMLCanvasElement, anchors: [anchors[0]] },
    dispose: () => { sourceDisposed = true; } };
  try {
    Object.defineProperty(globalThis, "document", { configurable: true, value: { createElement: () => {
      const canvas = { width: 0, height: 0, calls: [] as unknown[][],
        getContext: () => failSecond && created.length === 2 ? null : { drawImage: (...args: unknown[]) => canvas.calls.push(args) } };
      created.push(canvas); return canvas;
    } } });
    const composite = composeMannequinAsset(sample, mannequin, 894, 569);
    const rect = mannequinSampleLayout(854, 480, 894, 569);
    for (const [index, view] of (["scene", "anatomy"] as const).entries()) {
      assert.deepEqual(created[index].calls, [[mannequin, 0, 0, 894, 569], [sample[view].canvas, rect.x, rect.y, rect.width, rect.height]]);
      assert.deepEqual(composite[view].anchors, projectMannequinAnchors(sample[view].anchors, rect));
    }
    assert.equal(composite.id, sample.id); assert.equal(composite.credit, sample.credit);
    assert.match(composite.sourceLabel, /Mannequin composite.*Supplied dataset annotation/);
    composite.dispose(); composite.dispose();
    assert.ok(created.every(canvas => canvas.width === 1 && canvas.height === 1));
    assert.equal(sourceDisposed, false); assert.equal(sample.scene.canvas.width, 854); assert.equal(sample.anatomy.canvas.height, 480);
    created.length = 0; failSecond = true;
    assert.throws(() => composeMannequinAsset(sample, mannequin, 894, 569), /canvas rendering/);
    assert.equal(created.length, 2); assert.ok(created.every(canvas => canvas.width === 1 && canvas.height === 1));
    assert.equal(sourceDisposed, false);
    sample.scene.canvas.width = 1;
    assert.throws(() => composeMannequinAsset(sample, mannequin, 894, 569), /texture dimensions/);
  } finally {
    if (previous) Object.defineProperty(globalThis, "document", previous); else Reflect.deleteProperty(globalThis, "document");
  }
});
