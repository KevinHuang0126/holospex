import assert from "node:assert/strict";
import test from "node:test";
import { imagePixelToPlanePoint, imagePlaneLayout, projectImagePlaneAnchors, projectScreenAnchors, screenImagePlacement } from "../src/camera/ImagePlaneRenderer";
import { sceneMatrices } from "../src/camera/cameraProjection";
import { estimateModelPose, projectPoint, type ModelPose } from "../src/camera/modelRegistration";
import { createMarkerTestRegistration } from "../src/camera/markerSetup";
import type { HudAnchor } from "../src/overlays/drawHud";

const anchors: HudAnchor[] = [
  { id: "duct", structureId: "cystic_duct", x: 420, y: 280 },
  { id: "artery", structureId: "cystic_artery", x: 750, y: 480 },
];

test("an image plane preserves top-left orientation, aspect and marker clearance", () => {
  const layout = imagePlaneLayout(1280, 720, 320, 80);
  assert.equal(layout.width / layout.height, 1280 / 720);
  assert.equal(layout.bottomY - 80 / 2, 10);
  const corners = [[0, 0], [1280, 0], [1280, 720], [0, 720]].map(([x, y]) => imagePixelToPlanePoint(x, y, 1280, 720, layout).toArray());
  assert.deepEqual(corners, [[-160, 230, 0.5], [160, 230, 0.5], [160, 50, 0.5], [-160, 50, 0.5]]);
  const portrait = imagePlaneLayout(720, 1280, 180, 100);
  assert.equal(portrait.height, 320);
  assert.equal(portrait.bottomY, 60);
  assert.throws(() => imagePlaneLayout(0, 720, 320, 80));
});

test("mask anchors stay on the source image through marker tilt and camera resizing", () => {
  const config = createMarkerTestRegistration(1280, 720, 80);
  config.calibration = { width: 1280, height: 720, fx: 1100, fy: 1030, cx: 613, cy: 341 };
  const layout = imagePlaneLayout(1280, 720, 320, 80);
  for (const angle of [-0.38, 0.19, 0.46]) {
    const rotation = [[1, 0, 0], [0, Math.cos(angle), -Math.sin(angle)], [0, Math.sin(angle), Math.cos(angle)]];
    const translation = [16, -90, 800];
    const corners = [[-40, 40, 0], [40, 40, 0], [40, -40, 0], [-40, -40, 0]].map(point => {
      const pixel = projectPoint(point, rotation, translation, config.calibration)!;
      return { x: pixel[0] / 2, y: pixel[1] / 2 };
    });
    const pose = estimateModelPose(config, corners, 640, 360);
    assert.ok(pose);
    for (const [width, height] of [[1280, 720], [640, 360], [960, 540]]) {
      const projected = projectImagePlaneAnchors(anchors, 1280, 720, pose, width, height, layout);
      assert.equal(projected.length, 2);
      // These are independently measured source-pixel positions at 0.25 mm/pixel,
      // converted to POSIT's z convention rather than Three's scene coordinates.
      const positions = [[-55, 160, -0.5], [27.5, 110, -0.5]];
      positions.forEach((position, index) => {
        const expected = projectPoint(position, pose.rotation, pose.translation, pose.calibration)!;
        assert.ok(Math.abs(projected[index].x - expected[0] * width / 640) < 0.00001);
        assert.ok(Math.abs(projected[index].y - expected[1] * height / 360) < 0.00001);
        assert.equal(projected[index].structureId, anchors[index].structureId);
      });
      const determinant = sceneMatrices(pose, width, height).model.determinant();
      assert.ok(Math.abs(determinant - 1) < 0.001, "The image must not be reflected by the camera conversion");
    }
  }
});

test("labels disappear when the image is behind the camera, outside view, or has invalid pixels", () => {
  const pose: ModelPose = {
    rotation: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], translation: [0, -140, 800], reprojectionError: 0,
    calibration: { width: 1280, height: 720, fx: 1100, fy: 1100, cx: 640, cy: 360 },
  };
  const layout = imagePlaneLayout(1280, 720, 320, 80);
  assert.equal(projectImagePlaneAnchors(anchors, 1280, 720, pose, 1280, 720, layout).length, 2);
  for (const translation of [[0, 0, -800], [5000, 0, 800], [0, 0, 20000]]) {
    assert.deepEqual(projectImagePlaneAnchors(anchors, 1280, 720, { ...pose, translation }, 1280, 720, layout), []);
  }
  const malformed = [
    { ...anchors[0], x: NaN }, { ...anchors[0], y: Infinity },
    { ...anchors[0], x: -1 }, { ...anchors[0], y: 721 },
  ];
  assert.deepEqual(projectImagePlaneAnchors(malformed, 1280, 720, pose, 1280, 720, layout), []);
});

test("screen overlay and labels remain centered and contained on portrait and landscape cameras", () => {
  for (const [outputWidth, outputHeight] of [[390, 844], [1280, 720], [480, 640]]) {
    for (const [imageWidth, imageHeight] of [[1280, 720], [720, 1280]]) {
      const rect = screenImagePlacement(imageWidth, imageHeight, outputWidth, outputHeight);
      assert.ok(rect.width <= outputWidth * 0.8 + 0.000001);
      assert.ok(rect.height <= outputHeight * 0.8 + 0.000001);
      assert.equal(rect.x + rect.width / 2, outputWidth / 2);
      assert.equal(rect.y + rect.height / 2, outputHeight / 2);
      assert.ok(Math.abs(rect.width / rect.height - imageWidth / imageHeight) < 0.000001);
      const points = projectScreenAnchors([
        { ...anchors[0], x: 0, y: 0 }, { ...anchors[1], x: imageWidth, y: imageHeight },
        { ...anchors[0], x: imageWidth / 2, y: imageHeight / 2 }, { ...anchors[1], x: -1, y: 0 },
      ], rect);
      assert.equal(points.length, 3);
      assert.equal(points[0].x, rect.x); assert.equal(points[0].y, rect.y);
      assert.equal(points[1].x, rect.x + rect.width); assert.equal(points[1].y, rect.y + rect.height);
      assert.equal(points[2].x, outputWidth / 2); assert.equal(points[2].y, outputHeight / 2);
    }
  }
  assert.throws(() => screenImagePlacement(1280, 720, 390, 844, 1.1));
  assert.throws(() => screenImagePlacement(1280, 720, 0, 844));
});
