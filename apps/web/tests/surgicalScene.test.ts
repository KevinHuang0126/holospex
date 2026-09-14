import assert from "node:assert/strict";
import test from "node:test";
import { Box3, Mesh, Vector3 } from "three";
import { createSurgicalRegistration, createSurgicalScene, disposeSurgicalScene, surgicalLandmarks } from "../src/camera/surgicalScene";
import { estimateModelPose, projectModelAnchors, projectPoint } from "../src/camera/modelRegistration";
import { sceneMatrices } from "../src/camera/cameraProjection";
import { drawHud } from "../src/overlays/drawHud";

test("the tabletop asset contains a 3D open torso and all three named anatomy meshes", () => {
  const scene = createSurgicalScene();
  for (const name of ["Open torso shell", "Abdominal cavity floor", "Sterile teal drape", "Gallbladder", "Cystic duct", "Cystic artery", "Liver right lobe", "Retractor blade"]) {
    assert.ok(scene.getObjectByName(name) instanceof Mesh, name);
  }
  const box = new Box3().setFromObject(scene), size = box.getSize(new Vector3());
  assert.ok(size.z > 50, "The scene must have volume, not be a flat image");
  assert.ok(box.min.y > 40, "The model must leave the black 80 mm marker uncovered");
  assert.ok(box.min.z >= 0, "Model cannot sit beneath the table");
  scene.traverse(object => {
    if (object instanceof Mesh) assert.ok(Array.from(object.geometry.getAttribute("position").array).every(Number.isFinite));
  });
  const config = createSurgicalRegistration(1280, 720);
  assert.equal(config.provenance, "synthetic_mock");
  assert.deepEqual(config.anchors.map(a => a.structureId), ["gallbladder", "cystic_duct", "cystic_artery"]);
  disposeSurgicalScene(scene);
});

test("the 3D body and HUD labels share calibrated projection across marker tilt and resized camera frames", () => {
  const config = createSurgicalRegistration(1280, 720);
  config.calibration = { width: 1280, height: 720, fx: 1100, fy: 1040, cx: 622, cy: 352 };
  for (const angle of [-0.32, 0.2, 0.48]) {
    const rotation = [[1, 0, 0], [0, Math.cos(angle), -Math.sin(angle)], [0, Math.sin(angle), Math.cos(angle)]];
    const translation = [20, -120, 850];
    const half = config.markerSizeMm / 2;
    const corners = [[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]].map(p => {
      const projected = projectPoint(p, rotation, translation, config.calibration)!;
      return { x: projected[0] / 2, y: projected[1] / 2 };
    });
    const pose = estimateModelPose(config, corners, 640, 360);
    assert.ok(pose);
    const labels = projectModelAnchors(config, pose);
    assert.equal(labels.length, 3);
    for (const [width, height] of [[1280, 720], [640, 360], [960, 540]]) {
      const matrices = sceneMatrices(pose, width, height);
      assert.ok(Math.abs(matrices.model.determinant() - 1) < 0.001, "Coordinate conversion must not mirror the body");
      surgicalLandmarks.forEach(({ id, position }) => {
        const point = new Vector3(...position).applyMatrix4(matrices.model).applyMatrix4(matrices.projection);
        const label = labels.find(a => a.id === id)!;
        assert.ok(Math.abs((point.x + 1) * width / 2 - label.x * width / 640) < 0.00001);
        assert.ok(Math.abs((1 - point.y) * height / 2 - label.y * height / 360) < 0.00001);
        assert.ok(point.z > -1 && point.z < 1, "Anatomy must lie within the camera clipping range");
      });
    }
  }
});

test("composed camera frames layer the 3D scene beneath labels and clear it without shifting the image", () => {
  const calls: { name: string; args: unknown[] }[] = [];
  const context = new Proxy({}, { get: (_, name: string) => (...args: unknown[]) => {
    calls.push({ name, args });
    if (name === "measureText") return { width: String(args[0]).length * 9 };
  } });
  const cameraImage = {} as CanvasImageSource, modelImage = {} as CanvasImageSource;
  for (const width of [360, 1100]) {
    const canvas = { width: 0, height: 0, clientWidth: width, getContext: () => context } as unknown as HTMLCanvasElement;
    const scene = { width: 1280, height: 720, sourceLabel: "Synthetic 3D scene", structures: [], labelSlots: 3, warning: null };
    calls.length = 0;
    const shown = drawHud(canvas, cameraImage, { ...scene, modelLayer: modelImage, anchors: [{ id: "gallbladder", structureId: "gallbladder", x: 600, y: 350 }] });
    assert.deepEqual(calls.filter(c => c.name === "drawImage").map(c => c.args[0]), [cameraImage, modelImage]);
    assert.ok(calls.findIndex(c => c.name === "drawImage" && c.args[0] === modelImage) < calls.findIndex(c => c.name === "lineTo"));
    calls.length = 0;
    const lost = drawHud(canvas, cameraImage, { ...scene, anchors: [], warning: "Tracking lost." });
    assert.deepEqual(lost, shown);
    assert.deepEqual(calls.filter(c => c.name === "drawImage").map(c => c.args[0]), [cameraImage]);
    assert.ok(!calls.some(c => c.name === "lineTo" || c.args[0] === "Gallbladder"));
  }
});
