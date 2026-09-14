import assert from "node:assert/strict";
import test from "node:test";
import aruco from "js-aruco2";
import { estimateModelPose, parseModelRegistration, projectModelAnchors, projectPoint } from "../src/camera/modelRegistration";
import { createMarkerTestRegistration, printableMarkerSvg } from "../src/camera/markerSetup";
const { AR } = aruco;

const config = parseModelRegistration({ modelId: "synthetic-test-only", provenance: "synthetic_mock", dictionary: "ARUCO_MIP_36h12", markerId: 7, markerSizeMm: 80,
  calibration: { width: 640, height: 480, fx: 640, fy: 640, cx: 320, cy: 240 },
  anchors: [{ id: "test-center", structureId: "gallbladder", positionMm: [0, 0, 0] }] });
const identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];

test("marker coordinates project with explicit calibrated focal lengths and y direction", () => {
  assert.deepEqual(projectPoint([10, 20, 0], identity, [0, 0, 640], config.calibration), [330, 220]);
  assert.equal(projectPoint([0, 0, 0], identity, [0, 0, -10], config.calibration), null);
});
test("invalid model registration is rejected before tracking", () => {
  for (const value of [null, {}, { ...config, markerSizeMm: 0 }, { ...config, markerId: 250 }, { ...config, anchors: [{ ...config.anchors[0], positionMm: [NaN, 0, 0] }] }]) assert.throws(() => parseModelRegistration(value));
});
test("a synthetic marker yields a measured pose, follows translation, and rejects wrong camera aspect", () => {
  const corners = [{ x: 240, y: 160 }, { x: 400, y: 160 }, { x: 400, y: 320 }, { x: 240, y: 320 }];
  const pose = estimateModelPose(config, corners, 640, 480);
  assert.ok(pose);
  const anchors = projectModelAnchors(config, pose);
  assert.equal(anchors.length, 1);
  assert.ok(Math.abs(anchors[0].x - 320) < 0.1 && Math.abs(anchors[0].y - 240) < 0.1);
  const translatedPose = estimateModelPose(config, corners.map(p => ({ x: p.x + 20, y: p.y + 10 })), 640, 480);
  assert.ok(translatedPose);
  const translated = projectModelAnchors(config, translatedPose);
  assert.equal(translated.length, 1);
  assert.ok(Math.abs(translated[0].x - 340) < 0.5 && Math.abs(translated[0].y - 250) < 0.5);
  assert.equal(estimateModelPose(config, corners, 640, 360), null);
  assert.equal(estimateModelPose(config, [], 640, 480), null);
});
test("the actual marker detector recognizes the configured dictionary and rejects a blank frame", () => {
  const dictionary = new AR.Dictionary(config.dictionary);
  const bits = dictionary.codeList[config.markerId];
  const width = 400, height = 400, data = new Uint8ClampedArray(width * height * 4).fill(255);
  for (let y = 80; y < 320; y++) for (let x = 80; x < 320; x++) {
    const column = Math.floor((x - 80) / 30), row = Math.floor((y - 80) / 30);
    const white = column > 0 && column < 7 && row > 0 && row < 7 && bits[(row - 1) * 6 + column - 1] === "1";
    const offset = (y * width + x) * 4;
    data[offset] = data[offset + 1] = data[offset + 2] = white ? 255 : 0;
  }
  const detector = new AR.Detector({ dictionaryName: config.dictionary, maxHammingDistance: 0 });
  const detected = detector.detect({ width, height, data } as ImageData);
  assert.equal(detected.length, 1);
  assert.equal(detected[0].id, config.markerId);
  assert.deepEqual(detector.detect({ width, height, data: new Uint8ClampedArray(data.length).fill(255) } as ImageData), []);
});

test("printable marker preserves its physical black-square size and decodes through the real detector", () => {
  // Rasterize the actual exported SVG's rectangles, including its white border.
  for (const markerId of [7, 249]) {
    const fixture = { ...createMarkerTestRegistration(400, 400), markerId };
    const svg = printableMarkerSvg(fixture);
    assert.match(svg, /width="100mm" height="100mm"/);
    assert.match(svg, /viewBox="0 0 10 10"/);
    const width = 400, height = 400, cell = 40;
    const data = new Uint8ClampedArray(width * height * 4).fill(255);
    for (const match of svg.matchAll(/<rect x="(\d+)" y="(\d+)" width="(\d+)" height="(\d+)" fill="(white|black)"\/>/g)) {
      const [, sx, sy, sw, sh, fill] = match;
      for (let y = Number(sy) * cell; y < (Number(sy) + Number(sh)) * cell; y++) {
        for (let x = Number(sx) * cell; x < (Number(sx) + Number(sw)) * cell; x++) {
          const offset = (y * width + x) * 4;
          data[offset] = data[offset + 1] = data[offset + 2] = fill === "white" ? 255 : 0;
        }
      }
    }
    const detector = new AR.Detector({ dictionaryName: fixture.dictionary, maxHammingDistance: 0 });
    const markers = detector.detect({ width, height, data } as ImageData);
    assert.equal(markers.length, 1); assert.equal(markers[0].id, markerId);
    const pose = estimateModelPose(fixture, markers[0].corners, width, height);
    assert.ok(pose);
    const points = projectModelAnchors(fixture, pose);
    assert.ok(points.length >= 1);
    assert.ok(Math.hypot(points[0].x - 200, points[0].y - 200) < 1);
  }
  assert.match(printableMarkerSvg(createMarkerTestRegistration(1280, 720, 64)), /width="80mm" height="80mm"/);
});

test("off-center 3D mannequin locations follow small viewpoint changes and detector resizing", () => {
  const fixture = parseModelRegistration({ ...config,
    calibration: { width: 1280, height: 720, fx: 1150, fy: 1100, cx: 630, cy: 350 },
    anchors: [
      { id: "left", structureId: "gallbladder", positionMm: [-55, 20, -10] },
      { id: "right", structureId: "cystic_duct", positionMm: [50, 35, 20] },
    ],
  });
  const half = fixture.markerSizeMm / 2;
  const square = [[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]];
  for (const angle of [-0.18, 0.12, 0.2]) {
    const rotation = [[Math.cos(angle), 0, Math.sin(angle)], [0, 1, 0], [-Math.sin(angle), 0, Math.cos(angle)]];
    const translation = [15, -20, 550];
    const corners = square.map(point => { const [x, y] = projectPoint(point, rotation, translation, fixture.calibration)!; return { x, y }; });
    for (const scale of [1, 0.5]) {
      const pose = estimateModelPose(fixture, corners.map(p => ({ x: p.x * scale, y: p.y * scale })), 1280 * scale, 720 * scale);
      assert.ok(pose);
      const points = projectModelAnchors(fixture, pose);
      assert.equal(points.length, 2);
      fixture.anchors.forEach((anchor, index) => {
        const expected = projectPoint(anchor.positionMm, rotation, translation, fixture.calibration)!;
        const point = points[index];
        assert.ok(Math.hypot(point.x / scale - expected[0], point.y / scale - expected[1]) < 2, `Viewpoint ${angle}, scale ${scale}: anchor drift`);
      });
    }
  }
});

test("marker test uses the actual frame dimensions but never declares its locations measured", () => {
  const fixture = createMarkerTestRegistration(1920, 1080);
  assert.equal(fixture.provenance, "synthetic_mock");
  assert.equal(fixture.calibration.width / fixture.calibration.height, 16 / 9);
  assert.throws(() => createMarkerTestRegistration(1920, 0));
  assert.throws(() => createMarkerTestRegistration(640, 480, NaN));
});
