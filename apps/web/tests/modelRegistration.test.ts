import assert from "node:assert/strict";
import test from "node:test";
import aruco from "js-aruco2";
import { parseModelRegistration, projectPoint, registerModel } from "../src/camera/modelRegistration";
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
  const anchors = registerModel(config, corners, 640, 480);
  assert.ok(anchors && anchors.length === 1);
  assert.ok(Math.abs(anchors[0].x - 320) < 0.1 && Math.abs(anchors[0].y - 240) < 0.1);
  const translated = registerModel(config, corners.map(p => ({ x: p.x + 20, y: p.y + 10 })), 640, 480);
  assert.ok(translated && Math.abs(translated[0].x - 340) < 0.5 && Math.abs(translated[0].y - 250) < 0.5);
  assert.equal(registerModel(config, corners, 640, 360), null);
  assert.equal(registerModel(config, [], 640, 480), null);
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
