import { Matrix4 } from "three";
import type { ModelPose } from "./modelRegistration";

/** Three uses -z for camera depth; scene +z is height above the marker plane. */
export function sceneMatrices(pose: ModelPose, width: number, height: number) {
  const { rotation: r, translation: t, calibration: c } = pose;
  const fx = c.fx * width / c.width, fy = c.fy * height / c.height;
  const cx = c.cx * width / c.width, cy = c.cy * height / c.height;
  const near = 1, far = 10000;
  return {
    model: new Matrix4().set(
      r[0][0], r[0][1], -r[0][2], t[0],
      r[1][0], r[1][1], -r[1][2], t[1],
      -r[2][0], -r[2][1], r[2][2], -t[2],
      0, 0, 0, 1,
    ),
    projection: new Matrix4().set(
      2 * fx / width, 0, 1 - 2 * cx / width, 0,
      0, 2 * fy / height, 2 * cy / height - 1, 0,
      0, 0, -(far + near) / (far - near), -2 * far * near / (far - near),
      0, 0, -1, 0,
    ),
  };
}
