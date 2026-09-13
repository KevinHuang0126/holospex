import { ACESFilmicToneMapping, DirectionalLight, HemisphereLight, PerspectiveCamera, Scene, Vector3, WebGLRenderer } from "three";
import type { ModelPose } from "./modelRegistration";
import type { HudAnchor } from "../overlays/drawHud";
import { createSurgicalScene, disposeSurgicalScene, surgicalLandmarks } from "./surgicalScene";
import { sceneMatrices } from "./cameraProjection";
export { sceneMatrices } from "./cameraProjection";

/** One GPU canvas is copied into the same recorded HUD canvas as the camera frame. */
export class SurgicalRenderer {
  readonly renderer: WebGLRenderer;
  readonly world = new Scene();
  readonly model = createSurgicalScene();
  readonly camera = new PerspectiveCamera(42, 1, 1, 10000);
  private lost = false;
  private onLost = (event: Event) => { event.preventDefault(); this.lost = true; };
  private onRestored = () => { this.lost = false; };

  constructor() {
    try { this.renderer = new WebGLRenderer({ alpha: true, antialias: true }); }
    catch (cause) { disposeSurgicalScene(this.model); throw cause; }
    this.renderer.setPixelRatio(1); this.renderer.setClearColor(0x000000, 0);
    this.renderer.toneMapping = ACESFilmicToneMapping;
    this.renderer.domElement.addEventListener("webglcontextlost", this.onLost);
    this.renderer.domElement.addEventListener("webglcontextrestored", this.onRestored);
    const ambient = new HemisphereLight(0xe5f3ff, 0x45302b, 2.3); ambient.position.set(0, 0, 1);
    const key = new DirectionalLight(0xfff4e8, 2.7); key.position.set(-160, -100, 600);
    const fill = new DirectionalLight(0xbce9ff, 1.1); fill.position.set(200, 300, 350);
    this.world.add(this.model, ambient, key, fill);
    this.model.matrixAutoUpdate = false;
  }

  render(pose: ModelPose, width: number, height: number): HTMLCanvasElement {
    if (this.lost) throw new Error("3D rendering interrupted. Restart the camera if it does not recover.");
    const matrices = sceneMatrices(pose, width, height);
    this.camera.position.set(0, 0, 0); this.camera.quaternion.identity(); this.camera.updateMatrixWorld(true);
    this.camera.projectionMatrix.copy(matrices.projection);
    this.camera.projectionMatrixInverse.copy(matrices.projection).invert();
    this.model.matrix.copy(matrices.model);
    if (this.renderer.domElement.width !== width || this.renderer.domElement.height !== height) this.renderer.setSize(width, height, false);
    this.renderer.render(this.world, this.camera);
    return this.renderer.domElement;
  }

  preview(width: number, height: number): { image: HTMLCanvasElement; anchors: HudAnchor[] } {
    if (this.lost) throw new Error("3D preview interrupted. Reload to restore it.");
    this.model.matrix.identity();
    this.camera.aspect = width / height; this.camera.position.set(285, -235, 640);
    this.camera.up.set(0, 0, 1); this.camera.lookAt(0, 235, 30); this.camera.updateProjectionMatrix(); this.camera.updateMatrixWorld(true);
    if (this.renderer.domElement.width !== width || this.renderer.domElement.height !== height) this.renderer.setSize(width, height, false);
    this.renderer.render(this.world, this.camera);
    return { image: this.renderer.domElement, anchors: surgicalLandmarks.map(({ id, position }) => {
      const p = new Vector3(...position).project(this.camera);
      return { id, structureId: id, x: (p.x + 1) * width / 2, y: (1 - p.y) * height / 2 };
    }) };
  }

  dispose() {
    this.renderer.domElement.removeEventListener("webglcontextlost", this.onLost);
    this.renderer.domElement.removeEventListener("webglcontextrestored", this.onRestored);
    disposeSurgicalScene(this.model); this.renderer.dispose(); this.renderer.forceContextLoss();
  }
}
