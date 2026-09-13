import { CanvasTexture, DoubleSide, Group, Mesh, MeshBasicMaterial, NoToneMapping, PerspectiveCamera, PlaneGeometry, Scene, SRGBColorSpace, Vector3, WebGLRenderer } from "three";
import type { HudAnchor } from "../overlays/drawHud";
import type { ModelPose } from "./modelRegistration";
import { sceneMatrices } from "./cameraProjection";

export interface ImageOverlayLayer { canvas: HTMLCanvasElement; anchors: HudAnchor[] }
export interface ImagePlaneLayout { width: number; height: number; bottomY: number; centerY: number; z: number }
export interface ScreenImagePlacement { x: number; y: number; width: number; height: number; scale: number }

function positiveDimensions(...dimensions: number[]) {
  if (!dimensions.every(value => Number.isFinite(value) && value > 0)) throw new Error("Image and display dimensions must be positive.");
}

/** Source pixels remain image coordinates until this explicit marker-space placement. */
export function imagePlaneLayout(imageWidth: number, imageHeight: number, planeWidthMm: number, markerSizeMm: number): ImagePlaneLayout {
  positiveDimensions(imageWidth, imageHeight, planeWidthMm, markerSizeMm);
  const height = planeWidthMm * imageHeight / imageWidth;
  const bottomY = markerSizeMm / 2 + 10;
  return { width: planeWidthMm, height, bottomY, centerY: bottomY + height / 2, z: 0.5 };
}

/** Image origin is top-left; marker scene +y points toward the top of the image. */
export function imagePixelToPlanePoint(x: number, y: number, imageWidth: number, imageHeight: number, layout: ImagePlaneLayout): Vector3 {
  return new Vector3((x / imageWidth - 0.5) * layout.width, layout.bottomY + (1 - y / imageHeight) * layout.height, layout.z);
}

export function projectImagePlaneAnchors(anchors: HudAnchor[], imageWidth: number, imageHeight: number,
  pose: ModelPose, width: number, height: number, layout: ImagePlaneLayout): HudAnchor[] {
  positiveDimensions(imageWidth, imageHeight, width, height);
  const matrices = sceneMatrices(pose, width, height);
  return anchors.flatMap(anchor => {
    if (!Number.isFinite(anchor.x) || !Number.isFinite(anchor.y) || anchor.x < 0 || anchor.x > imageWidth || anchor.y < 0 || anchor.y > imageHeight) return [];
    const point = imagePixelToPlanePoint(anchor.x, anchor.y, imageWidth, imageHeight, layout).applyMatrix4(matrices.model);
    if (!Number.isFinite(point.z) || point.z >= -1) return [];
    point.applyMatrix4(matrices.projection);
    if (![point.x, point.y, point.z].every(Number.isFinite) || point.x < -1 || point.x > 1 || point.y < -1 || point.y > 1 || point.z < -1 || point.z > 1) return [];
    return [{ ...anchor, x: (point.x + 1) * width / 2, y: (1 - point.y) * height / 2 }];
  });
}

/** Screen placement is fixed to the camera display; it does not claim table tracking. */
export function screenImagePlacement(imageWidth: number, imageHeight: number, outputWidth: number, outputHeight: number, widthFraction = 0.8): ScreenImagePlacement {
  positiveDimensions(imageWidth, imageHeight, outputWidth, outputHeight, widthFraction);
  if (widthFraction > 1) throw new Error("Image display fraction cannot exceed one.");
  const scale = Math.min(outputWidth * widthFraction / imageWidth, outputHeight * widthFraction / imageHeight);
  const width = imageWidth * scale, height = imageHeight * scale;
  return { x: (outputWidth - width) / 2, y: (outputHeight - height) / 2, width, height, scale };
}

export function projectScreenAnchors(anchors: HudAnchor[], rect: ScreenImagePlacement): HudAnchor[] {
  return anchors.flatMap(anchor => {
    const x = anchor.x * rect.scale, y = anchor.y * rect.scale;
    return Number.isFinite(x) && Number.isFinite(y) && x >= 0 && x <= rect.width && y >= 0 && y <= rect.height
      ? [{ ...anchor, x: rect.x + x, y: rect.y + y }] : [];
  });
}

/** The original image and mask pixels are one unlit texture; labels share its pose. */
export class ImagePlaneRenderer {
  readonly renderer: WebGLRenderer;
  private readonly world = new Scene();
  private readonly root = new Group();
  private readonly camera = new PerspectiveCamera(42, 1, 1, 10000);
  private readonly texture: CanvasTexture;
  private readonly geometry = new PlaneGeometry(1, 1);
  private readonly material: MeshBasicMaterial;
  private readonly plane: Mesh<PlaneGeometry, MeshBasicMaterial>;
  private lost = false;
  private disposed = false;
  private onLost = (event: Event) => { event.preventDefault(); this.lost = true; };
  private onRestored = () => { this.lost = false; this.texture.needsUpdate = true; };

  constructor(private readonly layer: ImageOverlayLayer) {
    positiveDimensions(layer.canvas.width, layer.canvas.height);
    this.texture = new CanvasTexture(layer.canvas);
    this.texture.colorSpace = SRGBColorSpace;
    this.texture.flipY = true;
    this.material = new MeshBasicMaterial({ map: this.texture, transparent: true, side: DoubleSide, toneMapped: false, depthWrite: false });
    this.plane = new Mesh(this.geometry, this.material);
    this.root.matrixAutoUpdate = false;
    this.root.add(this.plane); this.world.add(this.root);
    try { this.renderer = new WebGLRenderer({ alpha: true, antialias: true }); }
    catch (cause) { this.texture.dispose(); this.geometry.dispose(); this.material.dispose(); throw cause; }
    this.renderer.setPixelRatio(1); this.renderer.setClearColor(0x000000, 0);
    this.renderer.outputColorSpace = SRGBColorSpace; this.renderer.toneMapping = NoToneMapping;
    this.renderer.domElement.addEventListener("webglcontextlost", this.onLost);
    this.renderer.domElement.addEventListener("webglcontextrestored", this.onRestored);
  }

  render(pose: ModelPose, width: number, height: number, planeWidthMm: number, markerSizeMm: number, opacity = 1): { image: HTMLCanvasElement; anchors: HudAnchor[] } {
    if (this.disposed) throw new Error("Image overlay renderer has been released.");
    if (this.lost) throw new Error("Image rendering interrupted. Restart the camera if it does not recover.");
    positiveDimensions(width, height);
    if (!Number.isFinite(opacity) || opacity < 0 || opacity > 1) throw new Error("Image opacity must be between zero and one.");
    const layout = imagePlaneLayout(this.layer.canvas.width, this.layer.canvas.height, planeWidthMm, markerSizeMm);
    const matrices = sceneMatrices(pose, width, height);
    this.camera.projectionMatrix.copy(matrices.projection);
    this.camera.projectionMatrixInverse.copy(matrices.projection).invert();
    this.root.matrix.copy(matrices.model); this.root.matrixWorldNeedsUpdate = true;
    this.plane.position.set(0, layout.centerY, layout.z); this.plane.scale.set(layout.width, layout.height, 1);
    this.material.opacity = opacity;
    if (this.renderer.domElement.width !== width || this.renderer.domElement.height !== height) this.renderer.setSize(width, height, false);
    this.renderer.render(this.world, this.camera);
    const anchors = opacity === 0 ? [] : projectImagePlaneAnchors(this.layer.anchors, this.layer.canvas.width, this.layer.canvas.height, pose, width, height, layout);
    return { image: this.renderer.domElement, anchors };
  }

  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    this.renderer.domElement.removeEventListener("webglcontextlost", this.onLost);
    this.renderer.domElement.removeEventListener("webglcontextrestored", this.onRestored);
    this.texture.dispose(); this.geometry.dispose(); this.material.dispose();
    this.renderer.dispose(); this.renderer.forceContextLoss();
  }
}
