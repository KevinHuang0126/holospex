import {
  BoxGeometry, CatmullRomCurve3, DoubleSide, ExtrudeGeometry, Group,
  Mesh, MeshStandardMaterial, Path, Shape, SphereGeometry, TubeGeometry, Vector3,
} from "three";
import { anatomy, type AnatomyId } from "@holospex/contracts";
import { createMarkerTestRegistration } from "./markerSetup";

// Authored synthetic geometry: an exposed, enlarged biliary teaching view.
// Scene z is height ABOVE the table; the POSIT model axis has the opposite sign.
const BODY_Y = 210;
export const surgicalLandmarks: { id: AnatomyId; position: [number, number, number] }[] = [
  { id: "gallbladder", position: [-25, BODY_Y + 4, 61] },
  { id: "cystic_duct", position: [-4, BODY_Y + 21, 56] },
  { id: "cystic_artery", position: [7, BODY_Y + 35, 64] },
];

export function createSurgicalRegistration(width = 640, height = 480, markerSizeMm = 80) {
  const config = createMarkerTestRegistration(width, height, markerSizeMm);
  return { ...config, modelId: "Open abdomen · gallbladder scene", anchors: surgicalLandmarks.map(({ id, position: [x, y, z] }) => ({
    id, structureId: id, positionMm: [x, y, -z] as [number, number, number],
  })) };
}

/** No downloaded patient imagery or third-party anatomical mesh is used. */
export function createSurgicalScene(): Group {
  const root = new Group(); root.name = "Synthetic open-abdomen surgical scene";
  const material = (color: string, roughness = 0.65, metalness = 0) => new MeshStandardMaterial({ color, roughness, metalness, side: DoubleSide });
  const skin = material("#d5a181"), fat = material("#ddbc74"), muscle = material("#a95158");
  const cavity = material("#663a46"), drape = material("#237b83", 0.94), edge = material("#124f5c");
  const liver = material("#9f4e4e", 0.4), bowel = material("#ce8e88", 0.52), metal = material("#b2c7cf", 0.3, 0.65);
  const gallbladder = material(anatomy.gallbladder.color, 0.32);
  const duct = material(anatomy.cystic_duct.color, 0.38), artery = material(anatomy.cystic_artery.color, 0.38);
  const bile = material("#b4a665", 0.5);
  const add = (parent: Group, geometry: Mesh["geometry"], surface: MeshStandardMaterial, position: number[], name: string) => {
    const mesh = new Mesh(geometry, surface); mesh.position.set(position[0], position[1], position[2]); mesh.name = name;
    parent.add(mesh); return mesh;
  };
  const ellipsoid = (parent: Group, position: number[], scale: number[], surface: MeshStandardMaterial, name: string) => {
    const mesh = add(parent, new SphereGeometry(1, 28, 18), surface, position, name);
    mesh.scale.set(scale[0], scale[1], scale[2]); return mesh;
  };
  const tube = (parent: Group, points: number[][], radius: number, surface: MeshStandardMaterial, name: string) => {
    const curve = new CatmullRomCurve3(points.map(p => new Vector3(p[0], p[1], p[2])));
    return add(parent, new TubeGeometry(curve, Math.max(18, points.length * 7), radius, 8, false), surface, [0, 0, 0], name);
  };
  const rounded = (w: number, h: number, r: number) => {
    const s = new Shape(); s.moveTo(-w / 2 + r, -h / 2);
    s.lineTo(w / 2 - r, -h / 2); s.quadraticCurveTo(w / 2, -h / 2, w / 2, -h / 2 + r);
    s.lineTo(w / 2, h / 2 - r); s.quadraticCurveTo(w / 2, h / 2, w / 2 - r, h / 2);
    s.lineTo(-w / 2 + r, h / 2); s.quadraticCurveTo(-w / 2, h / 2, -w / 2, h / 2 - r);
    s.lineTo(-w / 2, -h / 2 + r); s.quadraticCurveTo(-w / 2, -h / 2, -w / 2 + r, -h / 2);
    return s;
  };
  const extrude = (shape: Shape, depth: number, bevel = 2) => new ExtrudeGeometry(shape, { depth, bevelEnabled: bevel > 0, bevelSegments: 3, steps: 1, bevelSize: bevel, bevelThickness: bevel, curveSegments: 28 });

  add(root, extrude(rounded(252, 374, 18), 5), edge, [0, 236, 2], "Surgical platform");
  add(root, extrude(rounded(242, 364, 16), 1, 1), drape, [0, 236, 8], "Sterile teal drape");
  const body = new Group(); body.position.y = BODY_Y; root.add(body);
  const torso = new Shape();
  torso.moveTo(-42, -124); torso.bezierCurveTo(-72, -121, -65, -56, -69, -18);
  torso.bezierCurveTo(-79, 40, -92, 87, -75, 104); torso.bezierCurveTo(-57, 118, -37, 121, -25, 128);
  torso.lineTo(25, 128); torso.bezierCurveTo(37, 121, 57, 118, 75, 104);
  torso.bezierCurveTo(92, 87, 79, 40, 69, -18); torso.bezierCurveTo(65, -56, 72, -121, 42, -124); torso.closePath();
  const opening = new Path(); opening.absellipse(0, -4, 58, 80, 0, Math.PI * 2, true); torso.holes.push(opening);
  add(body, extrude(torso, 30, 4), skin, [0, 0, 12], "Open torso shell");
  const ring = (rx: number, ry: number, inset: number, z: number, surface: MeshStandardMaterial, name: string) => {
    const shape = new Shape(); shape.absellipse(0, -4, rx, ry, 0, Math.PI * 2, false);
    const hole = new Path(); hole.absellipse(0, -4, rx - inset, ry - inset, 0, Math.PI * 2, true); shape.holes.push(hole);
    add(body, extrude(shape, 4, 0.7), surface, [0, 0, z], name);
  };
  ring(59, 81, 5, 38, fat, "Cutaway fat layer");
  ring(54, 76, 4, 34, muscle, "Abdominal muscle rim");
  ellipsoid(body, [0, -4, 19], [53, 75, 9], cavity, "Abdominal cavity floor");
  ellipsoid(body, [0, 133, 29], [20, 24, 21], skin, "Neck");
  ellipsoid(body, [0, 165, 34], [31, 37, 28], skin, "Training mannequin head");
  ellipsoid(body, [0, 151, 58], [5, 9, 4], skin, "Mannequin nose");
  for (const sign of [-1, 1]) {
    ellipsoid(body, [sign * 39, 93, 36], [34, 29, 12], skin, "Upper chest");
    const arm = ellipsoid(body, [sign * 89, 36, 25], [16, 61, 17], skin, "Upper arm"); arm.rotation.z = sign * 0.09;
    const forearm = ellipsoid(body, [sign * 96, -46, 22], [13, 36, 14], skin, "Forearm"); forearm.rotation.z = sign * -0.08;
    ellipsoid(body, [sign * 97, -84, 22], [12, 16, 9], skin, "Mannequin hand");
  }
  // Two lobes form a lifted inferior liver edge, exposing the biliary teaching area.
  const rightLobe = ellipsoid(body, [-22, 49, 37], [35, 27, 19], liver, "Liver right lobe"); rightLobe.rotation.z = -0.18;
  const leftLobe = ellipsoid(body, [19, 54, 35], [29, 19, 13], liver, "Liver left lobe"); leftLobe.rotation.z = 0.2;
  // Background abdominal context sits below the three highlighted structures.
  for (let i = 0; i < 5; i++) {
    const y = -24 - i * 10;
    tube(body, [[-33, y, 31], [-17, y + 4, 33], [0, y - 3, 32], [20, y + 3, 32], [32, y - 1, 29]], 4.5, bowel, "Small-intestine fold");
  }
  tube(body, [[35, 0, 30], [44, -13, 30], [43, -31, 31], [29, -39, 29], [18, -31, 29]], 6, bowel, "Duodenum");
  tube(body, [[-5, 66, 47], [2, 49, 50], [9, 37, 50], [9, 15, 51], [13, -12, 43], [27, -27, 35]], 2.5, bile, "Hepatic and common bile duct");
  tube(body, [[29, 61, 44], [21, 45, 48], [9, 37, 50]], 2.5, bile, "Hepatic duct branch");
  const gall = ellipsoid(body, [-23, 8, 48], [12, 23, 10], gallbladder, "Gallbladder");
  const positions = gall.geometry.getAttribute("position");
  for (let i = 0; i < positions.count; i++) {
    const taper = 0.85 - 0.35 * positions.getY(i);
    positions.setX(i, positions.getX(i) * taper); positions.setZ(i, positions.getZ(i) * taper);
  }
  gall.geometry.computeVertexNormals(); gall.rotation.z = -0.3;
  tube(body, [[-16, 29, 48], [-13, 30, 54], [-4, 21, 56], [7, 15, 51]], 2.7, duct, "Cystic duct");
  tube(body, [[24, 48, 51], [19, 40, 59], [7, 35, 64], [-4, 29, 62], [-14, 26, 56]], 2.1, artery, "Cystic artery");

  // Retractors around the opening make the cutaway read as a surgical training scene.
  for (const side of [-1, 1]) {
    for (const y of [-43, 33]) {
      tube(body, [[side * 94, y, 47], [side * 74, y, 49], [side * 54, y, 46], [side * 49, y, 34]], 1.9, metal, "Retractor handle");
      add(body, new BoxGeometry(3, 14, 11), metal, [side * 50, y, 37], "Retractor blade");
    }
    tube(root, [[side * 118, 63, 11], [side * 115, 132, 11], [side * 116, 244, 11], [side * 114, 404, 11]], 0.7, edge, "Drape seam");
  }
  // Mesh names are retained for replacing individual organs with reviewed assets later.
  return root;
}

export function disposeSurgicalScene(root: Group) {
  const materials = new Set<MeshStandardMaterial>();
  root.traverse(object => {
    if (object instanceof Mesh) {
      object.geometry.dispose();
      for (const surface of Array.isArray(object.material) ? object.material : [object.material]) materials.add(surface as MeshStandardMaterial);
    }
  });
  materials.forEach(surface => surface.dispose());
}
