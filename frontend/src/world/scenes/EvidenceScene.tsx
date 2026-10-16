// Evidence, "The Archive": a wall of drawers. A pulled drawer is an exhibit that is open on the page; the dashed
// empty drawers are the things nobody tested; one drawer holds the sealed envelope. Two instanced meshes and one
// line list, so the whole wall is three draw calls. The page's exhibit buttons are the keyboard controls.
import { useFrame, useThree } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import {
  BoxGeometry, BufferAttribute, BufferGeometry, Color, EdgesGeometry, InstancedMesh, LineDashedMaterial, LineSegments,
  Matrix4, MeshLambertMaterial, SphereGeometry, Mesh, AmbientLight, DirectionalLight,
} from "three";
import { EMPTY_DRAWERS, ENVELOPE_DRAWER, EXHIBIT_DRAWERS, WALL_COLS, WALL_ROWS, type EvidenceSceneData } from "../evidenceData";
import { palette } from "../palettes";
import { live } from "../store";
import { useCompiled } from "../useCompiled";
import { smoothstep } from "../ticker";

const W = 2.0, H = 1.3, GAP_X = 0.14, GAP_Y = 0.16, D = 1.1;
const COUNT = WALL_COLS * WALL_ROWS;
const OPEN_BY = 1.15;

function centre(i: number): [number, number] {
  const c = i % WALL_COLS, r = Math.floor(i / WALL_COLS);
  return [(c - (WALL_COLS - 1) / 2) * (W + GAP_X), ((WALL_ROWS - 1) / 2 - r) * (H + GAP_Y)];
}

// camera presets per chapter: eye then target
const VIEWS: { eye: [number, number, number]; at: [number, number, number] }[] = [
  { eye: [0, 0.4, 15], at: [0, 0, 0] },
  { eye: [2.2, 1.2, 10.5], at: [1.6, 0.6, 0] },
  { eye: [-3, -1.2, 11], at: [-2.4, -1, 0] },
  { eye: [0.6, 0.4, 8.5], at: [0.6, 0.2, 0] },
  { eye: [0, 0.2, 17], at: [0, 0, 0] },
];

export default function EvidenceScene() {
  const rig = useMemo(() => {
    const drawerMat = new MeshLambertMaterial();
    const faceMat = new MeshLambertMaterial();
    const sealMat = new MeshLambertMaterial();
    const dashed = new LineDashedMaterial({ dashSize: 0.16, gapSize: 0.12, transparent: true, opacity: 0.85 });
    const drawers = new InstancedMesh(new BoxGeometry(W, H, D), drawerMat, COUNT);
    const faces = new InstancedMesh(new BoxGeometry(W * 0.5, 0.16, 0.06), faceMat, COUNT); // the label slot on each front
    drawers.frustumCulled = false;
    faces.frustumCulled = false;
    // dashed outlines for the empty drawers, merged into one geometry
    const edge = new EdgesGeometry(new BoxGeometry(W, H, D)).attributes.position.array as Float32Array;
    const merged = new Float32Array(edge.length * EMPTY_DRAWERS.length);
    EMPTY_DRAWERS.forEach((idx, n) => {
      const [x, y] = centre(idx);
      for (let k = 0; k < edge.length; k += 3) merged.set([edge[k] + x, edge[k + 1] + y, edge[k + 2]], n * edge.length + k);
    });
    const emptyGeo = new BufferGeometry();
    emptyGeo.setAttribute("position", new BufferAttribute(merged, 3));
    const empties = new LineSegments(emptyGeo, dashed);
    empties.computeLineDistances();
    empties.frustumCulled = false;
    const seal = new Mesh(new SphereGeometry(0.17, 12, 10), sealMat);
    const [ex, ey] = centre(ENVELOPE_DRAWER);
    seal.position.set(ex, ey - 0.15, D / 2 + 0.1);
    const lights = [new AmbientLight(0xffffff, 1.0), new DirectionalLight(0xffffff, 1.5)];
    lights[1].position.set(-3, 5, 8);
    return { drawerMat, faceMat, sealMat, dashed, drawers, faces, empties, emptyGeo, seal, lights };
  }, []);

  const amount = useRef(new Float32Array(COUNT)); // 0 shut, 1 pulled out, per drawer
  const seen = useRef(-1);
  const stage = useRef(0);
  const matrix = useMemo(() => new Matrix4(), []);
  const tmp = useMemo(() => new Color(), []);
  const open = useRef<Set<number>>(new Set());
  const envelope = useRef(0);

  const camera = useThree((st) => st.camera);
  useEffect(() => {
    live.continuous = false; // drawers move only when something changes; nothing needs to drift
    camera.position.set(...VIEWS[0].eye); // start at the first view; later views are eased to
    return () => {
      rig.drawers.geometry.dispose();
      rig.faces.geometry.dispose();
      rig.emptyGeo.dispose();
      rig.seal.geometry.dispose();
      for (const m of [rig.drawerMat, rig.faceMat, rig.sealMat, rig.dashed]) m.dispose();
      rig.drawers.dispose();
      rig.faces.dispose();
    };
  }, [rig, camera]);

  useFrame((state, dt) => {
    const d = live.data.evidence as EvidenceSceneData | undefined;
    if (d && d.version !== seen.current) {
      seen.current = d.version;
      open.current = new Set(d.open);
    }
    const k = live.reduced ? 1 : 1 - Math.exp(-dt * 7);
    let moving = false;
    const a = amount.current;
    const want = (d?.envelopeOpen ? 1 : 0) as number;
    envelope.current += (want - envelope.current) * k;
    if (Math.abs(want - envelope.current) > 1 / 200) moving = true;
    for (let i = 0; i < COUNT; i++) {
      const target = open.current.has(i) ? 1 : 0;
      a[i] += (target - a[i]) * k;
      if (Math.abs(target - a[i]) > 1 / 200) moving = true;
      const [x, y] = centre(i);
      const isEmpty = EMPTY_DRAWERS.includes(i);
      const pull = i === ENVELOPE_DRAWER ? envelope.current * 0.6 : a[i];
      const z = pull * OPEN_BY;
      matrix.makeTranslation(x, y, z);
      if (isEmpty) matrix.makeScale(0, 0, 0); // the empty drawers are drawn by the dashed outline instead
      rig.drawers.setMatrixAt(i, matrix);
      matrix.makeTranslation(x, y + 0.12, D / 2 + 0.03 + z);
      if (isEmpty) matrix.makeScale(0, 0, 0);
      rig.faces.setMatrixAt(i, matrix);
      const exhibit = EXHIBIT_DRAWERS.includes(i);
      tmp.copy(exhibit || i === ENVELOPE_DRAWER ? palette.accent2 : palette.accent).lerp(palette.bg, exhibit ? 0.0 : 0.45);
      rig.drawers.setColorAt(i, tmp);
    }
    rig.drawers.instanceMatrix.needsUpdate = true;
    rig.faces.instanceMatrix.needsUpdate = true;
    if (rig.drawers.instanceColor) rig.drawers.instanceColor.needsUpdate = true;
    rig.seal.position.z = D / 2 + 0.1 + envelope.current * 0.6 * OPEN_BY;
    rig.seal.scale.setScalar(1 - envelope.current * 0.9); // the seal breaks when the envelope opens

    const want2 = (live.data.stage as number | undefined) ?? 0;
    stage.current += (want2 - stage.current) * (live.reduced ? 1 : 1 - Math.exp(-dt * 2));
    if (Math.abs(want2 - stage.current) > 1 / 200) moving = true;
    if (moving) live.dirty = true;

    rig.drawerMat.color.set(0xffffff);
    rig.faceMat.color.copy(palette.ink);
    rig.sealMat.color.copy(palette.flag);
    rig.dashed.color.copy(palette.ink);

    const s = Math.min(4, stage.current), i = Math.floor(s), f = smoothstep(0, 1, s - i);
    const A = VIEWS[i], B = VIEWS[Math.min(4, i + 1)];
    const cam = state.camera.position;
    const px = live.reduced ? 0 : live.px * 0.7, py = live.reduced ? 0 : live.py * 0.4;
    const tx = A.eye[0] + (B.eye[0] - A.eye[0]) * f + px, ty = A.eye[1] + (B.eye[1] - A.eye[1]) * f + py, tz = A.eye[2] + (B.eye[2] - A.eye[2]) * f;
    cam.x += (tx - cam.x) * 0.07;
    cam.y += (ty - cam.y) * 0.07;
    cam.z += (tz - cam.z) * 0.07;
    if (Math.abs(tx - cam.x) + Math.abs(ty - cam.y) + Math.abs(tz - cam.z) > 1 / 50) live.dirty = true; // keep drawing until the camera arrives
    state.camera.lookAt(A.at[0] + (B.at[0] - A.at[0]) * f, A.at[1] + (B.at[1] - A.at[1]) * f, 0);
  });

  const ready = useCompiled(() => [rig.drawers, rig.faces, rig.empties, rig.seal, ...rig.lights]);
  if (!ready) return null;
  return (
    <>
      <primitive object={rig.lights[0]} />
      <primitive object={rig.lights[1]} />
      <primitive object={rig.drawers} />
      <primitive object={rig.faces} />
      <primitive object={rig.empties} />
      <primitive object={rig.seal} />
    </>
  );
}
