// Case, "The Balance": two scores, one decision, honestly weighed. A beam balance: the Graph score on the left pan,
// the Voice score on the right, and a third pan hanging empty and dashed for the Visual slot, which is not built.
// The beam tilts from the scores of the case on screen; it moves like a machine (a damped spring with a small click).
import { useFrame } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import {
  AmbientLight, BoxGeometry, BufferAttribute, BufferGeometry, CylinderGeometry, DirectionalLight, EdgesGeometry, Group,
  InstancedMesh, LineBasicMaterial, LineDashedMaterial, LineSegments, Matrix4, Mesh, MeshLambertMaterial, SphereGeometry,
} from "three";
import type { CaseSceneData } from "../caseData";
import { palette } from "../palettes";
import { live } from "../store";
import { smoothstep } from "../ticker";

const PIVOT_Y = 2.5;
const ARM = 3.4;
const PAN_R = 1.15;
const DROP = 1.9;
const MAX_WEIGHTS = 12;

// camera presets per chapter: eye, then target
const VIEWS: { eye: [number, number, number]; at: [number, number, number] }[] = [
  { eye: [0.4, 2.2, 10.5], at: [0.8, 1.5, 0] },
  { eye: [0, 3.4, 6.2], at: [0, PIVOT_Y, 0] },
  { eye: [2.2, 2.4, 11.5], at: [1.6, 1.3, 0] },
  { eye: [-0.4, 0.6, 7.2], at: [0, 0.6, -1.2] },
];

export default function CaseScene() {
  const rig = useMemo(() => {
    const steel = new MeshLambertMaterial();
    const dark = new MeshLambertMaterial();
    const brass = new MeshLambertMaterial();
    const red = new MeshLambertMaterial();
    const stringMat = new LineBasicMaterial({ transparent: true, opacity: 0.8 });
    const ghostMat = new LineDashedMaterial({ dashSize: 0.14, gapSize: 0.1, transparent: true, opacity: 0.9 });

    const group = new Group();
    const base = new Mesh(new BoxGeometry(5.2, 0.3, 2.2), dark);
    base.position.y = -2.3;
    const pillar = new Mesh(new CylinderGeometry(0.15, 0.22, PIVOT_Y + 2.2, 16), steel);
    pillar.position.y = (PIVOT_Y - 2.3) / 2;
    const beamGeo = new BoxGeometry(ARM * 2 + 0.4, 0.16, 0.28);
    const beam = new Mesh(beamGeo, steel);
    beam.position.y = PIVOT_Y;
    const bolt = new Mesh(new CylinderGeometry(0.2, 0.2, 0.5, 20), red);
    bolt.rotation.x = Math.PI / 2;
    bolt.position.set(0, PIVOT_Y, 0.2);
    const panGeo = new CylinderGeometry(PAN_R, PAN_R, 0.1, 28);
    const left = new Mesh(panGeo, brass);
    const right = new Mesh(panGeo, brass);
    // the Visual slot: a hollow, dashed pan on a dashed line, far behind the others
    const ghost = new LineSegments(new EdgesGeometry(new CylinderGeometry(PAN_R, PAN_R, 0.1, 20)), ghostMat);
    ghost.computeLineDistances();
    ghost.position.set(0, 0.5, -1.4);
    const ghostString = new LineSegments(new BufferGeometry(), ghostMat);
    ghostString.geometry.setAttribute("position", new BufferAttribute(new Float32Array([0, PIVOT_Y, -1.4, 0, 0.55, -1.4]), 3));
    ghostString.computeLineDistances();
    // strings: three per pan, from the beam end to the pan rim (positions are rewritten each frame, 2 pans x 3 x 2 points)
    const stringGeo = new BufferGeometry();
    stringGeo.setAttribute("position", new BufferAttribute(new Float32Array(2 * 3 * 2 * 3), 3));
    const strings = new LineSegments(stringGeo, stringMat);
    strings.frustumCulled = false;
    const weights = new InstancedMesh(new CylinderGeometry(0.34, 0.34, 0.1, 18), brass, MAX_WEIGHTS * 2);
    weights.frustumCulled = false;
    const dot = new Mesh(new SphereGeometry(0.1, 8, 8), red);
    group.add(base, pillar, beam, bolt, left, right, ghost, ghostString, strings, weights, dot);
    const lights = [new AmbientLight(0xffffff, 0.9), new DirectionalLight(0xffffff, 1.6)];
    lights[1].position.set(-4, 7, 6);
    return { group, steel, dark, brass, red, stringMat, ghostMat, beam, bolt, left, right, ghost, strings, stringGeo, weights, lights, dot };
  }, []);

  useEffect(() => {
    live.continuous = !live.reduced;
    live.fpsCap = 30;
    return () => {
      live.continuous = false;
      live.fpsCap = 0;
      rig.group.traverse((o) => {
        const m = o as Mesh;
        m.geometry?.dispose?.();
      });
      for (const mat of [rig.steel, rig.dark, rig.brass, rig.red, rig.stringMat, rig.ghostMat]) mat.dispose();
      rig.weights.dispose();
    };
  }, [rig]);

  const angle = useRef({ a: 0, v: 0 });
  const stage = useRef(0);
  const matrix = useMemo(() => new Matrix4(), []);

  useFrame((state, dt) => {
    const d = live.data.case as CaseSceneData | undefined;
    const g = d?.graph ?? null, v = d?.voice ?? null;
    const wg = d?.weightGraph ?? 0.5, wv = d?.weightVoice ?? 0.5;
    // beam target: tilts toward the heavier weighted score; level with no scores
    const target = Math.max(-0.42, Math.min(0.42, ((g ?? 0) * wg - (v ?? 0) * wv) * 0.9));
    // damped spring: precise, a little overshoot, then still (the motion of a machine, not a toy)
    const s = angle.current;
    const step = Math.min(dt, 1 / 30);
    const force = (target - s.a) * 60 - s.v * 11;
    s.v += force * step;
    s.a += s.v * step;
    if (Math.abs(target - s.a) > 1 / 400 || Math.abs(s.v) > 1 / 200) live.dirty = true;
    const a = live.reduced ? target : s.a;

    const want = (live.data.stage as number | undefined) ?? 0;
    stage.current += (want - stage.current) * (live.reduced ? 1 : 1 - Math.exp(-dt * 2));
    if (Math.abs(want - stage.current) > 1 / 200) live.dirty = true;

    const { beam, left, right, stringGeo, weights, bolt, ghost, ghostMat, dot } = rig;
    beam.rotation.z = a;
    const lx = -Math.cos(a) * ARM, ly = PIVOT_Y - Math.sin(a) * ARM; // the left end goes down when a is positive
    const rx = Math.cos(a) * ARM, ry = PIVOT_Y + Math.sin(a) * ARM;
    left.position.set(lx, ly - DROP, 0);
    right.position.set(rx, ry - DROP, 0);
    const pos = stringGeo.attributes.position.array as Float32Array;
    let k = 0;
    for (const [ex, ey, px, py] of [[lx, ly, lx, ly - DROP], [rx, ry, rx, ry - DROP]]) {
      for (const ang of [0, 2.094, 4.189]) {
        pos.set([ex, ey, 0, px + Math.cos(ang) * PAN_R * 0.9, py, Math.sin(ang) * PAN_R * 0.9], k);
        k += 6;
      }
    }
    stringGeo.attributes.position.needsUpdate = true;

    // weights: a stack of brass discs on each pan, as tall as its score; neutral and equal when masked
    const masked = d?.masked ?? false;
    const count = (score: number | null) => (masked ? 4 : score === null ? 0 : Math.max(1, Math.ceil(score * 10)));
    const stacks: [number, number, number][] = [[lx, ly - DROP, count(g)], [rx, ry - DROP, count(v)]];
    let n = 0;
    for (const [sx, sy, c] of stacks) {
      for (let i = 0; i < MAX_WEIGHTS; i++) {
        const shown = i < c;
        matrix.makeTranslation(sx + (i % 2) * 0.04, sy + 0.1 + i * 0.11, 0);
        if (!shown) matrix.makeScale(0, 0, 0);
        weights.setMatrixAt(n++, matrix);
      }
    }
    weights.instanceMatrix.needsUpdate = true;

    // the bolt that holds the beam is not fixed: the weights are placeholders. In chapter 1 it works loose.
    const loose = smoothstep(0.5, 1, stage.current) * (1 - smoothstep(1.6, 2.2, stage.current));
    bolt.position.z = 0.2 + loose * (0.45 + (live.reduced ? 0 : Math.sin(live.time * 2.2) * 0.12));
    bolt.rotation.y = loose * (live.reduced ? 0 : Math.sin(live.time * 3) * 0.35);
    dot.position.set(0, PIVOT_Y + 0.25, 0.25);
    ghost.rotation.y = live.reduced ? 0 : live.time * 0.15;
    ghostMat.opacity = 0.55 + 0.4 * smoothstep(2.4, 3, stage.current);

    // colours come from the page palette
    rig.steel.color.copy(palette.thread);
    rig.dark.color.copy(palette.ink).lerp(palette.bg, 0.6);
    rig.brass.color.copy(palette.accent2);
    rig.red.color.copy(palette.accent);
    rig.stringMat.color.copy(palette.thread);
    ghostMat.color.copy(palette.ink);

    rig.group.position.x = 2.1 * (1 - smoothstep(0.4, 1.2, stage.current)) + 1.2 * smoothstep(1.4, 2.4, stage.current);
    const sIdx = Math.min(3, Math.floor(stage.current)), f = smoothstep(0, 1, stage.current - sIdx);
    const A = VIEWS[sIdx], B = VIEWS[Math.min(3, sIdx + 1)];
    const cam = state.camera.position;
    const px = live.reduced ? 0 : live.px * 0.4;
    cam.x += (A.eye[0] + (B.eye[0] - A.eye[0]) * f + px - cam.x) * 0.07;
    cam.y += (A.eye[1] + (B.eye[1] - A.eye[1]) * f - cam.y) * 0.07;
    cam.z += (A.eye[2] + (B.eye[2] - A.eye[2]) * f - cam.z) * 0.07;
    state.camera.lookAt(A.at[0] + (B.at[0] - A.at[0]) * f, A.at[1] + (B.at[1] - A.at[1]) * f, A.at[2] + (B.at[2] - A.at[2]) * f);
  });

  return (
    <>
      <primitive object={rig.lights[0]} />
      <primitive object={rig.lights[1]} />
      <primitive object={rig.group} />
    </>
  );
}
