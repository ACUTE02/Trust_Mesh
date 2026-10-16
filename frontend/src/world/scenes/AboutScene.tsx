// About, "The Workbench": an exploded line drawing of the system (data, models, API, Neo4j, the site) that assembles as
// you scroll and draws itself on. Each part is one line list with a draw-on shader; a dashed dimension line runs
// beneath. Six draw calls, no textures, no lights.
import { useFrame } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import {
  BoxGeometry, BufferAttribute, BufferGeometry, Color, CylinderGeometry, EdgesGeometry, IcosahedronGeometry, LineDashedMaterial,
  LineSegments, ShaderMaterial, type Group,
} from "three";
import { palette } from "../palettes";
import { live } from "../store";
import { useCompiled } from "../useCompiled";
import { clamp01, smoothstep } from "../ticker";

const vertex = /* glsl */ `
  attribute float aD; varying float vD;
  void main() { vD = aD; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }
`;
const fragment = /* glsl */ `
  uniform vec3 uColor; uniform float uDraw; varying float vD;
  void main() {
    if (vD > uDraw) discard;
    float head = smoothstep(uDraw - 0.06, uDraw, vD);   // the newest stretch of line glows
    gl_FragColor = vec4(mix(uColor, vec3(1.0), head * 0.6), 1.0);
  }
`;

type Parts = Float32Array[];

/** Edges of several shapes merged into one line list, each segment tagged with how far along the drawing it is. */
function drawing(shapes: { geo: BufferGeometry; at: [number, number, number] }[]): BufferGeometry {
  const chunks: Parts = shapes.map(({ geo, at }) => {
    const e = new EdgesGeometry(geo, 25).attributes.position.array as Float32Array;
    const out = new Float32Array(e.length);
    for (let i = 0; i < e.length; i += 3) out.set([e[i] + at[0], e[i + 1] + at[1], e[i + 2] + at[2]], i);
    geo.dispose();
    return out;
  });
  const total = chunks.reduce((n, c) => n + c.length, 0);
  const pos = new Float32Array(total);
  let o = 0;
  for (const c of chunks) {
    pos.set(c, o);
    o += c.length;
  }
  const segs = total / 6;
  const d = new Float32Array(total / 3);
  for (let s = 0; s < segs; s++) d[s * 2] = d[s * 2 + 1] = s / segs;
  const g = new BufferGeometry();
  g.setAttribute("position", new BufferAttribute(pos, 3));
  g.setAttribute("aD", new BufferAttribute(d, 1));
  return g;
}

const SPACING = 2.7;
const PARTS: { name: string; at: number; explode: [number, number, number]; build: () => BufferGeometry }[] = [
  {
    name: "data", at: -2 * SPACING, explode: [0, 2.4, -1.6],
    build: () => drawing([0, 1, 2].map((i) => ({ geo: new CylinderGeometry(0.95, 0.95, 0.34, 20), at: [0, (i - 1) * 0.5, 0] as [number, number, number] }))),
  },
  {
    name: "models", at: -SPACING, explode: [0, -1.9, 1.3],
    build: () => drawing([{ geo: new BoxGeometry(1.8, 1.3, 1.3), at: [0, 0, 0] }, { geo: new BoxGeometry(0.9, 0.6, 0.9), at: [0, 0, 0] }]),
  },
  { name: "api", at: 0, explode: [0, 2.9, 0.6], build: () => drawing([{ geo: new CylinderGeometry(1.0, 1.0, 1.1, 6), at: [0, 0, 0] }]) },
  {
    name: "graph", at: SPACING, explode: [0, -2.4, -1.1],
    build: () =>
      drawing([[-0.7, 0.3, 0], [0.6, 0.6, 0.3], [0, -0.5, 0.2], [0.8, -0.3, -0.4], [-0.5, -0.6, -0.3]].map((p) => ({ geo: new IcosahedronGeometry(0.28, 0), at: p as [number, number, number] }))),
  },
  { name: "site", at: 2 * SPACING, explode: [0, 1.7, 1.6], build: () => drawing([{ geo: new BoxGeometry(2.2, 1.4, 0.16), at: [0, 0.2, 0] }, { geo: new BoxGeometry(0.4, 0.4, 0.3), at: [0, -0.7, 0] }]) },
];

export default function AboutScene() {
  const rig = useMemo(() => {
    const parts = PARTS.map((p) => {
      const mat = new ShaderMaterial({ vertexShader: vertex, fragmentShader: fragment, uniforms: { uColor: { value: new Color() }, uDraw: { value: 0 } } });
      const lines = new LineSegments(p.build(), mat);
      lines.frustumCulled = false;
      return { ...p, mat, lines };
    });
    // dimension line under the assembled drawing: a dashed rule with a tick under each part
    const dim = new BufferGeometry();
    const pts: number[] = [-2 * SPACING - 1, -2.35, 0, 2 * SPACING + 1, -2.35, 0];
    for (const x of [-2, -1, 0, 1, 2].map((k) => k * SPACING)) pts.push(x, -2.1, 0, x, -2.6, 0);
    dim.setAttribute("position", new BufferAttribute(new Float32Array(pts), 3));
    const dimMat = new LineDashedMaterial({ dashSize: 0.18, gapSize: 0.12, transparent: true });
    const dimLines = new LineSegments(dim, dimMat);
    dimLines.computeLineDistances();
    dimLines.frustumCulled = false;
    return { parts, dim, dimMat, dimLines };
  }, []);

  const group = useRef<Group>(null);
  const stage = useRef(0);
  const born = useRef(-1);

  useEffect(() => {
    live.continuous = !live.reduced; // a slow turn once assembled
    live.fpsCap = 30;
    return () => {
      live.continuous = false;
      live.fpsCap = 0;
      for (const p of rig.parts) {
        p.lines.geometry.dispose();
        p.mat.dispose();
      }
      rig.dim.dispose();
      rig.dimMat.dispose();
    };
  }, [rig]);

  useFrame((state, dt) => {
    if (born.current < 0) born.current = live.time;
    const want = (live.data.stage as number | undefined) ?? 0;
    stage.current += (want - stage.current) * (live.reduced ? 1 : 1 - Math.exp(-dt * 2.2));
    const s = stage.current;
    // on load the drawing draws itself on over a couple of seconds, then follows the page
    const intro = live.reduced ? 1 : smoothstep(0, 2.6, live.time - born.current);
    if (intro < 1 || Math.abs(want - s) > 1 / 200) live.dirty = true;
    const base = (0.4 + 0.6 * smoothstep(0, 1.6, s)) * intro;
    const explode = 1 - smoothstep(0.4, 1.5, s);
    rig.parts.forEach((p, i) => {
      p.mat.uniforms.uDraw.value = clamp01(base * 1.4 - i * 0.07);
      p.mat.uniforms.uColor.value.copy(palette.thread);
      p.lines.position.set(p.at + p.explode[0] * explode, p.explode[1] * explode, p.explode[2] * explode);
    });
    rig.dimMat.color.copy(palette.accent2);
    rig.dimMat.opacity = clamp01((1 - explode) * 1.2);
    const g = group.current;
    if (g) {
      g.rotation.x = -0.32;
      g.rotation.y = 0.5 + (live.reduced ? 0 : Math.sin(live.time * 0.12) * 0.1 + live.px * 0.12);
    }
    state.camera.position.set(0, 0.8, 12.2 - smoothstep(1.5, 3, s) * 1.5);
    state.camera.lookAt(0, 0, 0);
  });

  const ready = useCompiled(() => [...rig.parts.map((p) => p.lines), rig.dimLines]);
  if (!ready) return null;
  return (
    <group ref={group}>
      {rig.parts.map((p) => (
        <primitive key={p.name} object={p.lines} />
      ))}
      <primitive object={rig.dimLines} />
    </group>
  );
}
