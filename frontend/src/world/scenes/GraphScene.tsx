// Graph, "The Ledger": a transaction is a bead on a copper thread. One loom of threads with beads, which the page's
// chapters re-arrange: two sheets (past and future, with a leak), bead-stack bars, and a live neighbourhood.
// Beads are lit-sphere point sprites, so the whole world is two draw calls.
import { useFrame } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import { BufferAttribute, BufferGeometry, Color, LineSegments, NormalBlending, Points, ShaderMaterial, type Group } from "three";
import { chapterProgress } from "../chapters";
import { buildBeads, buildThreads, fillBars, fillNeighbours, type Bead } from "../graphLayout";
import { palette } from "../palettes";
import { readPrefs } from "../prefs";
import { TIERS } from "../quality";
import { beadFragment, beadVertex, threadFragment, threadVertex } from "../shaders/beads";
import { live, useUi } from "../store";
import { smoothstep } from "../ticker";

/** What the page tells the scene (set in GraphPage; read here). Bars come from the data module, neighbours from the live query. */
export interface GraphSceneData {
  version: number;
  bars: { value: number; tint: number }[];
  centre: { flagged: boolean } | null;
  neighbours: { hops: number; flagged: boolean }[];
}

function applyData(b: Bead, d: GraphSceneData | undefined) {
  fillBars(b, d?.bars ?? []);
  fillNeighbours(b, d?.centre ?? null, d?.neighbours ?? []);
}

function geometryFor(b: Bead) {
  const g = new BufferGeometry();
  g.setAttribute("position", new BufferAttribute(b.t[0], 3));
  b.t.forEach((t, i) => g.setAttribute(`aT${i}`, new BufferAttribute(t, 3)));
  g.setAttribute("aLeak", new BufferAttribute(b.leak, 3));
  g.setAttribute("aVis", new BufferAttribute(b.vis, 4));
  g.setAttribute("aTint", new BufferAttribute(b.tint, 1));
  g.setAttribute("aKind", new BufferAttribute(b.kind, 1));
  g.setAttribute("aSeed", new BufferAttribute(b.seed, 1));
  return g;
}

export default function GraphScene() {
  const tier = useUi((s) => s.tier);
  const touch = useMemo(() => readPrefs().isTouch, []);
  const count = Math.round(TIERS[tier].graph * (touch ? 0.6 : 1));

  const beads = useMemo(() => buildBeads(count), [count]);
  const geometry = useMemo(() => geometryFor(beads), [beads]);

  const rig = useMemo(() => {
    const common = { uStage: { value: 0 }, uTime: { value: 0 }, uSway: { value: 1 }, uLiftY: { value: 0 }, uLift: { value: 0 } };
    const beadMat = new ShaderMaterial({
      vertexShader: beadVertex, fragmentShader: beadFragment, transparent: true, depthWrite: false, blending: NormalBlending,
      uniforms: {
        ...common, uLeak: { value: 0 }, uSize: { value: 20 }, uPixel: { value: 1 }, uBase: { value: new Color() },
        uAccent: { value: new Color() }, uFlag: { value: new Color() }, uLight: { value: new Color() },
      },
    });
    const threadMat = new ShaderMaterial({
      vertexShader: threadVertex, fragmentShader: threadFragment, transparent: true, depthWrite: false,
      uniforms: { ...common, uThread: { value: new Color() }, uAlpha: { value: 0.5 } },
    });
    const t = buildThreads();
    const tg = new BufferGeometry();
    tg.setAttribute("position", new BufferAttribute(new Float32Array(t.x.length * 3), 3));
    tg.setAttribute("aX", new BufferAttribute(t.x, 1));
    tg.setAttribute("aRow", new BufferAttribute(t.row, 1));
    return { beadMat, threadMat, tg, threads: new LineSegments(tg, threadMat), common };
  }, []);

  const points = useMemo(() => new Points(geometry, rig.beadMat), [geometry, rig]);

  const seen = useRef(-1);
  const stageRef = useRef(0);
  const group = useRef<Group>(null);

  // new bead set (a tier change) gets the current page data; old geometry is released
  useEffect(() => {
    applyData(beads, live.data.graph as GraphSceneData | undefined);
    seen.current = (live.data.graph as GraphSceneData | undefined)?.version ?? -1;
    return () => geometry.dispose();
  }, [beads, geometry]);

  useEffect(() => {
    live.continuous = !live.reduced;
    live.fpsCap = 30; // slow, weighty motion does not need more
    return () => {
      live.continuous = false;
      live.fpsCap = 0;
      rig.beadMat.dispose();
      rig.threadMat.dispose();
      rig.tg.dispose();
    };
  }, [rig]);

  useFrame((state, dt) => {
    const d = live.data.graph as GraphSceneData | undefined;
    if (d && d.version !== seen.current) {
      seen.current = d.version;
      applyData(beads, d);
      for (const name of ["aT2", "aT3", "aVis", "aTint"]) (geometry.attributes[name] as BufferAttribute).needsUpdate = true;
    }
    const target = (live.data.stage as number | undefined) ?? 0;
    stageRef.current += (target - stageRef.current) * (live.reduced ? 1 : 1 - Math.exp(-dt * 1.6));
    const s = stageRef.current;
    if (Math.abs(target - s) > 1 / 300) live.dirty = true;

    const { common, beadMat, threadMat } = rig;
    common.uStage.value = s;
    common.uTime.value = live.reduced ? 2 : live.time;
    common.uSway.value = live.reduced ? 0 : 1;
    const calm = 1 - Math.max(smoothstep(0.8, 2, s) * (1 - smoothstep(3, 4, s)), 0); // lift only on the loom
    common.uLift.value = live.reduced ? 0 : 0.34 * calm;
    common.uLiftY.value = live.py * 3.2;
    const p1 = chapterProgress(1);
    beadMat.uniforms.uLeak.value = smoothstep(0.3, 0.55, p1) * (1 - smoothstep(0.8, 0.97, p1));
    beadMat.uniforms.uPixel.value = state.gl.getPixelRatio();
    beadMat.uniforms.uBase.value.copy(palette.ledger);
    beadMat.uniforms.uAccent.value.copy(palette.accent);
    beadMat.uniforms.uFlag.value.copy(palette.flag);
    beadMat.uniforms.uLight.value.copy(palette.glow);
    threadMat.uniforms.uThread.value.copy(palette.thread);
    const dark = palette.bg.r + palette.bg.g + palette.bg.b < 1.2;
    threadMat.uniforms.uAlpha.value = dark ? 0.55 : 0.7;

    // The loom is turned toward you; the data chapters face the camera square-on.
    const face = smoothstep(1, 2, s) * (1 - smoothstep(3, 4, s));
    const g = group.current;
    if (g) {
      g.rotation.x = -0.3 * (1 - face);
      g.rotation.y = (0.38 + (live.reduced ? 0 : live.px * 0.08)) * (1 - face);
    }
    state.camera.position.z = 11.2 - face * 1.6;
    state.camera.position.y += (-0.2 - state.camera.position.y) * 0.05;
    state.camera.lookAt(0, 0, 0);
  });

  return (
    <group ref={group}>
      <primitive object={rig.threads} frustumCulled={false} />
      <primitive object={points} frustumCulled={false} />
    </group>
  );
}
