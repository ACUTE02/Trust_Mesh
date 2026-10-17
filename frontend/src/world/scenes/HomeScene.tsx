// Home, "The Mesh": thousands of nodes joined by faint threads, one continuous camera flight. The same nodes
// re-arrange for each chapter of the page (mesh, web and waveform, two time sheets, the silence ribbon, a
// spectrogram with its top cut off). Two engine nodes are solid; the Visual Engine is a hollow dashed ghost.
import { useFrame } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import {
  AdditiveBlending, BufferAttribute, BufferGeometry, Color, IcosahedronGeometry, LineBasicMaterial, LineDashedMaterial,
  LineSegments, NormalBlending, Points, ShaderMaterial, Vector2, WireframeGeometry,
} from "three";
import { buildEdges, buildMeshLayout } from "../layouts";
import { palette } from "../palettes";
import { readPrefs } from "../prefs";
import { TIERS } from "../quality";
import { createEngine } from "../engine3d";
import { lineFragment, lineVertex, meshFragment, meshVertex } from "../shaders/mesh";
import { live, useUi } from "../store";
import { useCompiled } from "../useCompiled";

const CHAPTER_ATTRS = ["aT0", "aT1", "aT2", "aT3", "aT4"];

function useMeshGeometry(count: number) {
  return useMemo(() => {
    const layout = buildMeshLayout(count);
    const points = new BufferGeometry();
    points.setAttribute("position", new BufferAttribute(layout.targets[0], 3));
    layout.targets.forEach((t, i) => points.setAttribute(CHAPTER_ATTRS[i], new BufferAttribute(t, 3)));
    points.setAttribute("aSeed", new BufferAttribute(layout.seed, 1));
    points.setAttribute("aKind", new BufferAttribute(layout.kind, 1));
    // The threads share the node attributes and only add an index: one geometry's data, two draw calls.
    const lines = new BufferGeometry();
    for (const [name, attr] of Object.entries(points.attributes)) lines.setAttribute(name, attr);
    lines.setIndex(new BufferAttribute(buildEdges(layout), 1));
    return { points, lines };
  }, [count]);
}

export default function HomeScene() {
  const tier = useUi((s) => s.tier);
  const touch = useMemo(() => readPrefs().isTouch, []);
  const count = Math.round(TIERS[tier].home * (touch ? 0.6 : 1));
  const { points, lines } = useMeshGeometry(count);

  const rig = useMemo(() => {
    const shared = {
      uTime: { value: 0 }, uStage: { value: 0 }, uDrift: { value: 1 }, uPointer: { value: new Vector2() },
      uHalf: { value: new Vector2(7, 3.8) }, uThread: { value: new Color() }, uAlpha: { value: 0.9 },
    };
    const pointMat = new ShaderMaterial({
      vertexShader: meshVertex, fragmentShader: meshFragment, transparent: true, depthWrite: false,
      uniforms: { ...shared, uAccent: { value: new Color() }, uFlag: { value: new Color() }, uSize: { value: 5.5 }, uPixel: { value: 1 } },
    });
    const lineMat = new ShaderMaterial({
      vertexShader: lineVertex, fragmentShader: lineFragment, transparent: true, depthWrite: false,
      uniforms: { ...shared, uAlpha: { value: 0.28 } },
    });
    const ghostSource = new IcosahedronGeometry(0.7, 1);
    const ghostGeo = new WireframeGeometry(ghostSource);
    ghostSource.dispose();
    const ghost = new LineSegments(ghostGeo, new LineDashedMaterial({ dashSize: 0.1, gapSize: 0.09, transparent: true }));
    ghost.computeLineDistances();
    ghost.position.set(3.4, -1.5, 1.2); // the Visual slot: hollow and dashed
    const graphEngine = createEngine();
    graphEngine.root.position.set(2.3, 1.7, 1.2);
    const voiceEngine = createEngine();
    voiceEngine.root.position.set(4.9, 0.8, 1.2);
    const cutGeo = new BufferGeometry();
    cutGeo.setAttribute("position", new BufferAttribute(new Float32Array([-6.5, 0.8, 0, 6.5, 0.8, 0]), 3));
    const cut = new LineSegments(cutGeo, new LineBasicMaterial({ transparent: true }));
    return { shared, pointMat, lineMat, graphEngine, voiceEngine, ghost, cut, pointsObj: new Points(points, pointMat), linesObj: new LineSegments(lines, lineMat) };
    // The geometry objects are replaced when the tier changes the node count (see the effect below).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    rig.pointsObj.geometry = points;
    rig.linesObj.geometry = lines;
    return () => {
      points.dispose();
      lines.dispose();
    };
  }, [points, lines, rig]);

  useEffect(() => {
    live.continuous = !live.reduced;
    return () => {
      live.continuous = false;
      rig.pointMat.dispose(); rig.lineMat.dispose(); rig.graphEngine.dispose(); rig.voiceEngine.dispose();
      rig.ghost.geometry.dispose(); (rig.ghost.material as LineDashedMaterial).dispose();
      rig.cut.geometry.dispose(); (rig.cut.material as LineBasicMaterial).dispose();
    };
  }, [rig]);

  const ready = useCompiled(() => [rig.linesObj, rig.pointsObj, rig.graphEngine.root, rig.voiceEngine.root, rig.ghost, rig.cut]);

  const stageRef = useRef(0);
  useFrame((state, dt) => {
    const { shared, pointMat, lineMat, graphEngine, voiceEngine, ghost, cut, pointsObj, linesObj } = rig;
    const target = (live.data.stage as number | undefined) ?? 0;
    const ease = live.reduced ? 1 : 1 - Math.exp(-dt * 2.4);
    stageRef.current += (target - stageRef.current) * ease;
    const stage = stageRef.current;
    if (Math.abs(target - stage) > 1 / 300) live.dirty = true;

    const dark = palette.bg.r + palette.bg.g + palette.bg.b < 1.2;
    const blending = dark ? AdditiveBlending : NormalBlending;
    if (pointMat.blending !== blending) { pointMat.blending = blending; lineMat.blending = blending; pointMat.needsUpdate = true; lineMat.needsUpdate = true; }
    shared.uStage.value = stage;
    shared.uTime.value = live.reduced ? 3 : live.time;
    shared.uDrift.value = live.reduced ? 0 : 1;
    shared.uPointer.value.set(live.reduced ? 0 : live.px, live.reduced ? 0 : live.py);
    shared.uThread.value.copy(dark ? palette.thread : palette.ink);
    shared.uAlpha.value = dark ? 0.9 : 0.7;
    lineMat.uniforms.uAlpha.value = dark ? 0.26 : 0.22;
    const pm = pointMat.uniforms;
    pm.uAccent.value.copy(palette.accent); pm.uFlag.value.copy(palette.flag); pm.uPixel.value = state.gl.getPixelRatio();
    const vis = 1 - Math.min(1, stage * 1.4); // the engines belong to the opening chapter only
    // solid engines: teal for the Graph Engine, magenta for the Voice Engine (the same colours as their bars on the page)
    graphEngine.update(live.time, state.camera, palette.ledger, palette.bg, vis, live.reduced);
    voiceEngine.update(live.time + 2, state.camera, palette.spectrum, palette.bg, vis, live.reduced);

    // camera flight: closer as the page goes on, a little parallax from the pointer
    state.camera.position.z = 10.6 - Math.min(stage, 4) * 0.55;
    state.camera.position.x += ((live.reduced ? 0 : live.px * 0.5) - state.camera.position.x) * 0.06;
    state.camera.position.y += ((live.reduced ? 0 : live.py * 0.3) - state.camera.position.y) * 0.06;
    state.camera.lookAt(0, 0, 0);
    pointsObj.rotation.y = linesObj.rotation.y = live.reduced ? 0 : Math.sin(live.time * 0.07) * 0.16;

    (ghost.material as LineDashedMaterial).opacity = vis;
    (ghost.material as LineDashedMaterial).color.copy(palette.ink);
    const cm = cut.material as LineBasicMaterial;
    cm.opacity = Math.max(0, 1 - Math.abs(stage - 4) * 2);
    cm.color.copy(palette.flag);
  });

  if (!ready) return null;
  return (
    <group>
      <primitive object={rig.linesObj} frustumCulled={false} />
      <primitive object={rig.pointsObj} frustumCulled={false} />
      <primitive object={rig.graphEngine.root} />
      <primitive object={rig.voiceEngine.root} />
      <primitive object={rig.ghost} />
      <primitive object={rig.cut} />
    </group>
  );
}
