// Voice, "The Terrain": your voice as a landscape. Frequency on the depth axis, time across, loudness as height,
// computed in the browser from the clip you record or upload (or from a synthesised stand-in when idle).
// One grid, one draw call; the page's guard results become scene events (collapse, fog bank, score light).
import { useFrame } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import { BufferAttribute, BufferGeometry, Color, Mesh, ShaderMaterial } from "three";
import { synthSpeechAsync, terrainHeights } from "../../lib/terrain";
import { palette } from "../palettes";
import { readPrefs } from "../prefs";
import { TIERS } from "../quality";
import { terrainFragment, terrainVertex } from "../shaders/terrain";
import { live, useUi } from "../store";
import { useCompiled } from "../useCompiled";
import { smoothstep } from "../ticker";
import type { VoiceSceneData } from "../voiceData";

const WIDTH = 14;
const DEPTH = 7;

function gridGeometry(cols: number, rows: number) {
  const g = new BufferGeometry();
  const pos = new Float32Array(cols * rows * 3);
  const uv = new Float32Array(cols * rows * 2);
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++) {
      const i = r * cols + c;
      const u = c / (cols - 1), v = r / (rows - 1);
      pos.set([(u - 0.5) * WIDTH, 0, (0.5 - v) * DEPTH], i * 3); // low frequencies near the camera
      uv.set([u, v], i * 2);
    }
  const index = new Uint32Array((cols - 1) * (rows - 1) * 6);
  let k = 0;
  for (let r = 0; r < rows - 1; r++)
    for (let c = 0; c < cols - 1; c++) {
      const a = r * cols + c, b = a + 1, d = a + cols, e = d + 1;
      index.set([a, d, b, b, d, e], k);
      k += 6;
    }
  g.setAttribute("position", new BufferAttribute(pos, 3));
  g.setAttribute("uv", new BufferAttribute(uv, 2));
  g.setAttribute("aH", new BufferAttribute(new Float32Array(cols * rows), 1));
  g.setIndex(new BufferAttribute(index, 1));
  return g;
}

// Camera presets per chapter: [x, y, z] of the eye. The terrain sits at the origin.
const EYE: [number, number, number][] = [[0, 4.6, 9.8], [-3.4, 3.4, 8.2], [5.2, 2.2, 7.4], [0, 5.4, 8.4], [0, 7, 9.6], [0, 7, 9.6]];

export default function VoiceScene() {
  const tier = useUi((s) => s.tier);
  const touch = useMemo(() => readPrefs().isTouch, []);
  const spec = TIERS[tier];
  const cols = touch ? Math.min(spec.voiceCols, 128) : spec.voiceCols;
  const rows = touch ? Math.min(spec.voiceRows, 64) : spec.voiceRows;

  const geometry = useMemo(() => gridGeometry(cols, rows), [cols, rows]);
  const rig = useMemo(() => {
    const material = new ShaderMaterial({
      vertexShader: terrainVertex, fragmentShader: terrainFragment, transparent: true,
      uniforms: {
        uAmp: { value: 2.6 }, uRise: { value: 0 }, uCollapse: { value: 0 }, uFog: { value: 0 }, uCut: { value: 0.6 }, uTime: { value: 0 }, uBreath: { value: 1 },
        uContour: { value: 0 }, uPaper: { value: new Color() }, uInk: { value: new Color() }, uFogColor: { value: new Color() },
        uGlow: { value: new Color() }, uScoreX: { value: 0 }, uScoreOn: { value: 0 }, uAlpha: { value: 1 },
      },
    });
    const mesh = new Mesh(geometry, material);
    mesh.frustumCulled = false;
    return { material, mesh };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // The idle terrain is built from a synthesised stretch of sound, made in small slices after the scene opens (never
  // in one long step); until it is ready the land is flat, then it rises.
  const idle = useRef<Float32Array | null>(null);
  const rise = useRef(0);
  const seen = useRef(-1);
  const collapse = useRef(0), fog = useRef(0), scoreT = useRef(0), cut = useRef(0.6), stage = useRef(0);

  // fill the height attribute for the current samples at this tier's resolution; replace the geometry when the tier changes it
  const heightsFor = (d: VoiceSceneData | undefined) => {
    const samples = d?.samples ?? idle.current;
    return samples ? terrainHeights(samples, cols, rows) : new Float32Array(cols * rows);
  };
  useEffect(() => {
    rig.mesh.geometry = geometry;
    (geometry.attributes.aH.array as Float32Array).set(heightsFor(live.data.voice as VoiceSceneData | undefined));
    geometry.attributes.aH.needsUpdate = true;
    live.dirty = true;
    return () => geometry.dispose();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [geometry, rig]);

  useEffect(() => {
    let alive = true;
    void synthSpeechAsync().then((samples) => {
      if (!alive) return;
      idle.current = samples;
      seen.current = -1; // refill the heights on the next frame
      live.dirty = true;
    });
    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => {
    live.continuous = !live.reduced;
    live.fpsCap = 30;
    return () => {
      live.continuous = false;
      live.fpsCap = 0;
      rig.material.dispose();
    };
  }, [rig]);

  useFrame((state, dt) => {
    const d = live.data.voice as VoiceSceneData | undefined;
    if (seen.current === -1 && (idle.current || d?.samples)) {
      seen.current = d?.version ?? 0;
      (geometry.attributes.aH.array as Float32Array).set(heightsFor(d));
      geometry.attributes.aH.needsUpdate = true;
    } else if (d && d.version !== seen.current) {
      seen.current = d.version;
      (geometry.attributes.aH.array as Float32Array).set(heightsFor(d));
      geometry.attributes.aH.needsUpdate = true;
      scoreT.current = 0; // a new clip or a new answer: the light sweeps again
    }
    const k = live.reduced ? 1 : 1 - Math.exp(-dt * 3.2);
    const wantCollapse = d?.notSpeech || d?.event === "collapse" ? 1 : 0;
    const wantFog = d?.lowQuality || d?.event === "fog" ? 1 : 0;
    collapse.current += (wantCollapse - collapse.current) * k;
    fog.current += (wantFog - fog.current) * k;
    const hz = d?.cutoffHz ?? 5000;
    cut.current += ((Math.min(8000, Math.max(1000, hz)) / 8000) - cut.current) * k;
    if (d?.verdict) scoreT.current = Math.min(1, scoreT.current + dt * 0.45);
    const target = (live.data.stage as number | undefined) ?? 0;
    stage.current += (target - stage.current) * (live.reduced ? 1 : 1 - Math.exp(-dt * 2));
    if (Math.abs(wantCollapse - collapse.current) > 1 / 200 || Math.abs(wantFog - fog.current) > 1 / 200 || Math.abs(target - stage.current) > 1 / 200 || (d?.verdict && scoreT.current < 1)) live.dirty = true;

    const u = rig.material.uniforms;
    if (idle.current || d?.samples) rise.current += (1 - rise.current) * (live.reduced ? 1 : 1 - Math.exp(-dt * 2.5));
    if (1 - rise.current > 1 / 200) live.dirty = true;
    u.uRise.value = rise.current;
    u.uTime.value = live.reduced ? 1 : live.time;
    u.uBreath.value = live.reduced ? 0 : 1;
    u.uCollapse.value = collapse.current;
    u.uFog.value = fog.current;
    u.uCut.value = cut.current;
    u.uScoreOn.value = d?.verdict ? Math.sin(Math.min(1, scoreT.current * 1.1) * Math.PI) : 0;
    u.uScoreX.value = scoreT.current;
    const dark = palette.bg.r + palette.bg.g + palette.bg.b < 1.2;
    u.uContour.value = dark ? 0 : 1;
    u.uPaper.value.copy(palette.bg);
    u.uInk.value.copy(palette.ink);
    u.uFogColor.value.copy(palette.bg).lerp(palette.glow, dark ? 0.12 : 0.04);
    u.uGlow.value.copy(d?.verdict === "high" ? palette.flag : d?.verdict === "medium" ? palette.accent : palette.ledger);

    const s = Math.min(5, stage.current), i = Math.floor(s), f = smoothstep(0, 1, s - i);
    const a = EYE[i], b = EYE[Math.min(5, i + 1)];
    const px = live.reduced ? 0 : live.px * 0.6, py = live.reduced ? 0 : live.py * 0.3;
    const cam = state.camera.position;
    cam.x += (a[0] + (b[0] - a[0]) * f + px - cam.x) * 0.06;
    cam.y += (a[1] + (b[1] - a[1]) * f + py - cam.y) * 0.06;
    cam.z += (a[2] + (b[2] - a[2]) * f - cam.z) * 0.06;
    state.camera.lookAt(0, 0.4, 0);
  });

  const ready = useCompiled(() => [rig.mesh]);
  return ready ? <primitive object={rig.mesh} /> : null;
}
