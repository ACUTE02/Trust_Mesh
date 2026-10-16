// The one persistent WebGL canvas. It is loaded after first paint (App.tsx), draws behind the page, and holds
// one scene at a time, chosen by the route. It renders on demand: the render loop is the shared ticker, and a
// frame is drawn only when something changed (scroll, pointer, a theme change, an animating scene).
import { Canvas, useThree } from "@react-three/fiber";
import { lazy, Suspense, useEffect, useMemo, useState, type ComponentType } from "react";
import { palette, refreshPalette } from "./palettes";
import { TIER_ORDER, TIERS, tierCap } from "./quality";
import { live, markDirty, setUi, useUi, type Tier } from "./store";
import { onTick } from "./ticker";
import { readPrefs } from "./prefs";
import type { WorldId } from "./worlds";

// One chunk per world: a world never loads another world's scene code.
const SCENES: Record<WorldId, ComponentType> = {
  home: lazy(() => import("./scenes/HomeScene")),
  graph: lazy(() => import("./scenes/GraphScene")),
  voice: lazy(() => import("./scenes/VoiceScene")),
  case: lazy(() => import("./scenes/CaseScene")),
  evidence: lazy(() => import("./scenes/EvidenceScene")),
  about: lazy(() => import("./scenes/AboutScene")),
  lost: lazy(() => import("./scenes/LostScene")),
};

/** Counts for the ?perf=1 readout and for the measurements in the report. Dev only; nothing is sent anywhere. */
interface PerfSnapshot { fps: number; frameMs: number; calls: number; triangles: number; geometries: number; textures: number; tier: Tier; world: string }
declare global { interface Window { __tmPerf?: PerfSnapshot & { visits?: Record<string, PerfSnapshot> } } }

function Driver({ perf }: { perf: boolean }) {
  const advance = useThree((s) => s.advance);
  const gl = useThree((s) => s.gl);

  useEffect(() => {
    const prefs = readPrefs();
    const cap = TIER_ORDER.indexOf(tierCap(prefs.isTouch, prefs.lowPower));
    let tierIndex = 0; // start low
    let ceiling = cap; // lowered after a demotion so the governor never flip-flops
    let frames = 0, spanStart = 0, goodWindows = 0, lastRender = 0, cooldownUntil = 0;
    let frameMsAvg = 16, lastPerf = 0, rendersSincePerf = 0;

    refreshPalette();
    gl.setClearColor(palette.bg, 0);
    const canvas = gl.domElement;
    const lost = (e: Event) => { e.preventDefault(); setUi({ webgl: false }); };
    canvas.addEventListener("webglcontextlost", lost);
    setUi({ tier: "low", webgl: true, stageReady: true });

    const stop = onTick((t) => {
      live.time = t;
      if (!live.dirty && !live.continuous) { frames = 0; return; }
      live.dirty = false;
      const t0 = performance.now();
      advance(t * 1000, true);
      const cost = performance.now() - t0;
      frameMsAvg = frameMsAvg * 0.9 + cost * 0.1;
      rendersSincePerf++;

      // Governor: judge only runs of back-to-back frames (an idle gap says nothing about speed).
      const now = performance.now();
      if (now - lastRender > 120) { frames = 0; spanStart = now; }
      lastRender = now;
      frames++;
      const span = now - spanStart;
      if (frames >= 30 && now > cooldownUntil) {
        const fps = (frames * 1000) / Math.max(span, 1);
        if (fps < 45 && tierIndex > 0) {                       // demote quickly on a dip
          tierIndex--; ceiling = Math.min(ceiling, tierIndex); goodWindows = 0; cooldownUntil = now + 5000;
          setUi({ tier: TIER_ORDER[tierIndex] });
        } else if (fps >= 57) {                                // promote only after three good windows in a row
          if (++goodWindows >= 3 && tierIndex < ceiling) {
            tierIndex++; goodWindows = 0; cooldownUntil = now + 5000;
            setUi({ tier: TIER_ORDER[tierIndex] });
          }
        } else goodWindows = 0;
        frames = 0; spanStart = now;
      }
      if (perf && now - lastPerf > 500) {
        const info = gl.info;
        // fps is frames actually drawn per second over the last window (an idle page draws none); frameMs is the cost of one draw
        const snap = { fps: Math.round((rendersSincePerf * 1000) / Math.max(now - lastPerf, 1)), frameMs: +frameMsAvg.toFixed(2), calls: info.render.calls,
          triangles: info.render.triangles, geometries: info.memory.geometries, textures: info.memory.textures,
          tier: TIER_ORDER[tierIndex], world: document.documentElement.dataset.world ?? "home" };
        lastPerf = now;
        rendersSincePerf = 0;
        window.__tmPerf = { ...snap, visits: { ...(window.__tmPerf?.visits ?? {}), [snap.world]: snap } };
      }
    });
    return () => { stop(); canvas.removeEventListener("webglcontextlost", lost); };
  }, [advance, gl, perf]);
  return null;
}

function Readout() {
  const [snap, setSnap] = useState<PerfSnapshot | null>(null);
  useEffect(() => {
    const id = setInterval(() => window.__tmPerf && setSnap({ ...window.__tmPerf }), 500);
    return () => clearInterval(id);
  }, []);
  if (!snap) return null;
  return (
    <pre aria-hidden style={{ position: "fixed", left: 8, bottom: 8, zIndex: 99, margin: 0, padding: "6px 8px", font: "11px/1.4 monospace", background: "#000c", color: "#fff", pointerEvents: "none" }}>
      {`fps ${snap.fps}  frame ${snap.frameMs} ms\ncalls ${snap.calls}  tris ${snap.triangles}\ngeom ${snap.geometries}  tex ${snap.textures}\ntier ${snap.tier}  world ${snap.world}`}
    </pre>
  );
}

export default function Stage() {
  const world = useUi((s) => s.world) as WorldId;
  const tier = useUi((s) => s.tier);
  const perf = useMemo(() => new URLSearchParams(window.location.search).get("perf") === "1", []);
  const Scene = SCENES[world] ?? SCENES.home;
  const spec = TIERS[tier];

  // A theme change re-reads the palette; the canvas stays.
  useEffect(() => {
    const obs = new MutationObserver(() => { refreshPalette(); markDirty(); });
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme", "data-world"] });
    return () => obs.disconnect();
  }, []);

  return (
    <>
    <div id="stage" aria-hidden="true">
      <Canvas
        frameloop="never"
        dpr={[1, spec.dpr]}
        gl={{ antialias: spec.antialias, alpha: true, powerPreference: "high-performance" }}
        camera={{ fov: 50, near: 0.1, far: 80, position: [0, 0, 9] }}
        onCreated={() => markDirty()}
      >
        <Driver perf={perf} />
        <Suspense fallback={null}>
          <Scene key={world} />
        </Suspense>
      </Canvas>
    </div>
    {perf && <Readout />}
    </>
  );
}
