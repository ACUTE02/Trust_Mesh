import { Suspense, lazy, useEffect, useRef, useState } from "react";
import { Link, Route, Routes, useLocation, type Location } from "react-router-dom";
import { Footer } from "./components/Footer";
import { CursorLabel } from "./components/interactions";
import { Header } from "./components/Nav";
import { Poster } from "./components/Poster";
import { GraphPage } from "./pages/GraphPage";
import { Landing } from "./pages/Landing";
import { VoicePage } from "./pages/VoicePage";
import { useMotionPrefs, webglAvailable } from "./world/prefs";
import { jumpTo, startSmoothScroll, trackPointer } from "./world/smooth";
import { live, markDirty, setUi, useUi } from "./world/store";
import { WORLDS, worldFor, type WorldId } from "./world/worlds";

// Pages of the newer worlds load on demand; the three first pages are part of the first load.
const CasePage = lazy(() => import("./pages/CasePage"));
const EvidencePage = lazy(() => import("./pages/EvidencePage"));
const AboutPage = lazy(() => import("./pages/AboutPage"));
const Stage = lazy(() => import("./world/Stage"));

/** The 3D stage, loaded after first paint, or the still poster when there is no WebGL, data saving is on, or the context is lost. */
function StageHost() {
  const prefs = useMotionPrefs();
  const world = useUi((s) => s.world) as WorldId;
  const ready = useUi((s) => s.stageReady);
  const webgl = useUi((s) => s.webgl);
  const [available] = useState(() => webglAvailable());
  const [go, setGo] = useState(false);
  const still = !available || prefs.saveData || (ready && !webgl);

  useEffect(() => {
    if (still) return;
    const ric = window.requestIdleCallback as undefined | ((cb: () => void, o?: { timeout: number }) => number);
    const id = ric ? ric(() => setGo(true), { timeout: 1500 }) : window.setTimeout(() => setGo(true), 400);
    return () => (ric ? window.cancelIdleCallback?.(id) : window.clearTimeout(id));
  }, [still]);

  if (still) return <Poster world={world} />;
  return go ? (
    <Suspense fallback={null}>
      <Stage />
    </Suspense>
  ) : null;
}

function NotFound() {
  // Hovering or focusing the link ties the snapped thread back to the node in the scene.
  const tie = (tied: boolean) => {
    live.data.lost = { tied };
    markDirty();
  };
  useEffect(() => () => void (live.data.lost = undefined), []);
  return (
    <section className="hero" aria-labelledby="lost-h">
      <div className="scrim chapter-card">
        <h1 id="lost-h" style={{ fontSize: "var(--text-hero)" }}>
          This node is not linked to anything.
        </h1>
        <p className="measure mt-4">There is no page at this address.</p>
        <p className="mb-0">
          <Link
            to="/"
            className="font-semibold"
            onPointerEnter={() => tie(true)}
            onPointerLeave={() => tie(false)}
            onFocus={() => tie(true)}
            onBlur={() => tie(false)}
          >
            Tie it back to the overview
          </Link>
        </p>
      </div>
    </section>
  );
}

/** Runs a world-specific exit curtain between two routes, swapping the page at its midpoint. Reduced motion: a short crossfade. */
function useShownLocation(location: Location, reduced: boolean) {
  const [shown, setShown] = useState(location);
  const [run, setRun] = useState<{ to: WorldId; key: number } | null>(null);
  const [fade, setFade] = useState(0);

  useEffect(() => {
    if (location.pathname === shown.pathname) {
      if (location.hash !== shown.hash || location.key !== shown.key) setShown(location);
      return;
    }
    const to = worldFor(location.pathname);
    if (reduced) {
      setShown(location);
      setFade((n) => n + 1);
      return;
    }
    document.documentElement.dataset.to = to;
    document.documentElement.style.setProperty("--t", `${WORLDS[to].transitionMs}ms`);
    setRun({ to, key: Date.now() });
    const swap = window.setTimeout(() => setShown(location), WORLDS[to].transitionMs * 0.45);
    const done = window.setTimeout(() => setRun(null), WORLDS[to].transitionMs);
    return () => {
      window.clearTimeout(swap);
      window.clearTimeout(done);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [location, reduced]);

  return { shown, run, fade };
}

export default function App() {
  const location = useLocation();
  const prefs = useMotionPrefs();
  const { shown, run, fade } = useShownLocation(location, prefs.reduced);
  const world = worldFor(shown.pathname);
  const first = useRef(true);

  // Smooth scroll and pointer tracking (both off for reduced motion / touch where they would hurt).
  useEffect(() => {
    live.reduced = prefs.reduced;
    const stopScroll = startSmoothScroll({ reduced: prefs.reduced, touch: prefs.isTouch });
    const stopPointer = prefs.reduced || prefs.isTouch ? () => {} : trackPointer();
    return () => {
      stopScroll();
      stopPointer();
    };
  }, [prefs.reduced, prefs.isTouch]);

  // The world's palette and fonts follow the page that is shown; the title and focus follow every route change.
  useEffect(() => {
    document.documentElement.dataset.world = world;
    setUi({ world, theme: document.documentElement.dataset.theme ?? "light" });
    document.title = WORLDS[world].title;
    if (shown.hash) jumpTo(shown.hash);
    else jumpTo(0);
    if (first.current) {
      first.current = false;
      return;
    }
    const h1 = document.querySelector<HTMLElement>("main h1");
    if (h1) {
      h1.tabIndex = -1;
      h1.focus({ preventScroll: true });
    }
    markDirty();
  }, [world, shown.pathname, shown.hash]);

  return (
    <>
      <div className="backdrop" aria-hidden="true" />
      <StageHost />
      <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:bg-sheet focus:p-2">
        Skip to the content
      </a>
      <Header />
      <main id="main" key={fade} data-fade={fade ? "1" : undefined} className="mx-auto max-w-[76rem] px-5 sm:px-8">
        <Suspense fallback={<p className="py-24 text-ink-soft">Loading the page.</p>}>
          <Routes location={shown}>
            <Route path="/" element={<Landing />} />
            <Route path="/graph" element={<GraphPage />} />
            <Route path="/voice" element={<VoicePage />} />
            <Route path="/case" element={<CasePage />} />
            <Route path="/evidence" element={<EvidencePage />} />
            <Route path="/about" element={<AboutPage />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </Suspense>
      </main>
      <Footer world={world} />
      <CursorLabel />
      {run && (
        <>
          <div className="curtain" data-run="1" aria-hidden="true" key={`c${run.key}`} />
          <div className="curtain-thread" data-run="1" aria-hidden="true" key={`t${run.key}`} />
        </>
      )}
    </>
  );
}
