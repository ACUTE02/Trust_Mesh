// Smooth scroll. Lenis is driven from the shared ticker (no loop of its own) and publishes the scroll
// position to `live`, which the 3D scenes read. With reduced motion, or a touch screen, native scrolling is
// left alone: touch scrolling is already smooth and must not be hijacked.
import Lenis from "lenis";
import { onTick } from "./ticker";
import { live, markDirty } from "./store";

let lenis: Lenis | null = null;
let stopTick: (() => void) | null = null;

function publish(scroll: number, velocity: number) {
  const max = Math.max(1, document.documentElement.scrollHeight - window.innerHeight);
  live.scroll = scroll;
  live.progress = Math.min(1, Math.max(0, scroll / max));
  live.velocity = velocity;
  markDirty();
}

export function startSmoothScroll(opts: { reduced: boolean; touch: boolean }): () => void {
  stopSmoothScroll();
  const onScroll = () => publish(window.scrollY, 0);
  window.addEventListener("scroll", onScroll, { passive: true });
  if (!opts.reduced && !opts.touch && typeof ResizeObserver !== "undefined") {
    try {
      lenis = new Lenis({ autoRaf: false, lerp: 0.11, smoothWheel: true, syncTouch: false });
      lenis.on("scroll", (l: Lenis) => publish(l.scroll, l.velocity));
      stopTick = onTick((t) => lenis?.raf(t * 1000));
    } catch {
      lenis = null; // native scrolling still works
    }
  }
  publish(window.scrollY, 0);
  return () => {
    window.removeEventListener("scroll", onScroll);
    stopSmoothScroll();
  };
}

function stopSmoothScroll() {
  stopTick?.();
  stopTick = null;
  lenis?.destroy();
  lenis = null;
}

/** Jump (no animation) to the top, or to an element id. Used on route changes. */
export function jumpTo(target: number | string) {
  if (lenis) lenis.scrollTo(target, { immediate: true, force: true });
  else if (typeof target === "number") window.scrollTo(0, target);
  else document.getElementById(target.replace(/^#/, ""))?.scrollIntoView();
  publish(window.scrollY, 0);
}

/** Pointer position for parallax, written into `live` (desktop only; the scenes ignore it when reduced). */
export function trackPointer(): () => void {
  const move = (e: PointerEvent) => {
    live.px = (e.clientX / window.innerWidth) * 2 - 1;
    live.py = -((e.clientY / window.innerHeight) * 2 - 1);
    markDirty();
  };
  window.addEventListener("pointermove", move, { passive: true });
  return () => window.removeEventListener("pointermove", move);
}
