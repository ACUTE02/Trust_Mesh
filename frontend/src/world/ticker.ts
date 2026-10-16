// The one animation loop of the site. Smooth scroll (Lenis), the 3D stage and the page transitions all
// subscribe here; there is no second requestAnimationFrame anywhere. It stops while the tab is hidden.
export type TickFn = (seconds: number, delta: number) => void;

const subscribers = new Set<TickFn>();
let raf = 0;
let last = 0;

function frame(now: number) {
  raf = requestAnimationFrame(frame);
  const delta = Math.min((now - last) / 1000, 0.1); // a long pause (tab switch) must not become one huge step
  last = now;
  subscribers.forEach((fn) => fn(now / 1000, delta));
}

function start() {
  if (raf || typeof requestAnimationFrame === "undefined" || document.hidden) return;
  last = performance.now();
  raf = requestAnimationFrame(frame);
}

function stop() {
  if (raf) cancelAnimationFrame(raf);
  raf = 0;
}

if (typeof document !== "undefined") {
  document.addEventListener("visibilitychange", () => (document.hidden ? stop() : subscribers.size && start()));
}

/** Subscribe; returns the unsubscribe function. The loop runs only while someone is subscribed. */
export function onTick(fn: TickFn): () => void {
  subscribers.add(fn);
  start();
  return () => {
    subscribers.delete(fn);
    if (!subscribers.size) stop();
  };
}

// Easing curves. Each world uses its own (see worlds.ts), so the pages move differently.
export const ease = {
  linear: (t: number) => t,
  drift: (t: number) => t * t * (3 - 2 * t), // Home: long, soft
  settle: (t: number) => 1 - Math.pow(1 - t, 4) * Math.cos(t * 6) * (1 - t), // Graph: heavy, settles like a bead
  elastic: (t: number) => (t === 0 || t === 1 ? t : 1 - Math.pow(2, -9 * t) * Math.cos((t * 10 - 0.75) * ((2 * Math.PI) / 3))), // Voice
  damped: (t: number) => 1 - Math.pow(1 - t, 3) + Math.sin(t * Math.PI * 3) * 0.04 * (1 - t), // Case: precise, tiny click
  drawer: (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2), // Evidence
  draw: (t: number) => 1 - Math.pow(1 - t, 2), // About: line draws on
};

export const clamp01 = (x: number) => (x < 0 ? 0 : x > 1 ? 1 : x);
export const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
export const smoothstep = (a: number, b: number, x: number) => {
  const t = clamp01((x - a) / (b - a));
  return t * t * (3 - 2 * t);
};
