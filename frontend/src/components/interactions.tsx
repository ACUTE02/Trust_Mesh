import { useEffect, useRef, type ReactNode } from "react";
import { readPrefs } from "../world/prefs";

/** A button or link that leans a few pixels toward the pointer. Desktop only; off with reduced motion. */
export function Magnetic({ children }: { children: ReactNode }) {
  const ref = useRef<HTMLSpanElement>(null);
  useEffect(() => {
    const el = ref.current;
    const prefs = readPrefs();
    if (!el || prefs.reduced || prefs.isTouch) return;
    const move = (e: PointerEvent) => {
      const r = el.getBoundingClientRect();
      const dx = (e.clientX - (r.left + r.width / 2)) / r.width;
      const dy = (e.clientY - (r.top + r.height / 2)) / r.height;
      el.style.transform = `translate3d(${dx * 8}px, ${dy * 6}px, 0)`;
    };
    const leave = () => (el.style.transform = "");
    el.addEventListener("pointermove", move);
    el.addEventListener("pointerleave", leave);
    return () => {
      el.removeEventListener("pointermove", move);
      el.removeEventListener("pointerleave", leave);
    };
  }, []);
  return (
    <span ref={ref} className="magnetic inline-block">
      {children}
    </span>
  );
}

/** A small tag that follows the pointer over elements marked data-cursor="label". Desktop only. */
export function CursorLabel() {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const prefs = readPrefs();
    const el = ref.current;
    if (!el || prefs.reduced || prefs.isTouch) return;
    const move = (e: PointerEvent) => {
      const target = (e.target as HTMLElement | null)?.closest?.("[data-cursor]") as HTMLElement | null;
      if (target) {
        el.textContent = target.dataset.cursor ?? "";
        el.style.opacity = "1";
        el.style.transform = `translate3d(${e.clientX + 14}px, ${e.clientY + 14}px, 0)`;
      } else el.style.opacity = "0";
    };
    window.addEventListener("pointermove", move, { passive: true });
    return () => window.removeEventListener("pointermove", move);
  }, []);
  return <div ref={ref} className="cursor-label" aria-hidden="true" />;
}
