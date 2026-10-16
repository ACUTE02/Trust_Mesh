import { useEffect, useRef, type ReactNode } from "react";
import { live, markDirty } from "./store";

// Chapters connect the page to the 3D scene. A <Chapter> is an ordinary page section. When it crosses the middle of
// the viewport it sets `live.data.stage`, and the scene eases toward it. Its position is measured when the page
// lays out (and when it resizes), never inside a scroll handler, so scrolling does no layout work: the scene turns
// `live.scroll` into a 0..1 progress through the active chapter with plain arithmetic (`chapterProgress`).

interface Box {
  top: number;
  height: number;
}
const boxes = new Map<number, Box>();

function measure(stage: number, el: HTMLElement) {
  const r = el.getBoundingClientRect();
  boxes.set(stage, { top: r.top + window.scrollY, height: r.height });
}

/** 0..1 progress through chapter `stage`: 0 when its top reaches the middle of the screen, 1 when its bottom does. */
export function chapterProgress(stage: number): number {
  const b = boxes.get(stage);
  if (!b || b.height <= 0) return 0;
  const p = (live.scroll + window.innerHeight * 0.5 - b.top) / b.height;
  return p < 0 ? 0 : p > 1 ? 1 : p;
}

export function Chapter({
  stage,
  id,
  label,
  className,
  children,
}: {
  stage: number;
  id?: string;
  label?: string; // id of the heading inside, for aria-labelledby
  className?: string;
  children: ReactNode;
}) {
  const ref = useRef<HTMLElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    measure(stage, el);
    const remeasure = () => {
      measure(stage, el);
      markDirty();
    };
    const ro = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(remeasure);
    ro?.observe(el);
    ro?.observe(document.body); // content above this chapter can change its position without changing its size
    let io: IntersectionObserver | null = null;
    if (typeof IntersectionObserver !== "undefined") {
      io = new IntersectionObserver(
        (entries) => {
          for (const e of entries) {
            if (e.isIntersecting) {
              live.data.stage = stage;
              markDirty();
            }
          }
        },
        { rootMargin: "-42% 0px -42% 0px", threshold: 0 },
      );
      io.observe(el);
    }
    return () => {
      ro?.disconnect();
      io?.disconnect();
      boxes.delete(stage);
    };
  }, [stage]);
  return (
    <section ref={ref} id={id} aria-labelledby={label} className={className} data-stage={stage}>
      {children}
    </section>
  );
}

/** Reset the chapter when a page unmounts, so the next world starts from its own first chapter. */
export function useResetStage() {
  useEffect(() => {
    live.data.stage = 0;
    markDirty();
    return () => {
      live.data.stage = 0;
    };
  }, []);
}
