import { useEffect, useRef, type ReactNode } from "react";
import { live, markDirty } from "./store";

/**
 * A page section that tells the 3D scene which chapter is on screen. When the section crosses the middle of the
 * viewport it sets `live.data.stage`; the scene eases toward it. Without IntersectionObserver the stage stays at 0.
 * The section is ordinary DOM: it reads the same with or without the 3D.
 */
export function Chapter({
  stage,
  id,
  label,
  className,
  children,
}: {
  stage: number;
  id?: string;
  label?: string; // aria-labelledby target id of the heading inside
  className?: string;
  children: ReactNode;
}) {
  const ref = useRef<HTMLElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof IntersectionObserver === "undefined") return;
    const io = new IntersectionObserver(
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
    return () => io.disconnect();
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
    live.data.mode = undefined;
    markDirty();
    return () => {
      live.data.stage = 0;
    };
  }, []);
}
