import { Fragment, useEffect, useRef, useState, type ElementType } from "react";
import { readPrefs } from "../world/prefs";

// Kinetic type that keeps the text real: the words stay in the DOM as ordinary text, so reading, searching and
// screen readers work. "words" reveals word by word, "wipe" wipes the whole line in, "scramble" settles
// random letters into the real ones (used only for short labels, and the real text is always in the DOM).
export function KineticText({
  text,
  as: Tag = "span",
  mode = "words",
  className,
  delay = 0,
}: {
  text: string;
  as?: ElementType;
  mode?: "words" | "wipe" | "scramble";
  className?: string;
  delay?: number;
}) {
  const ref = useRef<HTMLElement>(null);
  const [shown, setShown] = useState(text);
  const [visible, setVisible] = useState(() => typeof IntersectionObserver === "undefined" || readPrefs().reduced);

  useEffect(() => {
    if (visible || !ref.current) return;
    const io = new IntersectionObserver(([entry]) => entry.isIntersecting && (setVisible(true), io.disconnect()), { threshold: 0.2 });
    io.observe(ref.current);
    return () => io.disconnect();
  }, [visible]);

  useEffect(() => {
    if (mode !== "scramble" || !visible || readPrefs().reduced) return;
    const glyphs = "abcdefghijklmnopqrstuvwxyz";
    let step = 0;
    const id = setInterval(() => {
      step++;
      setShown(text.split("").map((c, i) => (c === " " || i < step / 2 ? c : glyphs[(i * 7 + step * 3) % glyphs.length])).join(""));
      if (step / 2 >= text.length) {
        clearInterval(id);
        setShown(text);
      }
    }, 28);
    return () => clearInterval(id);
  }, [mode, visible, text]);

  if (mode === "scramble") {
    return (
      <Tag ref={ref} className={className}>
        <span className="sr-only">{text}</span>
        <span aria-hidden>{shown}</span>
      </Tag>
    );
  }
  const words = text.split(" ");
  return (
    <Tag ref={ref} className={className} data-kinetic={mode} data-visible={visible ? "1" : "0"}>
      {words.map((w, i) => (
        <Fragment key={i}>
          <span className="kt-mask">
            <span className="kt-word" style={{ transitionDelay: `${delay + (mode === "wipe" ? 0 : i * 45)}ms` }}>
              {w}
            </span>
          </span>
          {i < words.length - 1 ? " " : ""}
        </Fragment>
      ))}
    </Tag>
  );
}
