import { createContext, useContext, useEffect, useState, type ReactNode } from "react";

// Two layers. Layer 1 is plain language and is what everyone sees. Layer 2 ("For engineers") holds the
// exact terms and protocol; it is collapsed by default and one toggle per page opens every block on
// that page. The choice is remembered for the browser session.

const Layer = createContext<{ open: boolean; setOpen: (v: boolean) => void }>({ open: false, setOpen: () => {} });

function read(key: string): boolean {
  try {
    return sessionStorage.getItem(key) === "1";
  } catch {
    return false;
  }
}

export function EngineerLayer({ page, children }: { page: string; children: ReactNode }) {
  const key = `tm-engineers-${page}`;
  const [open, setOpenState] = useState(() => read(key));
  useEffect(() => setOpenState(read(key)), [key]);
  const setOpen = (v: boolean) => {
    setOpenState(v);
    try {
      sessionStorage.setItem(key, v ? "1" : "0");
    } catch {
      /* private mode: the choice just isn't remembered */
    }
  };
  return <Layer.Provider value={{ open, setOpen }}>{children}</Layer.Provider>;
}

/** The one toggle per page. */
export function EngineerToggle() {
  const { open, setOpen } = useContext(Layer);
  return (
    <button type="button" className="btn btn-quiet text-sm" aria-pressed={open} onClick={() => setOpen(!open)}>
      {open ? "Hide the detail for engineers" : "Show the detail for engineers"}
    </button>
  );
}

/** A Layer 2 block. Renders nothing until the page's toggle is on. */
export function ForEngineers({ title, children }: { title: string; children: ReactNode }) {
  const { open } = useContext(Layer);
  if (!open) return null;
  return (
    <aside className="mt-6 border-l-2 border-ink-soft pl-4 sm:pl-5" data-layer="engineers" aria-label={`For engineers: ${title}`}>
      <div className="mb-3 text-xs font-semibold text-ink-soft">For engineers: {title}</div>
      <div className="text-sm">{children}</div>
    </aside>
  );
}

/** A plain-language line that says what a score means and which direction is better. */
export function Meaning({ children }: { children: ReactNode }) {
  return <p className="measure m-0 text-sm text-ink-soft">{children}</p>;
}

export const GRAPH_SCORE_MEANING = "How well it finds fraud without too many false alarms. From 0 to 1. Higher is better.";
export const VOICE_SCORE_MEANING =
  "How often it gets a recording wrong, when missed fakes and false alarms are equally common. Lower is better.";

export function Glossary({ terms }: { terms: [string, string][] }) {
  return (
    <dl className="m-0 grid gap-x-6 gap-y-2 sm:grid-cols-[11rem_minmax(0,1fr)]">
      {terms.map(([term, meaning]) => (
        <div key={term} className="contents">
          <dt className="font-semibold">{term}</dt>
          <dd className="m-0 mb-1 text-ink-soft sm:mb-0">{meaning}</dd>
        </div>
      ))}
    </dl>
  );
}
