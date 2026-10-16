import { useEffect, useRef, useState } from "react";
import { NavLink } from "react-router-dom";
import { markDirty, setUi } from "../world/store";
import { NAV_ORDER, WORLDS } from "../world/worlds";

const LABEL: Record<string, string> = {
  home: "Overview",
  graph: "Graph Engine",
  voice: "Voice Engine",
  case: "Case desk",
  evidence: "Evidence",
  about: "About",
};

function ThemeToggle() {
  const [theme, setTheme] = useState(() => document.documentElement.dataset.theme ?? "light");
  const next = theme === "dark" ? "light" : "dark";
  return (
    <button
      type="button"
      className="cursor-pointer border-0 bg-transparent p-0 text-sm underline"
      onClick={() => {
        document.documentElement.dataset.theme = next;
        try {
          localStorage.setItem("tm-theme", next);
        } catch {
          /* private mode: the choice just is not remembered */
        }
        setTheme(next);
        setUi({ theme: next });
        // The stage watches data-theme and re-reads the palette itself; here it only has to draw one more frame.
        requestAnimationFrame(markDirty);
      }}
    >
      Switch to {next} theme
    </button>
  );
}

/** The thread: one thin line under the header, in the thread colour of the world you are in. */
function ThreadLine() {
  return (
    <div className="thread-line" aria-hidden="true">
      <svg viewBox="0 0 1200 6" preserveAspectRatio="none">
        <path d="M0 3 C 200 0, 400 6, 600 3 S 1000 0, 1200 3" stroke="var(--thread)" strokeWidth="1.5" fill="none" vectorEffect="non-scaling-stroke" />
      </svg>
    </div>
  );
}

export function Header() {
  const [open, setOpen] = useState(false);
  const panel = useRef<HTMLDivElement>(null);
  const button = useRef<HTMLButtonElement>(null);
  const link = ({ isActive }: { isActive: boolean }) =>
    `py-1 no-underline ${isActive ? "border-b-2 border-ink font-semibold" : "border-b-2 border-transparent hover:border-ink-soft"}`;

  // The mobile menu traps focus while open; Escape closes it and returns focus to its button.
  useEffect(() => {
    if (!open) return;
    const items = () => [...(panel.current?.querySelectorAll<HTMLElement>("a, button") ?? [])];
    items()[0]?.focus();
    const key = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setOpen(false);
        button.current?.focus();
      } else if (e.key === "Tab") {
        const els = [button.current as HTMLElement, ...items()];
        const i = els.indexOf(document.activeElement as HTMLElement);
        const to = e.shiftKey ? (i <= 0 ? els.length - 1 : i - 1) : i === els.length - 1 ? 0 : i + 1;
        e.preventDefault();
        els[to].focus();
      }
    };
    document.addEventListener("keydown", key);
    return () => document.removeEventListener("keydown", key);
  }, [open]);

  return (
    <header className="site-header">
      <div className="relative mx-auto flex min-h-24 max-w-[76rem] flex-wrap items-center gap-x-8 gap-y-2 px-5 py-4 sm:px-8 md:min-h-0">
        <NavLink to="/" className="text-lg font-semibold no-underline" style={{ fontFamily: "var(--font-display)" }}>
          TrustMesh
        </NavLink>
        <nav aria-label="Pages" className="hidden gap-6 text-sm md:flex">
          {NAV_ORDER.map((id) => (
            <NavLink key={id} to={WORLDS[id].path} end={id === "home"} className={link} data-cursor={WORLDS[id].nav}>
              {LABEL[id]}
            </NavLink>
          ))}
        </nav>
        <button ref={button} type="button" className="btn btn-quiet md:hidden" aria-expanded={open} aria-controls="mobile-menu" onClick={() => setOpen((v) => !v)}>
          {open ? "Close menu" : "Menu"}
        </button>
        <div className="ml-auto">
          <ThemeToggle />
        </div>
        {open && (
          <div id="mobile-menu" ref={panel} className="panel basis-full md:hidden">
            <nav aria-label="Pages, mobile" className="grid gap-3 text-base">
              {NAV_ORDER.map((id) => (
                <NavLink key={id} to={WORLDS[id].path} end={id === "home"} className="no-underline" onClick={() => setOpen(false)}>
                  {LABEL[id]}
                </NavLink>
              ))}
            </nav>
          </div>
        )}
        <ThreadLine />
      </div>
    </header>
  );
}
