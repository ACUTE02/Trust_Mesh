import { Fragment, useId, useState, type ReactNode } from "react";
import {
  PHASE_NAMES,
  evidence,
  formatInterval,
  formatSpread,
  formatValue,
  sourceLine,
  type EvidenceId,
} from "../data/evidence";

/**
 * Tabular digits, proportional punctuation. Schibsted Grotesk's tabular-figures feature also widens the
 * full stop and the comma ("0 . 7730", "203 , 769"), so only the runs of digits get the feature.
 */
export function Digits({ text }: { text: string }) {
  return (
    <>
      {text.split(/(\d+)/).map((part, i) =>
        i % 2 ? (
          <span key={i} className="tn">
            {part}
          </span>
        ) : (
          <Fragment key={i}>{part}</Fragment>
        ),
      )}
    </>
  );
}

/** A number from the data module. The only way a result is rendered. */
export function Num({ id, spread = true, interval = false }: { id: EvidenceId; spread?: boolean; interval?: boolean }) {
  const e = evidence(id);
  if (e.value === null) {
    return (
      <span data-evidence={id} className="planned px-1 text-ink-soft">
        not yet run
      </span>
    );
  }
  const s = spread ? formatSpread(e) : null;
  const i = interval ? formatInterval(e) : null;
  return (
    <span data-evidence={id} className="num">
      <Digits text={formatValue(e)} />
      {s && <Digits text={` ${s}`} />}
      {i && <Digits text={` (95% interval ${i})`} />}
    </span>
  );
}

/** Live value from the API (never from the repository). Marked so the number test can tell the two apart. */
export function Live({ children, title }: { children: ReactNode; title?: string }) {
  return (
    <span data-live className="num" title={title}>
      {typeof children === "string" || typeof children === "number" ? <Digits text={String(children)} /> : children}
    </span>
  );
}

/**
 * Provenance marker: a yellow tag carrying the run code. It opens a short source note: which run,
 * which file, how many seeds or samples. It is information, not a counter.
 */
export function Marker({ ids }: { ids: EvidenceId[] }) {
  const [open, setOpen] = useState(false);
  const noteId = useId();
  const phases = [...new Set(ids.map((id) => evidence(id).phase))];
  return (
    <span className="inline-block align-top">
      <button
        type="button"
        className="marker"
        aria-expanded={open}
        aria-controls={noteId}
        onClick={() => setOpen((v) => !v)}
        title="Show where these numbers come from"
      >
        <span className="sr-only">Source: run </span>
        {phases.join(" ")}
      </button>
      {open && (
        <span id={noteId} role="note" className="mt-2 block bg-sheet p-3 text-xs leading-snug text-ink" style={{ maxWidth: "34rem" }}>
          {ids.map((id) => {
            const e = evidence(id);
            return (
              <span key={id} className="mb-2 block last:mb-0">
                <span className="block font-semibold">
                  <Num id={id} /> {e.metric ? `${e.metric}: ` : ": "}
                  <span className="font-normal">{e.measures}.</span>
                </span>
                <span className="block text-ink-soft">
                  {PHASE_NAMES[e.phase]}, run {e.runDate}
                  {e.n ? `, ${e.n}` : ""}.
                </span>
                <span className="block break-all font-mono text-[0.6875rem] text-ink-soft">{sourceLine(e)}</span>
                {e.note && <span className="block text-ink-soft">{e.note}</span>}
              </span>
            );
          })}
        </span>
      )}
    </span>
  );
}

/**
 * The one way a finding is laid out: marker in the left margin, one claim sentence, the figure,
 * a source line. On narrow screens the margin collapses and the marker sits above its row.
 */
export function EvidenceRow({
  ids,
  claim,
  children,
  more,
}: {
  ids: EvidenceId[];
  claim: ReactNode;
  children?: ReactNode;
  more?: ReactNode;
}) {
  return (
    <div className="rule-t grid gap-x-6 gap-y-2 py-6 md:grid-cols-[5.5rem_minmax(0,1fr)]">
      <div>
        <Marker ids={ids} />
      </div>
      <div>
        <p className="measure m-0 text-lg leading-snug">{claim}</p>
        {children && <div className="mt-4">{children}</div>}
        {more && <p className="measure mt-3 mb-0 text-sm text-ink-soft">{more}</p>}
      </div>
    </div>
  );
}

export function CaveatNote({ children, title }: { children: ReactNode; title?: string }) {
  return (
    <p role="note" className="measure m-0 border-l-4 border-flag bg-sheet px-3 py-2 text-sm">
      {title && <strong className="font-semibold">{title} </strong>}
      {children}
    </p>
  );
}

export function Section({ title, children, id }: { title: string; children: ReactNode; id?: string }) {
  return (
    <section id={id} className="mt-16" aria-labelledby={id ? `${id}-h` : undefined}>
      <h2 id={id ? `${id}-h` : undefined} className="mb-5 text-xl">
        {title}
      </h2>
      {children}
    </section>
  );
}

/** Explains the pattern language once. Every pattern is also named in words where it is used. */
export function Legend() {
  return (
    <ul className="m-0 grid list-none gap-x-8 gap-y-2 p-0 text-sm sm:grid-cols-2">
      <li className="flex items-center gap-3">
        <span className="fill-ledger inline-block h-3 w-10" aria-hidden /> Solid: a measured result
      </li>
      <li className="flex items-center gap-3">
        <span className="fill-shortcut inline-block h-3 w-10" aria-hidden /> Hatched: a shortcut, not a result
      </li>
      <li className="flex items-center gap-3">
        <span className="planned inline-block h-3 w-10" aria-hidden /> Dashed: designed or planned, not built or not run
      </li>
      <li className="flex items-center gap-3">
        <span className="bracket inline-block w-10" aria-hidden /> Bracket: a seed spread or a confidence interval
      </li>
      <li className="flex items-center gap-3 sm:col-span-2">
        <span className="marker" aria-hidden>
          P1D
        </span>
        Yellow tag: the run a row's numbers come from. Select it to see the file.
      </li>
    </ul>
  );
}
