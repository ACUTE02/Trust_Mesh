import { useEffect, useId, useState } from "react";
import { evidence, type EvidenceId } from "../data/evidence";
import { Marker, Num } from "./evidence";

interface Side {
  label: string; // the control's label, e.g. "Random split"
  id: EvidenceId;
}

/**
 * One model, scored two ways. A two-option control switches the state; the bar and the readout
 * follow. This is the page's argument, so it is the only place large type and motion are spent.
 * Keyboard: the control is a radio group (arrow keys switch). Reduced motion: the bar just appears.
 */
export function TwoScoresInstrument({
  title,
  first,
  second,
  differenceId,
  axisLabel,
  axisMax,
  hue,
  ticks,
  lines,
  meaning,
  shortcutNote = "This score is a shortcut, not a real result.",
}: {
  title: string;
  first: Side;
  second: Side;
  differenceId?: EvidenceId;
  axisLabel: string;
  axisMax: EvidenceTick;
  hue: "ledger" | "spectrum";
  ticks: EvidenceTick[];
  lines: string[]; // fixed sentences under the instrument
  meaning: string; // what the score means, in plain words, and which direction is better
  shortcutNote?: string;
}) {
  const name = useId();
  const [which, setWhich] = useState<0 | 1>(0);
  const [drawn, setDrawn] = useState(false); // the single page-load motion: the bar draws to its first value
  useEffect(() => {
    const t = requestAnimationFrame(() => setDrawn(true));
    return () => cancelAnimationFrame(t);
  }, []);

  const sides = [first, second];
  const active = sides[which];
  const e = evidence(active.id);
  const other = evidence(sides[1 - which].id);
  const pct = (v: number) => `${Math.min(100, (v / axisMax.at) * 100)}%`;
  const value = e.value ?? 0;
  const otherValue = other.value ?? 0;
  const shortcut = e.style === "shortcut";
  const lo = Math.min(value, otherValue);
  const hi = Math.max(value, otherValue);

  return (
    <figure className="m-0 bg-sheet p-5 sm:p-7" aria-label={title}>
      <figcaption className="mb-4 text-sm text-ink-soft">{title}</figcaption>

      <div role="radiogroup" aria-label="How the model was scored" className="mb-6 inline-flex border-[1.5px] border-ink">
        {sides.map((s, i) => (
          <label
            key={s.id}
            className={`cursor-pointer px-4 py-2 text-sm font-semibold has-[:focus-visible]:outline-3 has-[:focus-visible]:outline-offset-2 has-[:focus-visible]:outline-ink ${
              which === i ? "bg-ink text-slab" : "bg-transparent text-ink"
            }`}
          >
            <input
              type="radio"
              name={name}
              className="sr-only"
              checked={which === i}
              onChange={() => setWhich(i as 0 | 1)}
            />
            {s.label}
          </label>
        ))}
      </div>

      <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1" aria-live="polite">
        <span className="font-semibold leading-none" style={{ fontSize: "var(--text-readout)" }}>
          <Num id={active.id} spread={false} />
        </span>
        {shortcut && <span className="text-sm font-semibold text-flag">{shortcutNote}</span>}
        {which === 1 && differenceId && (
          <span className="text-base">
            a difference of <Num id={differenceId} />
          </span>
        )}
      </div>

      <p className="measure mt-2 mb-0 text-sm text-ink-soft">{meaning}</p>

      {/* The axis. The bar is the selected state; a thin tick keeps the other state in view. */}
      <div className="mt-5" aria-hidden>
        <div className="relative h-9 bg-track">
          <div
            className={`bar-anim absolute inset-y-0 left-0 ${shortcut ? "fill-shortcut" : hue === "ledger" ? "fill-ledger" : "fill-spectrum"}`}
            style={{ width: drawn ? pct(value) : "0%" }}
          />
          <div className="absolute inset-y-0 w-[2px] bg-ink" style={{ left: pct(otherValue) }} />
          {which === 1 && differenceId && (
            <div className="bracket absolute -bottom-5" style={{ left: pct(lo), width: `calc(${pct(hi)} - ${pct(lo)})` }} />
          )}
        </div>
        <div className="relative mt-6 h-5 text-xs text-ink-soft">
          {ticks.map((t) => (
            <span key={t.label} data-axis className="num absolute -translate-x-1/2" style={{ left: pct(t.at) }}>
              {t.label}
            </span>
          ))}
        </div>
        <div className="text-xs text-ink-soft">{axisLabel}</div>
      </div>
      <p className="sr-only">
        {first.label}: <Num id={first.id} spread={false} />. {second.label}: <Num id={second.id} spread={false} />.
      </p>

      <div className="measure mt-5 grid gap-1.5 text-sm">
        {lines.map((line, i) => (
          <p key={line} className={`m-0 ${i === 0 ? "font-semibold" : ""}`}>
            {line}
          </p>
        ))}
      </div>
      <div className="mt-4">
        <Marker ids={differenceId ? [first.id, second.id, differenceId] : [first.id, second.id]} />
      </div>
    </figure>
  );
}

/** An axis position and its label. Labels are axis furniture ("0", "0.5", "1"), not results. */
export interface EvidenceTick {
  at: number;
  label: string;
}
