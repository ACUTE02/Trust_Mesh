import type { ReactNode } from "react";
import { evidence, formatInterval, formatSpread, formatValue, type EvidenceId } from "../data/evidence";
import { Digits, Marker } from "./evidence";

export interface PlotRow {
  id: EvidenceId; // the dot
  label: string;
  reference?: boolean; // the model everything else is read against
  seedMeanId?: EvidenceId; // a 3-seed mean drawn as a bracket around its spread
  annotation?: ReactNode;
}

/**
 * A sorted dot-and-interval chart built as a real table: each row is label, track, value. So the
 * chart is its own text equivalent, and on a narrow screen it reads as a sorted list with inline bars.
 * Solid dot = measured, hatched = shortcut, bracket = seed spread or confidence interval.
 */
export function DotPlot({
  caption,
  rows,
  axisMax,
  axisLabel,
  hue,
  signedAxis = false,
}: {
  caption: string;
  rows: PlotRow[];
  axisMax: number;
  axisLabel: string;
  hue: "ledger" | "spectrum";
  signedAxis?: boolean;
}) {
  const pct = (v: number) => `${Math.max(0, Math.min(100, (v / axisMax) * 100))}%`;
  return (
    <table className="w-full border-collapse text-sm">
      <caption className="mb-3 text-left text-sm text-ink-soft">
        {caption} Axis: {axisLabel}, from 0 to <span data-axis className="num">{axisMax}</span>.
      </caption>
      <thead className="sr-only">
        <tr>
          <th scope="col">Source</th>
          <th scope="col">Model or condition</th>
          <th scope="col">Position on the axis</th>
          <th scope="col">Value</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => {
          const e = evidence(r.id);
          const mean = r.seedMeanId ? evidence(r.seedMeanId) : null;
          const band =
            mean && mean.spread !== undefined && mean.value !== null
              ? ([mean.value - mean.spread, mean.value + mean.spread] as const)
              : e.interval ?? (e.spread !== undefined && e.value !== null ? ([e.value - e.spread, e.value + e.spread] as const) : null);
          const shortcut = e.style === "shortcut";
          const value = e.value ?? 0;
          return (
            <tr key={r.id} className="rule-t align-middle">
              <td className="w-[4.2rem] py-2 pr-2 align-top">
                <Marker ids={r.seedMeanId ? [r.id, r.seedMeanId] : [r.id]} />
              </td>
              <th scope="row" className={`w-[34%] py-2 pr-3 text-left leading-snug ${r.reference ? "font-semibold" : "font-normal"}`}>
                {r.label}
                {r.reference && <span className="block text-xs font-normal text-ink-soft">reference</span>}
                {shortcut && <span className="block text-xs font-normal text-flag">shortcut, not a result</span>}
              </th>
              <td className="py-2 pr-3" aria-hidden>
                <div className="relative h-5">
                  <div className="absolute top-1/2 right-0 left-0 border-t border-track" />
                  {band && (
                    <div
                      className="bracket absolute top-1/2 -translate-y-1/2"
                      style={{ left: pct(Math.max(0, band[0])), width: `calc(${pct(band[1])} - ${pct(Math.max(0, band[0]))})`, minWidth: "5px" }}
                    />
                  )}
                  <div
                    className={`absolute top-1/2 h-3.5 w-3.5 -translate-x-1/2 -translate-y-1/2 ${
                      shortcut ? "fill-shortcut" : hue === "ledger" ? "fill-ledger" : "fill-spectrum"
                    } ${shortcut ? "" : "rounded-full"}`}
                    style={{ left: pct(signedAxis ? Math.max(0, value) : value) }}
                  />
                </div>
              </td>
              <td className="num w-[1%] py-2 text-right whitespace-nowrap" data-evidence={r.id}>
                <Digits text={formatValue(e)} />
                {mean ? (
                  <span className="block text-xs text-ink-soft" data-evidence={r.seedMeanId}>
                    <Digits text={`${formatValue(mean)} ${formatSpread(mean)}`} />, {mean.n}
                  </span>
                ) : formatSpread(e) ? (
                  <span className="block text-xs text-ink-soft">
                    <Digits text={formatSpread(e) ?? ""} />, {e.n}
                  </span>
                ) : e.interval ? (
                  <span className="block text-xs text-ink-soft"><Digits text={formatInterval(e) ?? ""} /></span>
                ) : null}
                {r.annotation && <span className="block text-xs whitespace-normal text-ink-soft">{r.annotation}</span>}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
