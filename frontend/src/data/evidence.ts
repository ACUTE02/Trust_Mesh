// The single data module. Every number a page shows is looked up here by id and rendered through
// <Num>; nothing is typed inline in a component. scripts/check_frontend_numbers.py re-checks each
// entry against its source file, and src/test/numbers.test.tsx fails the build if a page renders a
// number that did not come from here (or live from the API).
import raw from "./evidence.json";

export type Phase = "P1" | "P1B" | "P1C" | "P1D" | "P2" | "P2B" | "P2C" | "P2D" | "INFRA";

export interface Evidence {
  value: number | null; // null = not yet run
  decimals: number;
  spread?: number;
  interval?: [number, number];
  signed?: boolean;
  unit?: string;
  suffix?: string;
  plain?: boolean; // an identifier, shown without thousands separators
  metric?: string;
  style?: "measured" | "shortcut" | "planned";
  measures: string;
  runDate: string;
  phase: Phase;
  n?: string;
  note?: string;
  derived?: boolean; // a plain-language restatement of another entry (see source.derivedFrom and formula)
  source: { file: string; key?: string; text?: string; derivedFrom?: string; scale?: number; formula?: string };
}

export type EvidenceId = keyof typeof raw.entries;

const entries = raw.entries as unknown as Record<string, Evidence>;

export function evidence(id: EvidenceId): Evidence {
  const e = entries[id];
  if (!e) throw new Error(`No evidence entry "${id}" in the data module`);
  return e;
}

export function allEvidence(): [string, Evidence][] {
  return Object.entries(entries);
}

function fixed(value: number, e: Evidence): string {
  if (e.decimals === 0) return e.plain ? String(value) : value.toLocaleString("en-US");
  return value.toFixed(e.decimals);
}

/** The value as the page shows it, e.g. "0.7482", "+0.2361", "203,769", "77%". */
export function formatValue(e: Evidence): string {
  if (e.value === null) return "not yet run";
  const sign = e.signed && e.value > 0 ? "+" : e.value < 0 ? "−" : "";
  return `${sign}${fixed(Math.abs(e.value), e)}${e.suffix ?? ""}${e.unit ?? ""}`;
}

export function formatSpread(e: Evidence): string | null {
  return e.spread === undefined ? null : `± ${e.spread.toFixed(e.decimals)}`;
}

export function formatInterval(e: Evidence): string | null {
  if (!e.interval) return null;
  // Bounds recorded to two decimals (the small In-the-Wild sample) are shown to two; others as the value is.
  const coarse = e.interval.every((v) => Number(v.toFixed(2)) === v);
  const d = coarse ? 2 : e.decimals;
  const f = (v: number) => `${v < 0 ? "−" : ""}${Math.abs(v).toFixed(d)}`;
  return `${f(e.interval[0])} to ${f(e.interval[1])}`;
}

/** Where a number came from, for the marker's source note. */
export function sourceLine(e: Evidence): string {
  if (e.source.derivedFrom) return `derived from ${e.source.derivedFrom}: ${e.source.formula}`;
  const file = e.source.file.replace(/^\.\.\//, "");
  return `${file}${e.source.key ? ` → ${e.source.key.replace(/ > /g, ".")}` : ""}`;
}

export const PHASE_NAMES: Record<Phase, string> = {
  P1: "Phase 1: first graph run",
  P1B: "Phase 1B: graph depth pass",
  P1C: "Phase 1C: more architectures and a wider search",
  P1D: "Phase 1D: full graph leaderboard",
  P2: "Phase 2: first voice run",
  P2B: "Phase 2B: voice depth pass",
  P2C: "Phase 2C: voice head ablation",
  P2D: "Phase 2D: voice head trained on compressed copies",
  INFRA: "Serving layer",
};
