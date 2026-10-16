// Layout maths for the Graph world ("The Ledger"): beads on a loom of copper threads. Chapters:
//   0 the loom,  1 two sheets (past and future, with a few future beads leaking into the past),
//   2 bead stacks (bars, heights from the data module),  3 a transaction's neighbourhood from the live query.
// Pure functions, no three.js.
import { seeded } from "./rand";

export const ROWS = 22;
export const HALF_WIDTH = 8.6;
const BAR_CENTRE = 4.1; // x of the middle of the bar stacks
const NEIGHBOUR_CENTRE = 4.3; // x of the neighbourhood

export interface Bead {
  t: Float32Array[]; // 4 chapter positions, count*3 each
  leak: Float32Array; // count*3: where a leaking future bead sits in the past sheet
  vis: Float32Array; // count*4: visibility in each chapter (0..1)
  tint: Float32Array; // count: colour used in chapters 2 and 3 (0 teal, 1 accent, 2 flag)
  kind: Float32Array; // count: 0 normal, 1 flagged bead (illustration), 3 leaker
  seed: Float32Array;
  row: Float32Array;
  count: number;
}

export function rowY(row: number) {
  return (row - (ROWS - 1) / 2) * 0.32;
}
export function sag(x: number) {
  return -0.22 * (1 - Math.min(1, (x / HALF_WIDTH) ** 2));
}

export function buildBeads(count: number, seedValue = 11): Bead {
  const rand = seeded(seedValue);
  const b: Bead = {
    t: Array.from({ length: 4 }, () => new Float32Array(count * 3)),
    leak: new Float32Array(count * 3),
    vis: new Float32Array(count * 4),
    tint: new Float32Array(count),
    kind: new Float32Array(count),
    seed: new Float32Array(count),
    row: new Float32Array(count),
    count,
  };
  for (let i = 0; i < count; i++) {
    const row = i % ROWS;
    const x = (rand() * 2 - 1) * HALF_WIDTH;
    const z = (row - ROWS / 2) * 0.06;
    b.seed[i] = rand();
    b.row[i] = row;
    b.t[0].set([x, rowY(row) + sag(x), z], i * 3);
    // two sheets: the first half of the rows are the past, the second half the future
    const future = row >= ROWS / 2;
    const sx = (future ? 4.9 : -4.9) + x * 0.42;
    const sheetRow = future ? row - ROWS / 2 : row;
    b.t[1].set([sx, rowY(sheetRow) * 1.45 + sag(x) * 0.3, z * 2], i * 3);
    const leaker = future && i % 7 === 0;
    b.kind[i] = leaker ? 3 : rand() < 0.03 ? 1 : 0;
    b.leak.set([-4.9 + (rand() * 2 - 1) * 3.4 * 0.42 * 2, rowY(sheetRow) * 1.45, z * 2 + 0.2], i * 3);
    // default (no data yet): the neighbourhood chapter shows a dim copy of the loom, the bars chapter shows nothing
    b.t[2].set(b.t[0].subarray(i * 3, i * 3 + 3), i * 3);
    b.t[3].set(b.t[0].subarray(i * 3, i * 3 + 3), i * 3);
    b.vis.set([1, 1, 0, 0.3], i * 4);
  }
  return b;
}

/** Lay the beads out as stacked bars, one bar per value in 0..1. Beads not in a bar are hidden in this chapter. */
export function fillBars(b: Bead, bars: { value: number; tint: number }[]): void {
  const perUnit = 56;
  const spacing = 0.105;
  let cursor = 0;
  const k = Math.max(bars.length, 1);
  b.vis.forEach((_, j) => {
    if (j % 4 === 2) b.vis[j] = 0;
  });
  bars.forEach((bar, bi) => {
    const n = Math.round(Math.max(0, Math.min(1, bar.value)) * perUnit);
    const x0 = BAR_CENTRE + (bi - (k - 1) / 2) * 0.6; // the bars sit on the right; the page panel is on the left
    for (let j = 0; j < n && cursor < b.count; j++, cursor++) {
      const col = j % 3;
      b.t[2].set([x0 + (col - 1) * 0.17, -3 + Math.floor(j / 3) * spacing * 3.6, 0], cursor * 3);
      b.vis[cursor * 4 + 2] = 1;
      b.tint[cursor] = bar.tint;
    }
  });
}

/** Lay the beads out as a neighbourhood: the transaction in the middle, neighbours on rings by number of hops. */
export function fillNeighbours(b: Bead, centre: { flagged: boolean } | null, neighbours: { hops: number; flagged: boolean }[]): void {
  for (let i = 0; i < b.count; i++) b.vis[i * 4 + 3] = centre ? 0 : 0.3;
  if (!centre) {
    for (let i = 0; i < b.count; i++) b.t[3].set(b.t[0].subarray(i * 3, i * 3 + 3), i * 3);
    return;
  }
  b.t[3].set([NEIGHBOUR_CENTRE, 0, 0.5], 0);
  b.vis[3] = 1;
  b.tint[0] = centre.flagged ? 2 : 1;
  const perRing = new Map<number, number>();
  neighbours.slice(0, b.count - 1).forEach((n, idx) => {
    const i = idx + 1;
    const ring = Math.max(1, n.hops);
    const k = perRing.get(ring) ?? 0;
    perRing.set(ring, k + 1);
    const angle = k * 2.399963 + ring; // golden angle: an even spread without a grid
    const r = 0.95 * ring + (k % 3) * 0.08;
    b.t[3].set([NEIGHBOUR_CENTRE + Math.cos(angle) * r, Math.sin(angle) * r * 0.85, ((k % 5) - 2) * 0.1], i * 3);
    b.vis[i * 4 + 3] = 1;
    b.tint[i] = n.flagged ? 2 : 0;
  });
  for (let i = neighbours.length + 1; i < b.count; i++) b.vis[i * 4 + 3] = 0;
}

/** Thread polylines for the loom: ROWS rows of `segments` segments, as vertex pairs (x, row) for a line-list. */
export function buildThreads(segments = 36): { x: Float32Array; row: Float32Array } {
  const n = ROWS * segments * 2;
  const x = new Float32Array(n);
  const row = new Float32Array(n);
  let k = 0;
  for (let r = 0; r < ROWS; r++)
    for (let s = 0; s < segments; s++) {
      for (const e of [s, s + 1]) {
        x[k] = (e / segments) * 2 * HALF_WIDTH - HALF_WIDTH;
        row[k] = r;
        k++;
      }
    }
  return { x, row };
}
