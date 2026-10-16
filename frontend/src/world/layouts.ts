// Pure layout maths for the Home mesh: where each node sits in each "chapter" arrangement, and which nodes are
// joined by a thread. No three.js here, so it is cheap to test. All values are in scene units (the camera sits
// about nine units back, so the visible area at the origin is roughly 15 by 8.4).
import { seeded } from "./rand";

export const CHAPTERS = 5; // arrangements: 0 mesh, 1 web and wave, 2 two sheets, 3 shortcut ribbon, 4 spectrogram

export interface MeshLayout {
  count: number;
  targets: Float32Array[]; // CHAPTERS arrays of count*3
  seed: Float32Array; // count
  kind: Float32Array; // 0 normal, 1 leaked (a future node in the past sheet), 2 shortcut (leading silence), 3 above the band cut
}

export function buildMeshLayout(count: number, seed = 7): MeshLayout {
  const rand = seeded(seed);
  const targets = Array.from({ length: CHAPTERS }, () => new Float32Array(count * 3));
  const seeds = new Float32Array(count);
  const kind = new Float32Array(count);
  const half = Math.floor(count / 2);
  for (let i = 0; i < count; i++) {
    seeds[i] = rand();
    const u = rand(), v = rand(), w = rand();
    // 0: a loose ellipsoid cloud, denser toward the middle
    {
      const theta = 2 * Math.PI * u, phi = Math.acos(2 * v - 1), r = Math.cbrt(w);
      targets[0].set([7.4 * r * Math.sin(phi) * Math.cos(theta), 3.5 * r * Math.sin(phi) * Math.sin(theta), 3.6 * r * Math.cos(phi)], i * 3);
    }
    // 1: half a transaction web (a shell, left), half a waveform (three strands, right)
    if (i < half) {
      const theta = 2 * Math.PI * u, phi = Math.acos(2 * v - 1), r = 2.5 + 0.35 * w;
      targets[1].set([-4.4 + r * Math.sin(phi) * Math.cos(theta), r * Math.sin(phi) * Math.sin(theta), r * Math.cos(phi) * 0.8], i * 3);
    } else {
      const x = 1.4 + 6.4 * u, strand = Math.floor(w * 3);
      targets[1].set([x, Math.sin(x * 2.1 + strand * 1.3) * (1.5 - strand * 0.3) + (v - 0.5) * 0.28, (rand() - 0.5) * 0.7], i * 3);
    }
    // 2: two sheets, past (left) and future (right); every seventh future node leaks into the past sheet
    {
      const onRight = u > 0.5, leak = onRight && i % 7 === 0;
      const x = (onRight && !leak ? 3.4 : -3.4) + (leak ? (v - 0.5) * 0.5 : 0);
      targets[2].set([x, (w - 0.5) * 6.4, (rand() - 0.5) * 4.6], i * 3);
      kind[i] = leak ? 1 : 0;
    }
    // 3: a speech ribbon whose first stretch (the leading silence) is flat; those nodes are the shortcut
    {
      const x = -7 + 14 * u, lead = x < -5.2;
      const envelope = lead ? 0.02 : Math.min(1, (x + 5.2) / 2.2) * (0.6 + 0.4 * Math.sin(x * 0.9));
      targets[3].set([x, envelope * Math.sin(x * 5.5 + v * 0.6) * 2 + (v - 0.5) * 0.18, (w - 0.5) * 1.2], i * 3);
      if (lead && kind[i] === 0) kind[i] = 2;
    }
    // 4: a spectrogram plane (time across, frequency up); nodes above the cut-off line are the missing band
    {
      const x = -6 + 12 * u, y = -3 + 6 * v;
      targets[4].set([x, y, (w - 0.5) * 0.8], i * 3);
      if (y > 0.8 && kind[i] === 0) kind[i] = 3;
    }
  }
  return { count, targets, seed: seeds, kind };
}

/** Pairs of nearby nodes (in the resting mesh) joined by a thread. About 1.3 threads per node. */
export function buildEdges(layout: MeshLayout, radius = 1.15): Uint32Array {
  const p = layout.targets[0];
  const cell = radius;
  const grid = new Map<string, number[]>();
  const cellKey = (x: number, y: number, z: number) => `${Math.floor(x / cell)},${Math.floor(y / cell)},${Math.floor(z / cell)}`;
  for (let i = 0; i < layout.count; i++) {
    const k = cellKey(p[i * 3], p[i * 3 + 1], p[i * 3 + 2]);
    const bucket = grid.get(k);
    if (bucket) bucket.push(i);
    else grid.set(k, [i]);
  }
  const edges: number[] = [];
  const degree = new Uint8Array(layout.count);
  const r2 = radius * radius;
  for (let i = 0; i < layout.count; i++) {
    const x = p[i * 3], y = p[i * 3 + 1], z = p[i * 3 + 2];
    const cx = Math.floor(x / cell), cy = Math.floor(y / cell), cz = Math.floor(z / cell);
    for (let dx = -1; dx <= 1 && degree[i] < 2; dx++)
      for (let dy = -1; dy <= 1 && degree[i] < 2; dy++)
        for (let dz = -1; dz <= 1 && degree[i] < 2; dz++) {
          for (const j of grid.get(`${cx + dx},${cy + dy},${cz + dz}`) ?? []) {
            if (j <= i || degree[j] >= 3) continue;
            const d = (p[j * 3] - x) ** 2 + (p[j * 3 + 1] - y) ** 2 + (p[j * 3 + 2] - z) ** 2;
            if (d < r2 && degree[i] < 2) {
              edges.push(i, j);
              degree[i]++;
              degree[j]++;
            }
          }
        }
  }
  return Uint32Array.from(edges);
}
