// Quality tiers. The governor (Stage.tsx) starts on "low" and moves up only when the frame rate holds.
import type { Tier } from "./store";

export interface TierSpec {
  dpr: number; // maximum device pixel ratio
  home: number; // mesh nodes
  graph: number; // beads on the loom
  voiceCols: number; // terrain grid, columns
  voiceRows: number;
  post: boolean; // optional single grain pass (not required to read the page)
  antialias: boolean;
}

export const TIERS: Record<Tier, TierSpec> = {
  high: { dpr: 1.75, home: 6000, graph: 3000, voiceCols: 256, voiceRows: 128, post: true, antialias: true },
  medium: { dpr: 1.5, home: 3000, graph: 1500, voiceCols: 128, voiceRows: 64, post: false, antialias: true },
  low: { dpr: 1.25, home: 1200, graph: 600, voiceCols: 64, voiceRows: 32, post: false, antialias: false },
};

export const TIER_ORDER: Tier[] = ["low", "medium", "high"];

/** Touch screens and weak devices start on low and are capped at medium. */
export function tierCap(touch: boolean, lowPower: boolean): Tier {
  return touch || lowPower ? "medium" : "high";
}
