// The wall of drawers: which drawer holds what, and how the Evidence page tells the scene what is open.
import { live, markDirty } from "./store";

export const WALL_COLS = 6;
export const WALL_ROWS = 4;
/** Drawer index (row-major) for each exhibit, in page order. */
export const EXHIBIT_DRAWERS = [3, 8, 14, 17, 22];
/** The sealed envelope's drawer. */
export const ENVELOPE_DRAWER = 11;
/** Drawers drawn as empty and dashed: the things we could not test. */
export const EMPTY_DRAWERS = [1, 6, 9, 13, 19, 20, 23];

export interface EvidenceSceneData {
  version: number;
  open: number[]; // drawer indices that are pulled out
  envelopeOpen: boolean;
}

export function publishEvidence(patch: Partial<Omit<EvidenceSceneData, "version">>): void {
  const prev = (live.data.evidence as EvidenceSceneData | undefined) ?? { version: 0, open: [], envelopeOpen: false };
  live.data.evidence = { ...prev, ...patch, version: prev.version + 1 };
  markDirty();
}

export function clearEvidence(): void {
  live.data.evidence = undefined;
}
