// Small shared state. `live` is plain mutable data the 3D code reads every frame (no React, no allocation);
// `ui` is a tiny subscribable store for the few things React has to react to.
import { useSyncExternalStore } from "react";

export type Tier = "low" | "medium" | "high";

/** Read by scenes inside the render loop. Written by Lenis, pointer handlers and the governor. */
export const live = {
  scroll: 0, // pixels
  progress: 0, // 0..1 over the whole document
  velocity: 0,
  px: 0, // pointer, -1..1 across the viewport
  py: 0,
  dirty: true, // something changed: render one more frame
  tier: "low" as Tier,
  reduced: false,
  // set by pages, read by scenes
  data: {} as Record<string, unknown>,
  time: 0, // seconds, set by the Stage each tick
  fpsCap: 0, // 0 = every tick; a scene may ask for 30 when its motion is slow
  continuous: false, // the active scene animates by itself (drift), so draw every tick
};

export function markDirty() {
  live.dirty = true;
}

interface UiState {
  tier: Tier;
  webgl: boolean; // false until the stage proves it can draw; false again if the context is lost
  stageReady: boolean;
  theme: string;
  world: string;
}

let state: UiState = { tier: "low", webgl: false, stageReady: false, theme: "light", world: "home" };
const listeners = new Set<() => void>();

export function setUi(patch: Partial<UiState>) {
  state = { ...state, ...patch };
  if (patch.tier) live.tier = patch.tier;
  markDirty();
  listeners.forEach((l) => l());
}

export function getUi() {
  return state;
}

export function useUi<T>(select: (s: UiState) => T): T {
  return useSyncExternalStore(
    (cb) => {
      listeners.add(cb);
      return () => listeners.delete(cb);
    },
    () => select(state),
  );
}
