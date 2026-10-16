// How the Graph page hands data to the Graph scene. The page writes; the scene reads on its next frame.
import type { GraphSceneData } from "./scenes/GraphScene";
import { live, markDirty } from "./store";

export function publishGraph(patch: Partial<Omit<GraphSceneData, "version">>): void {
  const prev = (live.data.graph as GraphSceneData | undefined) ?? { version: 0, bars: [], centre: null, neighbours: [] };
  live.data.graph = { ...prev, ...patch, version: prev.version + 1 };
  markDirty();
}

export function clearGraph(): void {
  live.data.graph = undefined;
}
