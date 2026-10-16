// How the case desk hands the current case to the Balance scene. The page writes; the scene reads on its next frame.
// In recording mode `masked` is true and the transaction's score is not passed on at all.
import { live, markDirty } from "./store";

export interface CaseSceneData {
  version: number;
  graph: number | null; // 0..1, null when there is none or when masked
  voice: number | null; // 0..1 fake-voice score of the user's own recording
  masked: boolean;
  weightGraph: number;
  weightVoice: number;
}

export function publishCase(patch: Partial<Omit<CaseSceneData, "version">>): void {
  const prev = (live.data.case as CaseSceneData | undefined) ?? { version: 0, graph: null, voice: null, masked: false, weightGraph: 0.5, weightVoice: 0.5 };
  live.data.case = { ...prev, ...patch, version: prev.version + 1 };
  markDirty();
}

export function clearCase(): void {
  live.data.case = undefined;
}
