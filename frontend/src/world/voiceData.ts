// How the Voice page hands data to the Voice scene. The page writes; the scene reads on its next frame.
import { live, markDirty } from "./store";

export interface VoiceSceneData {
  version: number;
  samples: Float32Array | null; // the user's decoded clip (16 kHz mono); null = show the idle terrain
  verdict: "low" | "medium" | "high" | null; // set once the API has scored the clip
  notSpeech: boolean; // the API refused the clip: collapse the terrain
  lowQuality: boolean; // the API warned: draw the fog bank
  cutoffHz: number | null; // where the guard saw the sound cut off, if it said
  event: "collapse" | "fog" | null; // a demonstration chosen on the page (no clip needed)
}

const EMPTY: VoiceSceneData = { version: 0, samples: null, verdict: null, notSpeech: false, lowQuality: false, cutoffHz: null, event: null };

export function publishVoice(patch: Partial<Omit<VoiceSceneData, "version">>): void {
  const prev = (live.data.voice as VoiceSceneData | undefined) ?? EMPTY;
  live.data.voice = { ...prev, ...patch, version: prev.version + 1 };
  markDirty();
}

export function clearVoice(): void {
  live.data.voice = undefined;
}
