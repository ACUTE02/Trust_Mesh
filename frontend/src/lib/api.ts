// Thin client for the FastAPI service. Shapes follow http://localhost:8000/docs (serving/api.py).
// The browser always calls /api/...; Vite (dev) or nginx (Docker) forwards it.

export type Verdict = "low" | "medium" | "high";

export interface Health {
  graph: { ready: boolean; model: string };
  voice: {
    ready: boolean;
    requested_default?: string;
    serving_default?: string;
    weights?: Record<string, string | null>;
    fallback_active?: boolean;
    loaded?: string[];
    opt_in?: Record<string, boolean>;
    model_labels?: Record<string, string>;
    error?: string | null;
  };
  neo4j: { connected: boolean; nodes?: number; edges?: number };
  recording_mode?: boolean;
  limits?: { max_upload_mb: number; max_audio_seconds: number; scoring_requests_per_minute: number };
}

export interface GraphScore {
  node_id: number;
  tx_id: number;
  time_step: number;
  score: number;
  verdict: Verdict;
  model: string;
  thresholds: { medium: number; high: number };
  ground_truth: "licit" | "illicit" | "unknown";
  split: "train" | "val" | "test";
  in_sample: boolean;
  graph_features?: { degree: number; out_in_degree_ratio: number; pagerank: number; clustering: number };
}

export interface GraphExplain {
  available: boolean;
  label?: string;
  gat_probability?: number;
  n_incoming_neighbors?: number;
  top_attended_in_neighbors?: { node_id: number; attention: number; xgb_score: number; ground_truth: string }[];
  note?: string | null;
}

export interface Neighbour {
  node_id: number;
  hops: number;
  ground_truth: string;
  score: number;
  verdict: Verdict;
}

/** What the service's audio guard measured (serving/audio_guard.py). */
export interface AudioQuality {
  analysed: boolean;
  loud_frames: number;
  speech_like: boolean;
  low_quality_audio: boolean;
  reasons: string[];
  spectral_movement: number;
  hf_to_mid_db: number | null;
  cutoff_hz: number | null;
  thresholds: { spectral_movement_min: number; hf_to_mid_min_db: number; cutoff_min_hz: number };
}

export interface VoiceScore {
  clip: string;
  score: number;
  verdict: Verdict;
  model: string;
  model_label?: string;
  model_requested: string;
  fallback_used: boolean;
  thresholds: { medium: number; high: number; source: string; single_band?: boolean };
  duration_s: number;
  speech_s: number;
  lead_silence_s?: number;
  trail_silence_s?: number;
  scored_seconds?: number;
  low_confidence?: boolean;
  warnings?: string[];
  low_quality_audio?: boolean;
  audio_quality?: AudioQuality | null;
}

export interface CaseScore {
  case_id: number | null;
  graph: GraphScore;
  voice: VoiceScore;
  visual: null | { score: number };
  fusion: { combined_score: number; combined_verdict: Verdict; weights: Record<string, number>; caveat: string };
}

/** An error whose message is already written for the reader: what happened and what to do. */
export class ApiError extends Error {
  constructor(message: string, readonly status?: number) {
    super(message);
  }
}

/** One request as the browser saw it. */
export interface ApiCall {
  method: string;
  path: string;
  status: number | null;
  ms: number;
  body: unknown;
}

let callLog: ApiCall[] = [];
/** Start a fresh log before a desk action; read it afterwards. */
export function startCallLog(): void {
  callLog = [];
}
export function readCallLog(): ApiCall[] {
  return [...callLog];
}

const UNREACHABLE =
  "The scoring service didn't answer. Start the stack with docker compose up and try again.";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const method = init?.method ?? "GET";
  const started = performance.now();
  let response: Response;
  try {
    response = await fetch(`/api${path}`, init);
  } catch {
    callLog.push({ method, path, status: null, ms: performance.now() - started, body: null });
    throw new ApiError(UNREACHABLE);
  }
  let body: unknown = null;
  let parsed = true;
  try {
    body = await response.json();
  } catch {
    parsed = false;
  }
  callLog.push({ method, path, status: response.status, ms: performance.now() - started, body });
  if (response.status === 502 || response.status === 504) throw new ApiError(UNREACHABLE, response.status);
  if (!parsed && !response.ok) throw new ApiError(UNREACHABLE, response.status);
  if (!response.ok) {
    const detail = (body as { detail?: unknown } | null)?.detail;
    throw new ApiError(typeof detail === "string" ? detail : `The service answered with an error (${response.status}).`, response.status);
  }
  return body as T;
}

function audioForm(wav: Blob, name: string, extra?: Record<string, string>): FormData {
  const form = new FormData();
  for (const [k, v] of Object.entries(extra ?? {})) form.append(k, v);
  form.append("file", new File([wav], name, { type: "audio/wav" }));
  return form;
}

export const api = {
  health: () => request<Health>("/health"),
  graphScore: (id: number) => request<GraphScore>(`/graph/score/${id}`),
  graphExplain: (id: number) => request<GraphExplain>(`/graph/explain/${id}`),
  neighbours: (id: number, hops: number) =>
    request<{ count: number; neighbors: Neighbour[] }>(`/graph/query/neighbors/${id}?hops=${hops}&limit=60`),
  illicitNearby: (id: number, hops: number) =>
    request<{ known_illicit_within_hops: { node_id: number; hops: number; score: number }[] }>(
      `/graph/query/illicit/${id}?hops=${hops}`,
    ),
  flaggedNeighbours: (id: number) =>
    request<{ neighbors: number; flagged_neighbors: number; known_illicit_neighbors: number }>(
      `/graph/query/flagged-neighbors/${id}`,
    ),
  cypher: (hops: number) => request<Record<string, string>>(`/graph/query/cypher?hops=${hops}`),
  path: (source: number, target: number) =>
    request<{ connected: boolean; path: null | { hops: number; path: { node_id: number; ground_truth: string; verdict: Verdict }[] } }>(
      `/graph/query/path?source=${source}&target=${target}`,
    ),
  /** `model` is sent only for an opt-in choice (F2). The default model D is never named, so the service picks it. */
  voiceScore: (wav: Blob, name: string, model?: string) =>
    request<VoiceScore>(`/voice/score${model && model !== "D" ? `?model=${encodeURIComponent(model)}` : ""}`, {
      method: "POST",
      body: audioForm(wav, name),
    }),
  caseScore: (nodeId: number, wav: Blob, name: string) =>
    request<CaseScore>("/case/score", { method: "POST", body: audioForm(wav, name, { node_id: String(nodeId) }) }),
};
