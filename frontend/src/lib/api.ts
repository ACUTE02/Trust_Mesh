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
    error?: string | null;
  };
  neo4j: { connected: boolean; nodes?: number; edges?: number };
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

export interface VoiceScore {
  clip: string;
  score: number;
  verdict: Verdict;
  model: string;
  model_requested: string;
  fallback_used: boolean;
  thresholds: { medium: number; high: number; source: string };
  duration_s: number;
  speech_s: number;
  lead_silence_s?: number;
  trail_silence_s?: number;
  scored_seconds?: number;
}

export interface CaseScore {
  case_id: number;
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

const UNREACHABLE =
  "The scoring service didn't answer. Start the stack with docker compose up and try again.";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`/api${path}`, init);
  } catch {
    throw new ApiError(UNREACHABLE);
  }
  if (response.status === 502 || response.status === 504) throw new ApiError(UNREACHABLE, response.status);
  let body: unknown = null;
  try {
    body = await response.json();
  } catch {
    if (!response.ok) throw new ApiError(UNREACHABLE, response.status);
  }
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
  path: (source: number, target: number) =>
    request<{ connected: boolean; path: null | { hops: number; path: { node_id: number; ground_truth: string; verdict: Verdict }[] } }>(
      `/graph/query/path?source=${source}&target=${target}`,
    ),
  voiceScore: (wav: Blob, name: string) => request<VoiceScore>("/voice/score", { method: "POST", body: audioForm(wav, name) }),
  caseScore: (nodeId: number, wav: Blob, name: string) =>
    request<CaseScore>("/case/score", { method: "POST", body: audioForm(wav, name, { node_id: String(nodeId) }) }),
};
