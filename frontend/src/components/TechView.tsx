import { useId, useState } from "react";
import type { ApiCall, Health } from "../lib/api";

/**
 * Technical view for a case desk, off by default. It shows only what the service returned for what the
 * reader typed: each request, its round-trip time measured in this browser, and the raw response; the
 * model the service reports; and, on the graph desk, the Cypher the service says it ran.
 */
export function TechView({
  calls,
  health,
  cypher,
}: {
  calls: ApiCall[];
  health: Health | null | "down";
  cypher?: Record<string, string> | null;
}) {
  const [open, setOpen] = useState(false);
  const id = useId();
  const voice = health && health !== "down" ? health.voice : null;
  const graph = health && health !== "down" ? health.graph : null;
  const weights = voice?.weights?.[voice.serving_default ?? ""];
  const seed = weights?.match(/_s(\d+)\.pt$/)?.[1];
  return (
    <div className="mt-6">
      <button type="button" className="cursor-pointer border-0 bg-transparent p-0 text-sm underline" aria-expanded={open} aria-controls={id} onClick={() => setOpen(!open)}>
        {open ? "Hide the technical view" : "Show the technical view"}
      </button>
      {open && (
        <div id={id} data-live data-testid="tech-view" className="mt-3 grid gap-4 bg-sheet p-4 text-xs">
          <div>
            <div className="mb-1 font-semibold">Models the service reports (/health)</div>
            {health === null && <div>Asking the service.</div>}
            {health === "down" && <div>The service did not answer.</div>}
            {graph && <div className="font-mono">graph: {graph.model}</div>}
            {voice && (
              <div className="font-mono break-all">
                voice: model {voice.serving_default ?? "unknown"}, weights {weights ?? "unknown"}
                {seed ? `, seed ${seed}` : ""}, fallback_active {String(voice.fallback_active)}
              </div>
            )}
          </div>
          {calls.length === 0 && <div className="text-ink-soft">No request yet. Score something and the calls appear here.</div>}
          {calls.map((c, i) => (
            <div key={i}>
              <div className="font-mono font-semibold break-all">
                {c.method} {c.path}
              </div>
              <div className="text-ink-soft">
                status {c.status ?? "no answer"}, round trip {c.ms.toFixed(0)} ms (measured in this browser)
              </div>
              <pre className="mt-1 max-h-64 overflow-auto bg-slab p-2 font-mono text-[0.6875rem] leading-snug whitespace-pre-wrap">
                {JSON.stringify(c.body, null, 1)}
              </pre>
            </div>
          ))}
          {cypher && (
            <div>
              <div className="mb-1 font-semibold">Cypher the service runs for the neighbourhood (/graph/query/cypher)</div>
              {Object.entries(cypher).map(([name, text]) => (
                <div key={name} className="mb-2">
                  <div className="text-ink-soft">{name}</div>
                  <pre className="overflow-auto bg-slab p-2 font-mono text-[0.6875rem] leading-snug">{text}</pre>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
