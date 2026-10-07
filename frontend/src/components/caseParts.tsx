import { useEffect, useState, type ReactNode } from "react";
import { ApiError, api, type GraphScore, type Health, type Verdict, type VoiceScore } from "../lib/api";
import { CaveatNote, Live, Num } from "./evidence";

/** Shows whichever voice model the API reports. Never a hard-coded name: the fallback can occur. */
export function ModelInUse({
  model,
  requested,
  fallback,
  weightsFile,
}: {
  model: string | undefined;
  requested?: string;
  fallback?: boolean;
  weightsFile?: string | null;
}) {
  if (!model) return <span className="text-ink-soft">voice model unknown (the service did not report one)</span>;
  return (
    <span data-testid="model-in-use">
      Voice model {model}
      {weightsFile && (
        <>
          {" "}
          (<span className="font-mono text-[0.75em] break-all">{weightsFile}</span>)
        </>
      )}
      {fallback && (
        <strong className="font-semibold text-flag">
          {" "}
          Fallback in use: model {requested ?? "D"} was requested but its weights are not installed, so model {model} answered.
        </strong>
      )}
    </span>
  );
}

const VERDICT_WORDS: Record<Verdict, string> = { low: "low risk", medium: "medium risk", high: "high risk" };

/** Verdict as words plus a weight of rule, never colour alone. */
export function VerdictTag({ verdict }: { verdict: Verdict }) {
  const style =
    verdict === "high" ? "bg-ink text-slab" : verdict === "medium" ? "border-[1.5px] border-ink" : "border-[1.5px] border-dashed border-ink-soft text-ink-soft";
  return <span className={`inline-block px-2 py-0.5 text-xs font-semibold whitespace-nowrap ${style}`}>{VERDICT_WORDS[verdict]}</span>;
}

export function Score({ value }: { value: number }) {
  return <Live>{value.toFixed(4)}</Live>;
}

/** Loading / error / nothing-yet states shared by every demo. An error says what happened and what to do. */
export function DemoState({ loading, error, waitText }: { loading: boolean; error: string | null; waitText: string }) {
  if (loading) {
    return (
      <p role="status" className="m-0 text-sm text-ink-soft">
        {waitText}
      </p>
    );
  }
  if (error) {
    return (
      <p role="alert" className="measure m-0 border-l-4 border-flag bg-sheet px-3 py-2 text-sm">
        {error}
      </p>
    );
  }
  return null;
}

export function errorText(e: unknown): string {
  return e instanceof ApiError || e instanceof Error ? e.message : "Something went wrong. Try again.";
}

/**
 * /health for a page. null = still asking; "down" = the service did not answer. A failed request is
 * retried a few times, so a page opened while the stack is still starting corrects itself.
 */
export function useHealth(): Health | null | "down" {
  const [health, setHealth] = useState<Health | null | "down">(null);
  useEffect(() => {
    let live = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let attempts = 0;
    const ask = () => {
      attempts += 1;
      api
        .health()
        .then((h) => live && setHealth(h))
        .catch(() => {
          if (!live) return;
          setHealth("down");
          if (attempts < HEALTH_ATTEMPTS) timer = setTimeout(ask, HEALTH_RETRY_MS);
        });
    };
    ask();
    return () => {
      live = false;
      clearTimeout(timer);
    };
  }, []);
  return health;
}

const HEALTH_ATTEMPTS = 15; // about 75 seconds: longer than the service takes to load its models
const HEALTH_RETRY_MS = 5000;

function Row({ name, children, planned = false }: { name: string; children: ReactNode; planned?: boolean }) {
  return (
    <tr className={planned ? "" : "rule-t"}>
      <th scope="row" className={`w-28 py-3 pr-4 text-left align-top font-semibold ${planned ? "text-ink-soft" : ""}`}>
        {name}
      </th>
      {children}
    </tr>
  );
}

const EMPTY = <span className="text-ink-soft">empty</span>;

/**
 * One case, drawn like a row of the cases table. Empty means empty: a missing signal shows the word
 * "empty", never a zero. The combined score always carries its note.
 */
export function CaseCard({
  graph,
  voice,
  combined,
}: {
  graph: GraphScore | null;
  voice: VoiceScore | null;
  combined: { combined_score: number; combined_verdict: Verdict; weights: Record<string, number> } | null;
}) {
  return (
    <div data-testid="case-card">
      <table className="stack-sm w-full border-collapse text-sm">
        <caption className="sr-only">Scores for this case, one row per check</caption>
        <thead>
          <tr className="text-left text-xs text-ink-soft">
            <th scope="col" className="pb-2 font-normal">
              Check
            </th>
            <th scope="col" className="pb-2 font-normal">
              Score
            </th>
            <th scope="col" className="pb-2 font-normal">
              Verdict
            </th>
            <th scope="col" className="pb-2 font-normal">
              Produced by
            </th>
          </tr>
        </thead>
        <tbody>
          <Row name="Transaction">
            <td className="py-3 pr-4 align-top text-lg" data-testid="graph-score" data-label="Score">
              {graph ? <Score value={graph.score} /> : EMPTY}
            </td>
            <td className="py-3 pr-4 align-top" data-label="Verdict">{graph ? <VerdictTag verdict={graph.verdict} /> : EMPTY}</td>
            <td className="py-3 align-top">
              {graph ? (
                <>
                  The transaction model.{" "}
                  <span className="text-ink-soft">
                    Known label: {graph.ground_truth}; {graph.split} period{graph.in_sample ? ", seen in training" : ""}.
                  </span>
                </>
              ) : (
                EMPTY
              )}
            </td>
          </Row>
          <Row name="Voice">
            <td className="py-3 pr-4 align-top text-lg" data-testid="voice-score" data-label="Score">
              {voice ? <Score value={voice.score} /> : EMPTY}
            </td>
            <td className="py-3 pr-4 align-top" data-label="Verdict">{voice ? <VerdictTag verdict={voice.verdict} /> : EMPTY}</td>
            <td className="py-3 align-top">
              {voice ? (
                <ModelInUse model={voice.model} requested={voice.model_requested} fallback={voice.fallback_used} />
              ) : (
                <span className="text-ink-soft">No clip was added to this case.</span>
              )}
            </td>
          </Row>
          <tr>
            <td colSpan={4} className="p-0">
              <div className="planned my-1 flex flex-wrap items-baseline gap-x-4 px-3 py-2 text-ink-soft">
                <span className="w-24 font-semibold">Visual</span>
                <span>not connected. Designed, not built. No results yet.</span>
              </div>
            </td>
          </tr>
          <Row name="Combined">
            <td className="py-3 pr-4 align-top text-lg" data-testid="combined-score" data-label="Score">
              {combined ? <Score value={combined.combined_score} /> : EMPTY}
            </td>
            <td className="py-3 pr-4 align-top" data-label="Verdict">{combined ? <VerdictTag verdict={combined.combined_verdict} /> : EMPTY}</td>
            <td className="py-3 align-top">
              {combined ? "The average of the two scores above." : <span className="text-ink-soft">Needs both a transaction score and a voice score.</span>}
            </td>
          </Row>
        </tbody>
      </table>
      <div className="mt-3">
        <CombinedNote />
      </div>
    </div>
  );
}

/** The fixed note that sits beside every combined score. */
export function CombinedNote() {
  return (
    <CaveatNote>
      The combined score is a demonstration of the idea. It has no measured accuracy, because no dataset contains the same
      case in both checks. The two scores are simply averaged (<Num id="fusion.weight" /> and <Num id="fusion.weight" />).
    </CaveatNote>
  );
}
