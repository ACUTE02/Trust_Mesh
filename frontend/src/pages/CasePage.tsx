import { useState } from "react";
import { VerdictTag } from "../components/caseParts";
import { EngineerLayer, EngineerToggle, ForEngineers } from "../components/engineers";
import { Live, Marker, Num } from "../components/evidence";
import { KineticText } from "../components/KineticText";
import type { CaseScore, GraphScore } from "../lib/api";
import { Chapter, useResetStage } from "../world/chapters";
import { CaseDesk } from "./Landing";

/** One scored case in this visit. The server keeps its own table; this register lives in the page and is gone when you leave. */
interface Entry {
  n: number;
  graph: GraphScore;
  fused: CaseScore | null;
  masked: boolean;
}

const RANK = { low: 0, medium: 1, high: 2 } as const;

function risk(e: Entry): number {
  return e.fused ? e.fused.fusion.combined_score : e.graph.score;
}

function Register({ entries }: { entries: Entry[] }) {
  if (!entries.length) {
    return <p className="m-0 text-sm text-ink-soft">No case scored yet in this visit. Score one above and it is filed here, highest risk first.</p>;
  }
  const sorted = [...entries].sort((a, b) => RANK[(b.fused?.fusion.combined_verdict ?? b.graph.verdict)] - RANK[(a.fused?.fusion.combined_verdict ?? a.graph.verdict)] || risk(b) - risk(a));
  return (
    <ol className="m-0 grid list-none gap-3 p-0" data-testid="case-register">
      {sorted.map((e) => (
        <li key={e.n} className="rule-t grid gap-1 pt-3 sm:grid-cols-[6rem_minmax(0,1fr)]" data-cursor="Open the drawer">
          <span className="marker self-start">CASE {e.n}</span>
          <span>
            {e.masked ? (
              <span className="text-ink-soft">Details hidden in recording mode.</span>
            ) : (
              <>
                <VerdictTag verdict={e.fused?.fusion.combined_verdict ?? e.graph.verdict} />{" "}
                <span className="text-sm text-ink-soft">
                  {e.fused ? "combined score " : "transaction score "}
                  <Live>{risk(e).toFixed(4)}</Live>
                  {e.fused ? ", with a voice recording" : ", no voice recording"}
                </span>
              </>
            )}
          </span>
        </li>
      ))}
    </ol>
  );
}

export default function CasePage() {
  useResetStage();
  const [entries, setEntries] = useState<Entry[]>([]);
  return (
    <EngineerLayer page="case">
      <Chapter stage={0} className="hero" label="case-h">
        <div className="scrim chapter-card" style={{ maxWidth: "46rem" }}>
          <h1 id="case-h" style={{ fontSize: "var(--text-hero)" }}>
            <KineticText text="Two scores, one decision, honestly weighed." mode="wipe" />
          </h1>
          <p className="measure mt-5 mb-0 text-lg">A transaction check and a voice check each give a score. This desk averages them.</p>
          <p className="measure mt-2 mb-0">The average is a demonstration. It has no measured accuracy.</p>
          <div className="mt-6">
            <EngineerToggle />
          </div>
        </div>
        <div id="desk" className="panel mt-6" style={{ maxWidth: "52rem" }}>
          <h2 className="mb-3 text-xl">Score a case</h2>
          <p className="measure mt-0 mb-6">
            Pick a transaction. Add a voice recording if you have one. The balance behind this panel tilts toward the heavier score.
          </p>
          <CaseDesk onScored={(graph, fused, masked) => setEntries((prev) => [...prev, { n: prev.length + 1, graph, fused, masked }])} />
        </div>
      </Chapter>

      <Chapter stage={1} className="chapter" label="weights-h">
        <div className="scrim chapter-card" style={{ maxWidth: "44rem" }}>
          <p className="m-0 mb-3">
            <Marker ids={["fusion.weight"]} />
          </p>
          <h2 id="weights-h" className="text-xl">
            <KineticText text="The weights are placeholders" mode="wipe" />
          </h2>
          <p className="measure mt-4 mb-3">
            Each score counts for half: <Num id="fusion.weight" /> each. Nobody chose that number by testing. The bolt that holds
            the beam is shown loose on purpose: nothing fixes it in place.
          </p>
          <p className="measure mt-0 mb-3">
            A worked example, as arithmetic and nothing more. If the transaction score were as high as it can go and the voice
            score as low as it can go, the average would land exactly halfway, which falls in the medium band (the band edges are placeholders too). One check says alarm, the other says
            calm, and the average splits the difference.
          </p>
          <p className="measure m-0">
            It cannot be checked: no dataset links a transaction to a voice recording, so there has never been a real case with
            both. The combined score has no measured accuracy.
          </p>
          <ForEngineers title="the formula">
            <p className="m-0">
              combined = w_graph x graph score + w_voice x voice score, with both weights <Num id="fusion.weight" /> (set in
              serving/config.py, changeable with environment variables). Nothing here is calibrated on joint data.
            </p>
          </ForEngineers>
        </div>
      </Chapter>

      <Chapter stage={2} id="register" className="chapter" label="register-h">
        <div className="panel" style={{ maxWidth: "46rem" }}>
          <h2 id="register-h" className="mb-3 text-xl">
            <KineticText text="The cases register" mode="wipe" />
          </h2>
          <p className="measure mt-0 mb-5 text-sm text-ink-soft">
            Cases you scored in this visit, highest risk first. The service keeps its own table of scored cases; this site does
            not list it publicly.
          </p>
          <Register entries={entries} />
        </div>
      </Chapter>

      <Chapter stage={3} id="missing" className="chapter" label="missing-h">
        <div className="scrim chapter-card" style={{ maxWidth: "44rem" }}>
          <h2 id="missing-h" className="text-xl">
            <KineticText text="What would make this real" mode="wipe" />
          </h2>
          <ul className="m-0 mt-5 grid list-none gap-3 p-0">
            {[
              ["Joint data.", "Cases where a transaction and a voice recording belong to the same real event, with a known outcome."],
              ["Fitted, tested weights.", "Weights chosen on that data and checked on cases kept back."],
              ["A measured accuracy.", "How often the combined warning is right, with a margin of error."],
              ["The Visual Engine.", "A check for fake pictures and video. Designed, not built. No results yet."],
            ].map(([head, body]) => (
              <li key={head} className="planned px-3 py-3 text-sm">
                <strong className="font-semibold">{head}</strong> {body}
              </li>
            ))}
          </ul>
        </div>
      </Chapter>
    </EngineerLayer>
  );
}
