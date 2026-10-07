import { useState } from "react";
import { Link } from "react-router-dom";
import { CaseCard, DemoState, ModelInUse, errorText, useHealth } from "../components/caseParts";
import { ClipInput } from "../components/clip";
import { EvidenceRow, Marker, Num, Section } from "../components/evidence";
import { TwoScoresInstrument } from "../components/TwoScoresInstrument";
import examples from "../data/examples.json";
import { api, type CaseScore, type GraphScore } from "../lib/api";
import type { Clip } from "../lib/audio";

const UNIT_TICKS = [
  { at: 0, label: "0" },
  { at: 0.5, label: "0.5" },
  { at: 1, label: "1" },
];

export function SplitInstrument() {
  return (
    <TwoScoresInstrument
      title="One GraphSAGE model on the Elliptic Bitcoin transaction graph, scored two ways."
      hue="ledger"
      axisLabel="PR-AUC (higher is better)"
      axisMax={{ at: 1, label: "1" }}
      ticks={UNIT_TICKS}
      differenceId="graph.sage.splitGap"
      lines={[
        "Same model, same settings. Only the split changed.",
        "A random split trains and tests on transactions from the same period. A time split trains on earlier time steps and tests on later ones.",
      ]}
      first={{ label: "Random split", id: "graph.sage.random" }}
      second={{ label: "Time split", id: "graph.sage.temporal" }}
    />
  );
}

function StatusRegister() {
  const health = useHealth();
  const voice = health && health !== "down" ? health.voice : null;
  return (
    <table className="stack-sm w-full border-collapse text-sm">
      <caption className="sr-only">The three engines: what each detects, its status and its headline number</caption>
      <thead>
        <tr className="text-left text-xs text-ink-soft">
          <th scope="col" className="pb-2 pr-4 font-normal">
            Engine
          </th>
          <th scope="col" className="pb-2 pr-4 font-normal">
            Detects
          </th>
          <th scope="col" className="pb-2 pr-4 font-normal">
            Status
          </th>
          <th scope="col" className="pb-2 font-normal">
            The number that matters
          </th>
        </tr>
      </thead>
      <tbody>
        <tr className="rule-t align-top">
          <th scope="row" className="py-4 pr-4 text-left text-base font-semibold">
            <Link to="/graph">Graph Engine</Link>
          </th>
          <td className="py-4 pr-4" data-label="Detects">Transaction fraud</td>
          <td className="py-4 pr-4" data-label="Status">Measured</td>
          <td className="py-4">
            <span className="mr-2">
              <Marker ids={["graph.xgb12.mean"]} />
            </span>
            PR-AUC <Num id="graph.xgb12.mean" /> over three seeds, on a time split. XGBoost on the raw features plus degree,
            PageRank and clustering coefficient.
          </td>
        </tr>
        <tr className="rule-t align-top">
          <th scope="row" className="py-4 pr-4 text-left text-base font-semibold">
            <Link to="/voice">Voice Engine</Link>
          </th>
          <td className="py-4 pr-4" data-label="Detects">Spoofed and cloned speech</td>
          <td className="py-4 pr-4" data-label="Status">Measured</td>
          <td className="py-4">
            <span className="mr-2">
              <Marker ids={["voice.2c.D", "voice.2b.C"]} />
            </span>
            Equal error rate <Num id="voice.2c.D" /> over three seeds for the chosen model, D. Model C, statistically tied,
            scores <Num id="voice.2b.C" />. The split is this project's own, not the official ASVspoof protocol.
            <span className="mt-1 block text-ink-soft" data-testid="served-model">
              {health === null && "Asking the scoring service which model it is serving."}
              {health === "down" && "The scoring service is not running, so the served model is unknown right now."}
              {voice && voice.ready && (
                <>
                  Serving now: <ModelInUse model={voice.serving_default} requested={voice.requested_default} fallback={voice.fallback_active} weightsFile={voice.weights?.[voice.serving_default ?? ""]} />
                  {voice.serving_default === "D" && !voice.fallback_active && ", one seed of those three."}
                </>
              )}
              {voice && !voice.ready && "The voice model is not loaded in the scoring service."}
            </span>
          </td>
        </tr>
        <tr>
          <td colSpan={4} className="p-0">
            <div className="planned mt-1 grid gap-x-4 gap-y-1 px-3 py-4 text-ink-soft sm:grid-cols-[9rem_1fr]">
              <span className="text-base font-semibold">Visual Engine</span>
              <span>Fake images, then video. Designed, not built. No results yet.</span>
            </div>
          </td>
        </tr>
      </tbody>
    </table>
  );
}

export function CaseDesk() {
  const health = useHealth();
  const [nodeId, setNodeId] = useState<number>(examples.transactions[0].id);
  const [clip, setClip] = useState<Clip | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [graph, setGraph] = useState<GraphScore | null>(null);
  const [fused, setFused] = useState<CaseScore | null>(null);

  async function score() {
    setLoading(true);
    setError(null);
    setGraph(null);
    setFused(null);
    try {
      if (clip) {
        const result = await api.caseScore(nodeId, clip.wav, clip.name);
        setFused(result);
        setGraph(result.graph);
      } else {
        setGraph(await api.graphScore(nodeId));
      }
    } catch (e) {
      setError(errorText(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="grid gap-8 lg:grid-cols-[minmax(0,22rem)_minmax(0,1fr)]">
      <form
        className="grid content-start gap-5"
        onSubmit={(ev) => {
          ev.preventDefault();
          void score();
        }}
      >
        <label className="grid gap-1 text-sm font-semibold">
          Transaction
          <select className="field w-full min-w-0 font-normal" value={nodeId} onChange={(ev) => setNodeId(Number(ev.target.value))}>
            {examples.transactions.map((t) => (
              <option key={t.id} value={t.id}>
                {t.label}
              </option>
            ))}
          </select>
        </label>
        <div>
          <div className="mb-2 text-sm font-semibold">Voice clip</div>
          <ClipInput clip={clip} onClip={setClip} health={health} optional />
        </div>
        <div>
          <button type="submit" className="btn" disabled={loading}>
            Score this case
          </button>
        </div>
        {health === "down" && (
          <p className="m-0 text-sm text-ink-soft">
            This desk needs the scoring service. Start the stack with docker compose up; the rest of the page reads without it.
          </p>
        )}
      </form>
      <div>
        <DemoState loading={loading} error={error} waitText={clip ? "Scoring the transaction and the clip. The voice model takes a few seconds." : "Scoring the transaction."} />
        {!loading && !error && !graph && (
          <p className="m-0 text-sm text-ink-soft">Nothing scored yet. Choose a transaction, add a clip if you have one, and score the case.</p>
        )}
        {graph && <CaseCard graph={graph} voice={fused?.voice ?? null} combined={fused?.fusion ?? null} />}
        {graph && (
          <p className="measure mt-3 mb-0 text-xs text-ink-soft">
            The transaction comes from the Elliptic dataset and the clip is yours: they were never part of one fraud case. This
            desk shows the mechanism of combining two independent scores.
          </p>
        )}
      </div>
    </div>
  );
}

export function Landing() {
  return (
    <>
      <section className="grid items-start gap-x-12 gap-y-10 pt-14 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]" aria-labelledby="hero-h">
        <div>
          <h1 id="hero-h" style={{ fontSize: "var(--text-hero)" }}>
            Two fraud detectors, and the evidence for how far to trust each one.
          </h1>
          <p className="measure mt-6 mb-0">
            A transaction-graph model and a voice-spoof model, each with its failures on record. Start with the instrument:
            switch it to the time split.
          </p>
        </div>
        <SplitInstrument />
      </section>

      <Section title="Status of each engine" id="status">
        <StatusRegister />
      </Section>

      <Section title="Case desk" id="case-desk">
        <p className="measure mt-0 mb-6">
          One case, scored by both engines. The transaction is scored by the graph model; if you add a clip, the voice model
          scores it and the two are combined.
        </p>
        <CaseDesk />
      </Section>

      <Section title="What the evidence shows" id="evidence">
        <EvidenceRow
          ids={["graph.xgb12.mean", "graph.sage.best.mean", "graph.trials"]}
          claim={
            <>
              On Elliptic, XGBoost with cheap graph statistics scores <Num id="graph.xgb12.mean" />. The best graph neural
              network, after <Num id="graph.trials" /> tuning trials, scores <Num id="graph.sage.best.mean" />.
            </>
          }
          more={
            <>
              PR-AUC on the time split, three seeds each. <Link to="/graph">The full leaderboard is on the Graph Engine page.</Link>
            </>
          }
        />
        <EvidenceRow
          ids={["voice.p2.untrimmed", "voice.p2.trimmed"]}
          claim={
            <>
              The first voice model reached an equal error rate of{" "}
              <span className="fill-shortcut px-1.5 py-0.5">
                <Num id="voice.p2.untrimmed" />
              </span>{" "}
              by reading the silence at the start of clips. With the silence removed it scored <Num id="voice.p2.trimmed" />.
            </>
          }
          more={
            <>
              The hatched number is a shortcut, not a result. <Link to="/voice">The silence experiments are on the Voice Engine page.</Link>
            </>
          }
        />
        <EvidenceRow
          ids={["voice.gap.A", "voice.gap.C"]}
          claim={
            <>
              Facing four attack types it never trained on, a plain CNN's error rate worsened by <Num id="voice.gap.A" />. The
              frozen self-supervised model's changed by <Num id="voice.gap.C" />, which is not significant.
            </>
          }
          more={
            <>
              Change in equal error rate, unseen attacks minus the same attacks when seen. <Link to="/voice#generalisation">Intervals are on the Voice Engine page.</Link>
            </>
          }
        />
      </Section>

      <Section title="What was cut" id="cut">
        <ul className="measure m-0 grid list-none gap-3 p-0">
          <li className="rule-t pt-3">
            Neo4j is not in the science path. The graph fits in memory for training; Neo4j serves the demo's neighbourhood
            queries.
          </li>
          <li className="rule-t pt-3">No cloud deployment. The stack runs locally in Docker with one command.</li>
          <li className="rule-t pt-3">
            Voice models E and F are implemented but never completed a run on real data. They are untested, not disproven.
          </li>
          <li className="rule-t pt-3">The Visual Engine is paused before its first run. It has no results.</li>
          <li className="rule-t pt-3">
            The combined score is a mechanism. It has no measured accuracy, because no dataset holds the same case in both
            signals.
          </li>
        </ul>
      </Section>
    </>
  );
}
