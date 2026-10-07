import { useState } from "react";
import { Link } from "react-router-dom";
import { CaseCard, DemoState, ModelInUse, RecordingBanner, errorText, isRecording, useHealth } from "../components/caseParts";
import { ClipInput } from "../components/clip";
import { EngineerLayer, EngineerToggle, ForEngineers, GRAPH_SCORE_MEANING, Meaning, VOICE_SCORE_MEANING } from "../components/engineers";
import { Marker, Num, Section } from "../components/evidence";
import { TechView } from "../components/TechView";
import { TwoScoresInstrument } from "../components/TwoScoresInstrument";
import { evidence } from "../data/evidence";
import examples from "../data/examples.json";
import { api, readCallLog, startCallLog, type ApiCall, type CaseScore, type GraphScore } from "../lib/api";
import type { Clip } from "../lib/audio";

const UNIT_TICKS = [
  { at: 0, label: "0" },
  { at: 0.5, label: "0.5" },
  { at: 1, label: "1" },
];

export function SplitInstrument() {
  return (
    <TwoScoresInstrument
      title="One fraud-finding model, tested two ways."
      hue="ledger"
      axisLabel="Score from 0 to 1. Higher is better."
      axisMax={{ at: 1, label: "1" }}
      ticks={UNIT_TICKS}
      differenceId="graph.sage.splitGap"
      meaning={GRAPH_SCORE_MEANING}
      lines={[
        "Same model, same settings. Only the split changed.",
        "Random split: the test uses transactions from the same period the model learned from.",
        "Time split: the model learns from earlier transactions and is tested on later ones.",
      ]}
      first={{ label: "Random split", id: "graph.sage.random" }}
      second={{ label: "Time split", id: "graph.sage.temporal" }}
    />
  );
}

/** Two engine cards, deliberately not twins: the graph card compares two bars, the voice card states one rate. */
function EngineCards() {
  const health = useHealth();
  const voice = health && health !== "down" ? health.voice : null;
  return (
    <div className="grid gap-10 lg:grid-cols-[minmax(0,7fr)_minmax(0,5fr)]">
      <article className="rule-t pt-5">
        <h3 className="text-lg">Transaction check</h3>
        <p className="measure mt-2 mb-4">This model looks at who a transaction is connected to.</p>
        <table className="w-full border-collapse text-sm">
          <caption className="sr-only">Two ways of using connections, and their scores</caption>
          <tbody>
            {(
              [
                ["A simple model that counts connections", "graph.xgb12.plain", "our best"],
                ["The best of the fancier graph models", "graph.sage.best.plain", ""],
              ] as const
            ).map(([label, id, note]) => (
              <tr key={id} className="rule-t align-middle">
                <th scope="row" className="w-[46%] py-2.5 pr-3 text-left font-normal">
                  {label}
                  {note && <span className="block text-xs text-ink-soft">{note}</span>}
                </th>
                <td className="py-2.5 pr-3" aria-hidden>
                  <div className="h-4 bg-track">
                    <div className="fill-ledger h-4" style={{ width: `${(evidence(id).value ?? 0) * 100}%` }} />
                  </div>
                </td>
                <td className="w-[1%] py-2.5 text-right text-lg font-semibold">
                  <Num id={id} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="mt-3">
          <Meaning>{GRAPH_SCORE_MEANING}</Meaning>
        </div>
        <p className="mt-4 mb-0 text-sm">
          <span className="mr-2">
            <Marker ids={["graph.xgb12.plain", "graph.xgb12.mean", "graph.sage.best.plain", "graph.sage.best.mean"]} />
          </span>
          <Link to="/graph">See every model we tried on the transaction page.</Link>
        </p>
      </article>

      <article className="rule-t pt-5">
        <h3 className="text-lg">Voice check</h3>
        <p className="measure mt-2 mb-4">This model listens for signs that a voice is computer-made.</p>
        <p className="m-0 leading-tight">
          <span className="font-semibold" style={{ fontSize: "var(--text-hero)" }}>
            <Num id="voice.2c.D.per100" />
          </span>{" "}
          <span className="text-lg">in 100 recordings wrong</span>
        </p>
        <p className="measure mt-2 mb-3 text-sm">
          That is an error rate of <Num id="voice.2c.D" spread={false} />. {VOICE_SCORE_MEANING}
        </p>
        <Meaning>We split the data our own way. So this cannot be compared with published results.</Meaning>
        <p className="mt-3 mb-0 text-sm text-ink-soft" data-testid="served-model">
          {health === null && "Asking the scoring service which model it is running."}
          {health === "down" && "The scoring service is not running, so the model in use is unknown right now."}
          {voice && voice.ready && (
            <>
              Running now: <ModelInUse model={voice.serving_default} requested={voice.requested_default} fallback={voice.fallback_active} />
            </>
          )}
          {voice && !voice.ready && "The voice model is not loaded in the scoring service."}
        </p>
        <p className="mt-4 mb-0 text-sm">
          <span className="mr-2">
            <Marker ids={["voice.2c.D.per100", "voice.2c.D"]} />
          </span>
          <Link to="/voice">See how the voice model was caught cheating, and fixed.</Link>
        </p>
      </article>
    </div>
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
  const [calls, setCalls] = useState<ApiCall[]>([]);
  const mask = isRecording(health);

  async function score() {
    setLoading(true);
    setError(null);
    setGraph(null);
    setFused(null);
    startCallLog();
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
      setCalls(readCallLog());
      setLoading(false);
    }
  }

  return (
    <div>
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
              {examples.transactions.map((t, i) => (
                <option key={t.id} value={t.id}>
                  {mask ? `Example transaction ${i + 1}` : t.label}
                </option>
              ))}
            </select>
          </label>
          <div>
            <div className="mb-2 text-sm font-semibold">Voice recording</div>
            <ClipInput clip={clip} onClip={setClip} health={health} optional />
          </div>
          <div>
            <button type="submit" className="btn" disabled={loading}>
              Score this case
            </button>
          </div>
          {health === "down" && (
            <p className="m-0 text-sm text-ink-soft">
              This desk needs the scoring service. Start the stack with docker compose up. The rest of the page reads without it.
            </p>
          )}
        </form>
        <div>
          <DemoState loading={loading} error={error} waitText={clip ? "Scoring the transaction and the recording. The voice check takes a few seconds." : "Scoring the transaction."} />
          {!loading && !error && !graph && (
            <p className="m-0 text-sm text-ink-soft">Nothing scored yet. Pick a transaction, add a recording if you have one, and score the case.</p>
          )}
          {mask && <RecordingBanner />}
          {graph && <CaseCard graph={graph} voice={fused?.voice ?? null} combined={fused?.fusion ?? null} mask={mask} />}
          {graph && (
            <p className="measure mt-3 mb-0 text-xs text-ink-soft">
              The transaction is from a public dataset. The recording is yours. They were never part of one real case.
            </p>
          )}
        </div>
      </div>
      {!mask && <TechView calls={calls} health={health} />}
    </div>
  );
}

/** Hand-built system diagram. The caption under it says the same thing in words. */
function SystemDiagram() {
  const box = (x: number, y: number, w: number, h: number, title: string, sub?: string, dashed = false) => (
    <g key={title}>
      <rect x={x} y={y} width={w} height={h} fill="var(--sheet)" stroke={dashed ? "var(--ink-soft)" : "var(--ink)"} strokeWidth="1.5" strokeDasharray={dashed ? "5 4" : undefined} />
      <text x={x + w / 2} y={y + (sub ? h / 2 - 3 : h / 2 + 4)} textAnchor="middle" fontSize="12.5" fontWeight="600" fill="var(--ink)">
        {title}
      </text>
      {sub && (
        <text x={x + w / 2} y={y + h / 2 + 13} textAnchor="middle" fontSize="10.5" fill="var(--ink-soft)">
          {sub}
        </text>
      )}
    </g>
  );
  const line = (x1: number, y1: number, x2: number, y2: number, dashed = false) => (
    <line key={`${x1}-${y1}-${x2}-${y2}`} x1={x1} y1={y1} x2={x2} y2={y2} stroke={dashed ? "var(--ink-soft)" : "var(--ink)"} strokeWidth="1.5" strokeDasharray={dashed ? "5 4" : undefined} />
  );
  return (
    <figure className="m-0">
      <div className="overflow-x-auto">
        <svg viewBox="0 0 780 372" className="w-full min-w-[40rem]" role="img" aria-label="System diagram. The caption below describes the same connections.">
          {line(120, 40, 170, 40)}
          {line(300, 40, 350, 40)}
          {line(415, 62, 415, 96)}
          {line(85, 96, 705, 96)}
          {[85, 240, 395, 550, 705].map((x) => line(x, 96, x, 122))}
          {line(395, 174, 395, 262)}
          {line(85, 174, 85, 232)}
          {line(85, 232, 395, 232)}
          {line(705, 174, 705, 290, true)}
          {line(705, 290, 470, 290, true)}
          {line(395, 314, 395, 326)}
          {box(10, 18, 110, 44, "Browser", "React site")}
          {box(170, 18, 130, 44, "nginx", "static files, /api proxy")}
          {box(350, 18, 130, 44, "FastAPI", "one service")}
          {box(20, 122, 130, 52, "XGBoost Tier-1+2", "the graph score")}
          {box(175, 122, 130, 52, "GAT attention", "explanation only")}
          {box(330, 122, 130, 52, "XLS-R + head", "the voice score")}
          {box(485, 122, 130, 52, "Neo4j", "neighbourhood queries")}
          {box(640, 122, 130, 52, "Visual slot", "not connected", true)}
          {box(320, 262, 150, 52, "Fusion", "weighted average")}
          {box(320, 326, 150, 40, "SQLite: cases")}
          <text x="500" y="34" fontSize="10.5" fill="var(--ink-soft)">
            /graph/score, /graph/explain,
          </text>
          <text x="500" y="49" fontSize="10.5" fill="var(--ink-soft)">
            /voice/score, /case/score
          </text>
        </svg>
      </div>
      <figcaption className="mt-2 text-xs text-ink-soft">
        The browser talks only to nginx, which serves the site and forwards /api to FastAPI. FastAPI loads XGBoost (the graph
        score), a GAT (explanation only) and the frozen XLS-R encoder with its head (the voice score), and queries Neo4j for
        neighbourhoods. The graph and voice scores go into the fusion step, which writes to the SQLite cases table. The
        visual slot is empty.
      </figcaption>
    </figure>
  );
}

function EngineersLanding() {
  return (
    <>
      <ForEngineers title="system">
        <SystemDiagram />
      </ForEngineers>
      <ForEngineers title="footprint">
        <table className="w-full max-w-xl border-collapse">
          <tbody>
            {(
              [
                ["API process, both models loaded", <Num key="a" id="infra.apiMemoryGb" />],
                ["Neo4j, peak", <Num key="b" id="infra.neo4jMemoryGb" />],
                ["Graph in Neo4j: nodes", <Num key="c" id="graph.nodes" />],
                ["Graph in Neo4j: edges", <Num key="d" id="graph.edges" />],
              ] as const
            ).map(([k, v]) => (
              <tr key={k} className="rule-t">
                <th scope="row" className="py-1.5 pr-4 text-left font-normal">
                  {k}
                </th>
                <td className="py-1.5 text-right">{v}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="mt-2 mb-0 text-xs text-ink-soft">Measured on one Windows laptop with Docker Desktop. CPU only; no GPU is used for serving.</p>
      </ForEngineers>
      <ForEngineers title="verification">
        <ul className="m-0 grid list-none gap-1.5 p-0">
          <li>
            <Num id="infra.pythonTests" /> Python unit tests and <Num id="infra.frontendTests" /> frontend tests pass.
          </li>
          <li>
            <code className="font-mono text-[0.85em]">scripts/verify_serving.py</code>: <Num id="infra.verifyChecks" /> checks pass
            against the running Docker stack.
          </li>
          <li>The build fails if a page renders a number that is not in the data module or from the live API.</li>
          <li>On every start the service checks that the loaded graph weights reproduce the reported test score.</li>
          <li className="text-ink-soft">
            Not tested: a start from an empty machine, any public deployment, real speech through the microphone button, and
            screen readers.
          </li>
        </ul>
      </ForEngineers>
      <ForEngineers title="decisions and trade-offs">
        <table className="stack-sm w-full border-collapse">
          <thead>
            <tr className="text-left text-xs text-ink-soft">
              <th scope="col" className="pb-2 pr-4 font-normal">Decision</th>
              <th scope="col" className="pb-2 pr-4 font-normal">Alternative</th>
              <th scope="col" className="pb-2 font-normal">Evidence</th>
            </tr>
          </thead>
          <tbody>
            {(
              [
                ["Score with XGBoost on raw features plus degree, PageRank and clustering", "Serve the best GNN (GraphSAGE)", <>PR-AUC <Num id="graph.xgb12.mean" /> against <Num id="graph.sage.best.mean" />, three seeds each</>],
                ["Evaluate on a temporal split", "A random split", <>The random split overstates the same model: <Num id="graph.sage.random" /> against <Num id="graph.sage.temporal" /></>],
                ["Keep a GAT for explanation only", "Explain with the scorer, or not at all", <>GAT scores PR-AUC <Num id="graph.gat.first" />, so it never produces the score</>],
                ["Trim leading and trailing silence before training and scoring", "Score the raw clip", <>Untrimmed EER <Num id="voice.p2.untrimmed" /> was a shortcut; trimmed, the same model scored <Num id="voice.p2.trimmed" /></>],
                ["Serve Model D (mean-only pooling)", "Model C (mean and std pooling)", <>Tied EER (<Num id="voice.2c.D" /> against <Num id="voice.2b.C" />) at <Num id="voice.params.D" /> head parameters against <Num id="voice.params.C" /></>],
                ["Fuse with a fixed weighted average", "Fit fusion weights", "No dataset links a transaction to a voice clip, so there is nothing to fit or validate against"],
                ["Neo4j for serving queries only", "Neo4j in the training path", "The graph fits in memory for training; the database earns its place on neighbourhood and path queries"],
                ["nginx proxies /api on one origin", "Publish the API on its own port with open CORS", "Only one service is exposed in a deployment; Neo4j and the API publish no port"],
              ] as const
            ).map(([d, a, e]) => (
              <tr key={d} className="rule-t align-top">
                <th scope="row" className="py-2 pr-4 text-left font-semibold">{d}</th>
                <td className="py-2 pr-4" data-label="Alternative">{a}</td>
                <td className="py-2" data-label="Evidence">{e}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </ForEngineers>
    </>
  );
}

export function Landing() {
  return (
    <EngineerLayer page="landing">
      <section className="grid items-start gap-x-12 gap-y-10 pt-14 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]" aria-labelledby="hero-h">
        <div>
          <h1 id="hero-h" style={{ fontSize: "var(--text-hero)" }}>
            Same model. Tested two ways. The honest way scores much lower.
          </h1>
          <p className="measure mt-6 mb-0 text-lg">TrustMesh is a student project about catching fraud.</p>
          <p className="measure mt-2 mb-0">It shows what each check found, and how far to trust it.</p>
        </div>
        <SplitInstrument />
      </section>

      <Section title="What TrustMesh does" id="what">
        <ul className="measure m-0 grid list-none gap-3 p-0 text-lg">
          <li>It checks a transaction by looking at who it is connected to.</li>
          <li>It checks a voice recording for signs that it is fake.</li>
          <li>It combines the two into one warning. That part is a demonstration only.</li>
        </ul>
      </Section>

      <Section title="What each check scored" id="engines">
        <EngineCards />
      </Section>

      <Section title="Try it: one case, two checks" id="case-desk">
        <p className="measure mt-0 mb-6">Pick a transaction. Add a voice recording if you have one. Then score the case.</p>
        <CaseDesk />
      </Section>

      <Section title="What is tested, and what is not" id="status">
        <ul className="measure m-0 grid list-none gap-0 p-0">
          <li className="rule-t py-4">
            <strong className="font-semibold">Measured.</strong> The transaction check and the voice check. Each was scored on
            data kept back from training. We tested on one dataset each.
          </li>
          <li className="rule-t py-4">
            <strong className="font-semibold">Demonstration only.</strong> The combined warning. It has not been tested for
            accuracy.
          </li>
          <li className="py-1">
            <div className="planned px-3 py-4 text-ink-soft">
              <strong className="font-semibold">Visual Engine.</strong> A check for fake pictures and video. Designed, not built.
              No results yet.
            </div>
          </li>
        </ul>
      </Section>

      <Section title="For engineers" id="engineers">
        <p className="measure mt-0 mb-4 text-sm text-ink-soft">The system, its size, how it was checked, and the main design decisions.</p>
        <EngineerToggle />
        <EngineersLanding />
      </Section>
    </EngineerLayer>
  );
}
