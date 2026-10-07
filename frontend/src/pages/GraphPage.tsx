import { useState } from "react";
import { Link } from "react-router-dom";
import { DemoState, Score, VerdictTag, errorText } from "../components/caseParts";
import { DotPlot, type PlotRow } from "../components/DotPlot";
import { CaveatNote, EvidenceRow, Live, Num, Section } from "../components/evidence";
import examples from "../data/examples.json";
import { api, type GraphExplain, type GraphScore, type Neighbour, type Verdict } from "../lib/api";

const LEADERBOARD: PlotRow[] = [
  { id: "graph.xgb12", seedMeanId: "graph.xgb12.mean", label: "XGBoost, raw features plus degree, PageRank and clustering", reference: true },
  { id: "graph.xgb1", label: "XGBoost, raw features only" },
  {
    id: "graph.sage.best",
    seedMeanId: "graph.sage.best.mean",
    label: "GraphSAGE, 100-trial search (best graph neural network)",
    annotation: (
      <>
        about <Num id="graph.gnnGap" /> below XGBoost
      </>
    ),
  },
  { id: "graph.sage.wide", label: "GraphSAGE, 50-trial search" },
  { id: "graph.sage.temporal", label: "GraphSAGE, untuned" },
  { id: "graph.care", label: "CARE-GNN, simplified" },
  { id: "graph.rgcn.direction", label: "RGCN, edge types from payment direction" },
  { id: "graph.rgcn.single", label: "RGCN, untyped control" },
  { id: "graph.rgcn.hub", label: "RGCN, direction by hub" },
  { id: "graph.gat.tuned", label: "GAT, tuned" },
  { id: "graph.gin", label: "GIN" },
  { id: "graph.gat.first", label: "GAT, first run" },
];

const CAMOUFLAGE: PlotRow[] = [
  { id: "graph.camo.xgb", label: "XGBoost", reference: true },
  { id: "graph.camo.sage", label: "GraphSAGE" },
  { id: "graph.camo.care", label: "CARE-GNN (the targeted defence)" },
];

interface DeskResult {
  score: GraphScore;
  explain: GraphExplain | null;
  neighbours: Neighbour[] | null;
  illicit: { node_id: number; hops: number; score: number }[] | null;
  counts: { neighbors: number; flagged_neighbors: number; known_illicit_neighbors: number } | null;
  graphError: string | null;
}

/** Neighbourhood as a small ring drawing plus the table that says the same thing. */
function NeighbourhoodView({ centre, neighbours }: { centre: GraphScore; neighbours: Neighbour[] }) {
  const rings = [1, 2].map((h) => neighbours.filter((n) => n.hops === h).slice(0, h === 1 ? 14 : 22));
  const size = 300;
  const mid = size / 2;
  const dot = (verdict: Verdict, x: number, y: number, r: number, key: string) =>
    verdict === "high" ? (
      <circle key={key} cx={x} cy={y} r={r} fill="var(--ink)" />
    ) : verdict === "medium" ? (
      <circle key={key} cx={x} cy={y} r={r - 1} fill="var(--sheet)" stroke="var(--ink)" strokeWidth="2" />
    ) : (
      <circle key={key} cx={x} cy={y} r={r - 1} fill="var(--sheet)" stroke="var(--ink-soft)" strokeWidth="1.5" strokeDasharray="2 2" />
    );
  return (
    <div className="grid gap-6 md:grid-cols-[minmax(0,18rem)_minmax(0,1fr)]">
      <figure className="m-0">
        <svg viewBox={`0 0 ${size} ${size}`} className="w-full max-w-[18rem]" role="img" aria-label={`Transaction ${centre.node_id} at the centre, with ${rings[0].length} neighbours one hop away and ${rings[1].length} two hops away. The table lists them.`}>
          {[55, 115].map((r) => (
            <circle key={r} cx={mid} cy={mid} r={r} fill="none" stroke="var(--rule)" strokeWidth="1" />
          ))}
          {rings.map((ring, ri) =>
            ring.map((n, i) => {
              const angle = (2 * Math.PI * i) / ring.length - Math.PI / 2 + ri * 0.3;
              const r = ri === 0 ? 55 : 115;
              return dot(n.verdict, mid + r * Math.cos(angle), mid + r * Math.sin(angle), 7, `${ri}-${n.node_id}`);
            }),
          )}
          {dot(centre.verdict, mid, mid, 11, "centre")}
        </svg>
        <figcaption className="text-xs text-ink-soft">
          Inner ring: one hop away. Outer ring: two hops. Filled: high risk; outlined: medium; dashed: low. Up to 36 neighbours
          are drawn; the table is the full answer the database returned.
        </figcaption>
      </figure>
      <div className="max-h-80 overflow-auto">
        <table className="w-full border-collapse text-sm">
          <caption className="sr-only">Transactions within two hops, nearest first</caption>
          <thead>
            <tr className="text-left text-xs text-ink-soft">
              <th scope="col" className="pb-2 pr-3 font-normal">Transaction</th>
              <th scope="col" className="pb-2 pr-3 font-normal">Hops</th>
              <th scope="col" className="pb-2 pr-3 font-normal">Known label</th>
              <th scope="col" className="pb-2 pr-3 font-normal">XGBoost score</th>
              <th scope="col" className="pb-2 font-normal">Verdict</th>
            </tr>
          </thead>
          <tbody>
            {neighbours.map((n) => (
              <tr key={n.node_id} className="rule-t">
                <td className="py-1.5 pr-3"><Live>{n.node_id}</Live></td>
                <td className="py-1.5 pr-3"><Live>{n.hops}</Live></td>
                <td className="py-1.5 pr-3">{n.ground_truth}</td>
                <td className="py-1.5 pr-3"><Score value={n.score} /></td>
                <td className="py-1.5"><VerdictTag verdict={n.verdict} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function GraphDesk() {
  const [input, setInput] = useState(String(examples.transactions[0].id));
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<DeskResult | null>(null);
  const [target, setTarget] = useState("");
  const [path, setPath] = useState<string | null>(null);

  async function score(raw: string) {
    const id = Number(raw);
    setPath(null);
    if (!Number.isInteger(id) || id < 0 || id > examples.maxId) {
      setError(`Enter a whole number from 0 to ${examples.maxId}. That is the range of transactions in the graph.`);
      setResult(null);
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const s = await api.graphScore(id);
      // The score does not need the graph database; the neighbourhood does. Report them separately.
      const [explain, graph] = await Promise.all([
        api.graphExplain(id).catch(() => null),
        Promise.all([api.neighbours(id, 2), api.illicitNearby(id, 3), api.flaggedNeighbours(id)])
          .then(([n, i, c]) => ({ n: n.neighbors, i: i.known_illicit_within_hops, c, err: null as string | null }))
          .catch((e) => ({ n: null, i: null, c: null, err: errorText(e) })),
      ]);
      setResult({ score: s, explain, neighbours: graph.n, illicit: graph.i, counts: graph.c, graphError: graph.err });
    } catch (e) {
      setError(errorText(e));
    } finally {
      setLoading(false);
    }
  }

  async function findPath() {
    if (!result) return;
    const t = Number(target);
    if (!Number.isInteger(t) || t < 0 || t > examples.maxId) {
      setPath(`Enter a whole number from 0 to ${examples.maxId}.`);
      return;
    }
    try {
      const p = await api.path(result.score.node_id, t);
      setPath(
        p.connected && p.path
          ? `Connected in ${p.path.hops} ${p.path.hops === 1 ? "hop" : "hops"}: ${p.path.path.map((n) => `${n.node_id} (${n.ground_truth})`).join(" to ")}.`
          : "No path of 8 hops or fewer connects these two transactions.",
      );
    } catch (e) {
      setPath(errorText(e));
    }
  }

  const r = result;
  return (
    <div>
      <form
        className="flex flex-wrap items-end gap-3"
        onSubmit={(ev) => {
          ev.preventDefault();
          void score(input);
        }}
      >
        <label className="grid gap-1 text-sm font-semibold">
          Transaction number
          <input className="field num w-40 font-normal" inputMode="numeric" value={input} onChange={(ev) => setInput(ev.target.value)} />
        </label>
        <button type="submit" className="btn" disabled={loading}>
          Score this transaction
        </button>
        <span className="text-sm text-ink-soft">
          Examples:{" "}
          {examples.transactions.map((t, i) => (
            <span key={t.id}>
              {i > 0 && ", "}
              <button
                type="button"
                className="underline"
                data-id
                onClick={() => {
                  setInput(String(t.id));
                  void score(String(t.id));
                }}
              >
                {t.id}
              </button>
            </span>
          ))}
        </span>
      </form>

      <div className="mt-6">
        <DemoState loading={loading} error={error} waitText="Scoring the transaction and asking the graph database for its neighbourhood." />
        {!loading && !error && !r && <p className="m-0 text-sm text-ink-soft">Nothing scored yet. Enter a transaction number or pick an example.</p>}
      </div>

      {r && (
        <div className="grid gap-8" data-testid="graph-result">
          <div className="grid gap-6 md:grid-cols-[minmax(0,16rem)_minmax(0,1fr)]">
            <div>
              <div className="text-xs text-ink-soft">XGBoost risk score for transaction <Live>{r.score.node_id}</Live></div>
              <div className="mt-1 flex items-baseline gap-3">
                <span className="text-xl font-semibold" data-testid="graph-desk-score"><Score value={r.score.score} /></span>
                <VerdictTag verdict={r.score.verdict} />
              </div>
              <p className="mt-2 mb-0 text-sm text-ink-soft">
                Known label: {r.score.ground_truth}. Time step <Live>{r.score.time_step}</Live>, {r.score.split} period
                {r.score.in_sample ? ". The model saw this transaction in training, so this score is in-sample" : ""}.
              </p>
            </div>
            {r.score.graph_features && (
              <table className="w-full border-collapse text-sm">
                <caption className="mb-2 text-left text-xs text-ink-soft">
                  The four graph statistics the scorer adds to the raw features, for this transaction
                </caption>
                <tbody>
                  {(
                    [
                      ["Degree (payments in and out)", r.score.graph_features.degree.toFixed(0)],
                      ["Out-to-in degree ratio", r.score.graph_features.out_in_degree_ratio.toFixed(2)],
                      ["PageRank", r.score.graph_features.pagerank.toExponential(2)],
                      ["Clustering coefficient", r.score.graph_features.clustering.toFixed(3)],
                    ] as const
                  ).map(([k, v]) => (
                    <tr key={k} className="rule-t">
                      <th scope="row" className="py-1.5 pr-4 text-left font-normal">{k}</th>
                      <td className="py-1.5 text-right"><Live>{v}</Live></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>

          <CaveatNote>
            The score comes from XGBoost alone. The neighbourhood below is an aid for reading the case; it did not decide
            anything. The risk bands are score thresholds set on validation-period transactions, not probabilities of fraud.
          </CaveatNote>

          {r.graphError && (
            <p role="alert" className="measure m-0 border-l-4 border-flag bg-sheet px-3 py-2 text-sm">
              The score above is real, but the neighbourhood is unavailable: the graph database did not answer. Check that
              the neo4j service is running and loaded.
            </p>
          )}

          {r.neighbours && r.counts && (
            <div>
              <h3 className="mb-2 text-base">Neighbourhood, from the graph database</h3>
              <p className="measure mt-0 mb-4 text-sm">
                <Live>{r.counts.neighbors}</Live> direct {r.counts.neighbors === 1 ? "neighbour" : "neighbours"};{" "}
                <Live>{r.counts.flagged_neighbors}</Live> scored medium or high by the model;{" "}
                <Live>{r.counts.known_illicit_neighbors}</Live> labelled illicit in the dataset.{" "}
                {r.illicit && r.illicit.length > 0 ? (
                  <>
                    Labelled-illicit transactions within three hops: {r.illicit.slice(0, 8).map((n, i) => (
                      <span key={n.node_id}>{i > 0 && ", "}<Live>{n.node_id}</Live> ({n.hops} {n.hops === 1 ? "hop" : "hops"})</span>
                    ))}
                    {r.illicit.length > 8 && " and more"}.
                  </>
                ) : (
                  "No labelled-illicit transaction within three hops."
                )}
              </p>
              {r.neighbours.length > 0 ? (
                <NeighbourhoodView centre={r.score} neighbours={r.neighbours} />
              ) : (
                <p className="m-0 text-sm text-ink-soft">This transaction has no neighbours within two hops.</p>
              )}
              <form
                className="mt-5 flex flex-wrap items-end gap-3"
                onSubmit={(ev) => {
                  ev.preventDefault();
                  void findPath();
                }}
              >
                <label className="grid gap-1 text-sm font-semibold">
                  Shortest path to transaction
                  <input className="field num w-40 font-normal" inputMode="numeric" value={target} onChange={(ev) => setTarget(ev.target.value)} />
                </label>
                <button type="submit" className="btn btn-quiet">Find the path</button>
              </form>
              {path && <p role="status" className="measure mt-3 mb-0 text-sm" data-live>{path}</p>}
            </div>
          )}

          {r.explain?.available && (
            <div>
              <h3 className="mb-2 text-base">Secondary explanation: what a graph attention network looked at</h3>
              <p className="measure mt-0 mb-3 text-sm text-ink-soft">
                From a separate, much weaker model (GAT, PR-AUC <Num id="graph.gat.first" />). It is not the source of the score
                above. It shows which incoming neighbours that model's attention weighted most.
              </p>
              {r.explain.top_attended_in_neighbors && r.explain.top_attended_in_neighbors.length > 0 ? (
                <table className="w-full max-w-xl border-collapse text-sm">
                  <thead>
                    <tr className="text-left text-xs text-ink-soft">
                      <th scope="col" className="pb-2 pr-3 font-normal">Incoming neighbour</th>
                      <th scope="col" className="pb-2 pr-3 font-normal">Attention weight</th>
                      <th scope="col" className="pb-2 font-normal">Known label</th>
                    </tr>
                  </thead>
                  <tbody>
                    {r.explain.top_attended_in_neighbors.map((n) => (
                      <tr key={n.node_id} className="rule-t">
                        <td className="py-1.5 pr-3"><Live>{n.node_id}</Live></td>
                        <td className="py-1.5 pr-3"><Live>{n.attention.toExponential(2)}</Live></td>
                        <td className="py-1.5">{n.ground_truth}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <p className="m-0 text-sm text-ink-soft">This transaction has no incoming payments, so there is nothing for attention to weigh.</p>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export function GraphPage() {
  return (
    <>
      <section className="pt-14" aria-labelledby="graph-h">
        <h1 id="graph-h" className="measure" style={{ fontSize: "var(--text-hero)" }}>
          On this dataset, cheap graph statistics beat message passing.
        </h1>
        <p className="measure mt-6 mb-0">
          Elliptic Bitcoin transactions: <Num id="graph.nodes" /> transactions, <Num id="graph.edges" /> payment flows,{" "}
          <Num id="graph.features" /> features each. <Num id="graph.licit" /> are labelled licit (<Num id="graph.licitPct" />),{" "}
          <Num id="graph.illicit" /> illicit (<Num id="graph.illicitPct" />) and <Num id="graph.unknown" /> have no label (
          <Num id="graph.unknownPct" />). Every model trains on time steps <Num id="graph.trainFirstStep" /> to{" "}
          <Num id="graph.trainLastStep" /> and is tested on <Num id="graph.testFirstStep" /> to <Num id="graph.testLastStep" />.
          The measure is PR-AUC, because with so few illicit transactions a model can look accurate while finding almost
          none of them.
        </p>
      </section>

      <Section title="Every model, one test" id="leaderboard">
        <DotPlot
          caption="Test PR-AUC on the time split, best first. A bracket is a three-seed mean with its spread."
          rows={LEADERBOARD}
          axisMax={1}
          axisLabel="PR-AUC, higher is better"
          hue="ledger"
        />
        <p className="measure mt-4 mb-0 text-sm text-ink-soft">
          Each row's yellow tag gives the run and the file. The same table, generated from the result files, is in the
          repository as docs/graph_leaderboard.md.
        </p>
      </Section>

      <Section title="Three checks on that result" id="checks">
        <EvidenceRow
          ids={["graph.sage.random", "graph.sage.temporal", "graph.sage.splitGap"]}
          claim={
            <>
              A random split would have reported <Num id="graph.sage.random" /> for a model that scores{" "}
              <Num id="graph.sage.temporal" /> on later transactions.
            </>
          }
          more={<Link to="/">The instrument on the overview page shows the two side by side.</Link>}
        />
        <EvidenceRow
          ids={["graph.camo.xgb", "graph.camo.sage", "graph.camo.care", "graph.camo.n"]}
          claim={
            <>
              On <Num id="graph.camo.n" /> illicit test transactions with no illicit neighbour, the defence built for
              camouflaged fraud did not help.
            </>
          }
          more={
            <>
              Recall at a matched <Num id="graph.camo.fpr" /> false-positive rate. Caveat: <Num id="graph.unknownPctRounded" /> of
              transactions are unlabelled, so "no illicit neighbour" mostly means "unlabelled neighbours".
            </>
          }
        >
          <DotPlot caption="Recall on camouflaged fraud." rows={CAMOUFLAGE} axisMax={1} axisLabel="recall, higher is better" hue="ledger" />
        </EvidenceRow>
        <EvidenceRow
          ids={["graph.explain.edgeRemoval", "graph.explain.node"]}
          claim={
            <>
              On one transaction, two explanation methods disagreed about its single neighbour. Removing that edge moved the
              prediction by about <Num id="graph.explain.edgeRemoval" />, so the attention weight was right and the explainer's
              edge score was not.
            </>
          }
          more={
            <>
              One node only (<Num id="graph.explain.node" />): this is a spot check, not a study of explanation methods.
            </>
          }
        />
      </Section>

      <Section title="Case desk: score a transaction" id="desk">
        <p className="measure mt-0 mb-6">
          Live against the local stack. The scorer is the XGBoost model from the top of the leaderboard; the neighbourhood
          comes from Neo4j.
        </p>
        <GraphDesk />
      </Section>

      <Section title="Limits" id="limits">
        <ul className="measure m-0 grid list-none gap-3 p-0">
          <li className="rule-t pt-3">One dataset. Nothing here says the same ordering holds on another transaction graph.</li>
          <li className="rule-t pt-3">
            Elliptic's edges have no type, so the RGCN edge types were derived from payment direction. That is a weaker test
            of RGCN than real typed edges would be.
          </li>
          <li className="rule-t pt-3">
            The graph neural network search had not plateaued: the best settings sat at the edge of the search range.
          </li>
          <li className="rule-t pt-3">
            The live desk does not show which raw features drove a score. Elliptic's features are anonymised, so a list of
            feature numbers would explain little.
          </li>
        </ul>
      </Section>
    </>
  );
}
