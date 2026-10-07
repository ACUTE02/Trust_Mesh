import { useState } from "react";
import { Link } from "react-router-dom";
import { DemoState, Score, VerdictTag, errorText } from "../components/caseParts";
import { DotPlot, type PlotRow } from "../components/DotPlot";
import { EngineerLayer, EngineerToggle, ForEngineers, GRAPH_SCORE_MEANING, Glossary, Meaning } from "../components/engineers";
import { CaveatNote, EvidenceRow, Live, MaskContext, Num, Section } from "../components/evidence";
import { TechView } from "../components/TechView";
import examples from "../data/examples.json";
import { api, readCallLog, startCallLog, type ApiCall, type GraphExplain, type GraphScore, type Neighbour, type Verdict } from "../lib/api";
import { RecordingBanner, isRecording, useHealth } from "../components/caseParts";

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

const LEADERBOARD_PLAIN: PlotRow[] = [
  { id: "graph.xgb12", seedMeanId: "graph.xgb12.mean", label: "A simple model that also counts connections (our best)", reference: true },
  { id: "graph.xgb1", label: "The same simple model, without the connection counts" },
  { id: "graph.sage.best", seedMeanId: "graph.sage.best.mean", label: "The best graph model, after long tuning" },
  { id: "graph.sage.wide", label: "The same graph model, with less tuning" },
  { id: "graph.sage.temporal", label: "The same graph model, with no tuning" },
];

const CAMOUFLAGE_PLAIN: PlotRow[] = [
  { id: "graph.camo.xgb", label: "The simple model", reference: true },
  { id: "graph.camo.sage", label: "A graph model" },
  { id: "graph.camo.care", label: "The graph model built to catch hidden fraud" },
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
function NeighbourhoodView({ centre, neighbours, mask }: { centre: GraphScore; neighbours: Neighbour[]; mask: boolean }) {
  const rings = [1, 2].map((h) => neighbours.filter((n) => n.hops === h).slice(0, h === 1 ? 14 : 22));
  const size = 300;
  const mid = size / 2;
  const dot = (verdict: Verdict, x: number, y: number, r: number, key: string) =>
    mask ? (
      <circle key={key} cx={x} cy={y} r={r - 1} fill="var(--track)" stroke="var(--ink-soft)" strokeWidth="1.5" />
    ) : verdict === "high" ? (
      <circle key={key} cx={x} cy={y} r={r} fill="var(--ink)" />
    ) : verdict === "medium" ? (
      <circle key={key} cx={x} cy={y} r={r - 1} fill="var(--sheet)" stroke="var(--ink)" strokeWidth="2" />
    ) : (
      <circle key={key} cx={x} cy={y} r={r - 1} fill="var(--sheet)" stroke="var(--ink-soft)" strokeWidth="1.5" strokeDasharray="2 2" />
    );
  return (
    <div className="grid gap-6 md:grid-cols-[minmax(0,18rem)_minmax(0,1fr)]">
      <figure className="m-0">
        <svg viewBox={`0 0 ${size} ${size}`} className="w-full max-w-[18rem]" role="img" aria-label={`The transaction at the centre, with ${rings[0].length} neighbours one hop away and ${rings[1].length} two hops away. The table lists them.`}>
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
              <th scope="col" className="pb-2 pr-3 font-normal">Steps away</th>
              <th scope="col" className="pb-2 pr-3 font-normal">Marked in the dataset</th>
              <th scope="col" className="pb-2 pr-3 font-normal">Fraud score</th>
              <th scope="col" className="pb-2 font-normal">Verdict</th>
            </tr>
          </thead>
          <tbody>
            {neighbours.map((n) => (
              <tr key={n.node_id} className="rule-t">
                <td className="py-1.5 pr-3"><Live>{n.node_id}</Live></td>
                <td className="py-1.5 pr-3"><Live>{n.hops}</Live></td>
                <td className="py-1.5 pr-3"><Live>{n.ground_truth}</Live></td>
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
  const [calls, setCalls] = useState<ApiCall[]>([]);
  const [cypher, setCypher] = useState<Record<string, string> | null>(null);
  const health = useHealth();
  const mask = isRecording(health);

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
    startCallLog();
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
      setCalls(readCallLog());
      setLoading(false);
    }
    api.cypher(2).then(setCypher).catch(() => setCypher(null));
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
          ? mask
            ? `Connected in ${p.path.hops} ${p.path.hops === 1 ? "hop" : "hops"}. The transactions on the path are hidden in recording mode.`
            : `Connected in ${p.path.hops} ${p.path.hops === 1 ? "hop" : "hops"}: ${p.path.path.map((n) => `${n.node_id} (${n.ground_truth})`).join(" to ")}.`
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
          <input className="field num w-40 font-normal" inputMode="numeric" type={mask ? "password" : "text"} autoComplete="off" value={input} onChange={(ev) => setInput(ev.target.value)} />
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
                {mask ? `example ${i + 1}` : t.id}
              </button>
            </span>
          ))}
        </span>
      </form>

      <div className="mt-6">
        <DemoState loading={loading} error={error} waitText="Scoring the transaction and looking up its neighbours." />
        {!loading && !error && !r && <p className="m-0 text-sm text-ink-soft">Nothing scored yet. Enter a transaction number or pick an example.</p>}
      </div>

      {mask && <RecordingBanner />}
      {r && (
        <MaskContext.Provider value={mask}>
        <div className="grid gap-8" data-testid="graph-result">
          <div className="grid gap-6 md:grid-cols-[minmax(0,16rem)_minmax(0,1fr)]">
            <div>
              <div className="text-xs text-ink-soft">Fraud score for transaction <Live>{r.score.node_id}</Live> (0 to 1, higher means more suspicious)</div>
              <div className="mt-1 flex items-baseline gap-3">
                <span className="text-xl font-semibold" data-testid="graph-desk-score"><Score value={r.score.score} /></span>
                <VerdictTag verdict={r.score.verdict} />
              </div>
              <p className="mt-2 mb-0 text-sm text-ink-soft">
                In the dataset this transaction is marked: <Live>{r.score.ground_truth}</Live>.
                {r.score.in_sample ? " The model saw it while learning, so this score is not a fair test." : ""}
              </p>
            </div>
            {r.score.graph_features && (
              <table className="w-full border-collapse text-sm">
                <caption className="mb-2 text-left text-xs text-ink-soft">
                  How this transaction is connected
                </caption>
                <tbody>
                  {(
                    [
                      ["Payments in and out", r.score.graph_features.degree.toFixed(0)],
                      ["Payments out compared with payments in", r.score.graph_features.out_in_degree_ratio.toFixed(2)],
                      ["How central it is in the network", r.score.graph_features.pagerank.toExponential(2)],
                      ["How tightly its neighbours link to each other", r.score.graph_features.clustering.toFixed(3)],
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
            The score comes from the simple model alone. The picture of neighbours below helps you read the case. It did not
            decide anything. The risk levels are cut-offs on the score, not chances of fraud.
          </CaveatNote>

          {r.graphError && (
            <p role="alert" className="measure m-0 border-l-4 border-flag bg-sheet px-3 py-2 text-sm">
              The score above is real, but the neighbours are unavailable. The graph database did not answer. Check that the
              neo4j service is running.
            </p>
          )}

          {r.neighbours && r.counts && (
            <div>
              <h3 className="mb-2 text-base">The transactions around it</h3>
              <p className="measure mt-0 mb-4 text-sm">
                <Live>{r.counts.neighbors}</Live> direct {r.counts.neighbors === 1 ? "neighbour" : "neighbours"};{" "}
                <Live>{r.counts.flagged_neighbors}</Live> scored medium or high risk;{" "}
                <Live>{r.counts.known_illicit_neighbors}</Live> marked as fraud in the dataset.{" "}
                {r.illicit && r.illicit.length > 0 ? (
                  <>
                    Known fraud within three steps: {r.illicit.slice(0, 8).map((n, i) => (
                      <span key={n.node_id}>{i > 0 && ", "}<Live>{n.node_id}</Live> ({n.hops} {n.hops === 1 ? "hop" : "hops"})</span>
                    ))}
                    {r.illicit.length > 8 && " and more"}.
                  </>
                ) : (
                  "No known fraud within three steps."
                )}
              </p>
              {r.neighbours.length > 0 ? (
                <NeighbourhoodView centre={r.score} neighbours={r.neighbours} mask={mask} />
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
                  <input className="field num w-40 font-normal" inputMode="numeric" type={mask ? "password" : "text"} autoComplete="off" value={target} onChange={(ev) => setTarget(ev.target.value)} />
                </label>
                <button type="submit" className="btn btn-quiet">Find the path</button>
              </form>
              {path && <p role="status" className="measure mt-3 mb-0 text-sm" data-live>{path}</p>}
            </div>
          )}

          {r.explain?.available && (
            <div>
              <h3 className="mb-2 text-base">A second opinion: which neighbours a graph model looked at</h3>
              <p className="measure mt-0 mb-3 text-sm text-ink-soft">
                This comes from a separate, much weaker model (it scores <Num id="graph.gat.first" />). It did not produce the
                score above. It shows which incoming neighbours that model paid most attention to.
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
                        <td className="py-1.5"><Live>{n.ground_truth}</Live></td>
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
        </MaskContext.Provider>
      )}
      {!mask && <TechView calls={calls} health={health} cypher={cypher} />}
    </div>
  );
}

const GLOSSARY: [string, string][] = [
  ["PR-AUC", "The score used on this page. It rewards finding fraud without raising many false alarms. 0 to 1, higher is better."],
  ["FPR", "False-positive rate: the share of normal transactions wrongly flagged."],
  ["Recall", "The share of real fraud that the model catches."],
  ["Temporal split", "Train on earlier transactions, test on later ones. The page calls it the time split."],
  ["XGBoost", "A widely used model built from many small decision trees. It reads a table of numbers, not a graph."],
  ["GNN", "Graph neural network: a model that passes information along the links between transactions."],
  ["GraphSAGE, GAT, GIN, RGCN, CARE-GNN", "Five kinds of graph neural network tried here."],
  ["Degree, PageRank, clustering coefficient", "Three simple counts that describe how a transaction is connected."],
  ["Seed", "The random starting point of a training run. Three seeds show how much a score moves by chance."],
  ["Calibration", "Checking that a score can be read as a real chance of fraud. Not done here."],
];

export function GraphPage() {
  return (
    <EngineerLayer page="graph">
      <section className="pt-14" aria-labelledby="graph-h">
        <h1 id="graph-h" className="measure" style={{ fontSize: "var(--text-hero)" }}>
          Simple counts of a transaction's connections beat the fancier graph models.
        </h1>
        <div className="measure mt-6 grid gap-2">
          <p className="m-0 text-lg">This model looks at who a transaction is connected to.</p>
          <p className="m-0">
            The data is <Num id="graph.nodes" /> Bitcoin transactions. <Num id="graph.illicit" /> are marked as fraud and{" "}
            <Num id="graph.licit" /> as normal. The rest have no label.
          </p>
          <p className="m-0">Every model learned from earlier transactions. Each was tested on later ones.</p>
          <Meaning>The score: {GRAPH_SCORE_MEANING}</Meaning>
        </div>
        <div className="mt-6">
          <EngineerToggle />
        </div>
        <ForEngineers title="protocol">
          <ul className="m-0 grid list-none gap-1.5 p-0">
            <li>
              Dataset: Elliptic. <Num id="graph.nodes" /> nodes, <Num id="graph.edges" /> directed edges, <Num id="graph.features" />{" "}
              features per node.
            </li>
            <li>
              Labels: licit <Num id="graph.licit" /> (<Num id="graph.licitPct" />), illicit <Num id="graph.illicit" /> (
              <Num id="graph.illicitPct" />), unknown <Num id="graph.unknown" /> (<Num id="graph.unknownPct" />). Unknown nodes
              are excluded from training loss and from metrics.
            </li>
            <li>
              Temporal split: train on time steps <Num id="graph.trainFirstStep" /> to <Num id="graph.trainLastStep" />, test on{" "}
              <Num id="graph.testFirstStep" /> to <Num id="graph.testLastStep" />; the steps between are validation.
            </li>
            <li>Metric: PR-AUC on the illicit class, labelled test nodes only. Class-weighted training.</li>
            <li>Seeds: three for the reference XGBoost and for the best GNN; single runs elsewhere.</li>
            <li>
              Served scorer: XGBoost Tier-1+2 (the <Num id="graph.features" /> raw features plus degree, out/in degree ratio,
              PageRank and clustering coefficient).
            </li>
          </ul>
        </ForEngineers>
      </section>

      <Section title="What we tried, and what each scored" id="leaderboard">
        <DotPlot
          caption="Best first. A bracket shows how much the score moved across three training runs."
          rows={LEADERBOARD_PLAIN}
          axisMax={1}
          axisLabel="the score, higher is better"
          hue="ledger"
        />
        <p className="measure mt-4 mb-0">Seven more graph models scored lower than these. The gap did not close with more tuning.</p>
        <ForEngineers title="full leaderboard">
          <DotPlot
            caption="Test PR-AUC, temporal split, sorted. Bracket: three-seed mean with its standard deviation."
            rows={LEADERBOARD}
            axisMax={1}
            axisLabel="PR-AUC, higher is better"
            hue="ledger"
          />
          <p className="mt-3 mb-0 text-xs text-ink-soft">
            Generated table: docs/graph_leaderboard.md. GNN search total: <Num id="graph.trials" /> trials across phases.
          </p>
        </ForEngineers>
      </Section>

      <Section title="Three checks on that result" id="checks">
        <EvidenceRow
          ids={["graph.sage.random", "graph.sage.temporal", "graph.sage.splitGap"]}
          claim={
            <>
              Tested the easy way, a graph model scores <Num id="graph.sage.random" />. Tested the honest way, it scores{" "}
              <Num id="graph.sage.temporal" />.
            </>
          }
          more={
            <>
              The easy way mixes old and new transactions. <Link to="/">The first page shows both side by side.</Link>
            </>
          }
        />
        <EvidenceRow
          ids={["graph.camo.xgb", "graph.camo.sage", "graph.camo.care", "graph.camo.n"]}
          claim={<>Fraud that hides among normal transactions: the model built to catch it did not help.</>}
          more={
            <>
              We took <Num id="graph.camo.n" /> fraud transactions with no known-fraud neighbour. Each model was allowed the same
              number of false alarms. The chart shows the share of that fraud each one caught. One caution: most transactions
              have no label, so "no fraud neighbour" often just means "unlabelled neighbour".
            </>
          }
        >
          <DotPlot caption="Share of the hidden fraud each model caught." rows={CAMOUFLAGE_PLAIN} axisMax={1} axisLabel="share caught, higher is better" hue="ledger" />
          <ForEngineers title="camouflage check">
            <p className="m-0">
              Recall @ <Num id="graph.camo.fpr" /> FPR (n = <Num id="graph.camo.n" />): illicit test nodes with zero
              illicit-labelled neighbours; threshold per model matched to <Num id="graph.camo.fpr" /> FPR on licit test nodes.
              CARE-GNN <Num id="graph.camo.care" />, GraphSAGE <Num id="graph.camo.sage" />, XGBoost Tier-1+2{" "}
              <Num id="graph.camo.xgb" />. <Num id="graph.unknownPctRounded" /> of nodes are unlabelled.
            </p>
          </ForEngineers>
        </EvidenceRow>
        <EvidenceRow
          ids={["graph.explain.edgeRemoval", "graph.explain.node"]}
          claim={<>We checked one explanation by hand. One tool was right and one was wrong.</>}
          more={
            <>
              Two tools disagreed about whether a neighbouring transaction mattered. We removed it. The prediction moved by
              about <Num id="graph.explain.edgeRemoval" />. So it did not matter. This was one transaction only.
            </>
          }
        >
          <ForEngineers title="explanation cross-check">
            <p className="m-0">
              Node <Num id="graph.explain.node" />, one incoming edge. GAT layer-1 attention gave the edge almost no weight;
              GNNExplainer's edge mask called it important. Edge-removal ablation: GAT's prediction changed by about{" "}
              <Num id="graph.explain.edgeRemoval" />, so attention was right. One node, not a study.
            </p>
          </ForEngineers>
        </EvidenceRow>
      </Section>

      <Section title="Try it: score a transaction" id="desk">
        <p className="measure mt-0 mb-6">Type a transaction number. You get its fraud score and the transactions around it.</p>
        <GraphDesk />
      </Section>

      <Section title="Limits" id="limits">
        <ul className="measure m-0 grid list-none gap-3 p-0">
          <li className="rule-t pt-3">We tested on one dataset only.</li>
          <li className="rule-t pt-3">The graph models might still improve with more tuning. We stopped before they levelled off.</li>
          <li className="rule-t pt-3">The score is a ranking, not a probability. A score halfway up does not mean an even chance of fraud.</li>
          <li className="rule-t pt-3">The live desk does not show which details of a transaction drove its score.</li>
        </ul>
        <ForEngineers title="limits">
          <ul className="m-0 grid list-none gap-1.5 p-0">
            <li>Single dataset; no cross-dataset check for the graph engine.</li>
            <li>RGCN relations are derived from payment direction (Elliptic edges are untyped).</li>
            <li>GNN search not plateaued: best hyperparameters sat at the search boundary.</li>
            <li>XGBoost was trained with a class weight, so its outputs are uncalibrated; verdict bands are score thresholds from validation-period licit nodes.</li>
            <li>No per-feature attribution in the desk: the API is unchanged and the features are anonymised.</li>
            <li>GAT-tuned and GIN figures are from the logged run; a later re-run differs slightly (see their tags).</li>
          </ul>
        </ForEngineers>
        <ForEngineers title="glossary">
          <Glossary terms={GLOSSARY} />
        </ForEngineers>
      </Section>
    </EngineerLayer>
  );
}
