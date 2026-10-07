"""Neo4j: loads the Elliptic graph with XGBoost Tier-1+2 risk scores and runs the fraud-ring queries.

Schema: (:Tx {id, tx_id, time_step, label, split, xgb_score, verdict, degree, pagerank, ...})
        -[:SENDS_TO]->(:Tx)   (the real, directed Elliptic edges)
`id` is the row index shared with the PyG graph and the models (0..203768); `tx_id` is Elliptic's own.
"""
import time

from neo4j import GraphDatabase

from . import config
from .graph_service import LABEL_NAMES, split_of

BATCH = 5000
MAX_HOPS = 4


class Neo4jStore:
    def __init__(self, uri=None, user=None, password=None):
        self.driver = GraphDatabase.driver(uri or config.NEO4J_URI,
                                           auth=(user or config.NEO4J_USER, password or config.NEO4J_PASSWORD))

    def close(self):
        self.driver.close()

    def _run(self, query, **params):
        with self.driver.session() as session:
            return [r.data() for r in session.run(query, **params)]

    def wait_until_ready(self, timeout=180):
        deadline = time.time() + timeout
        while True:
            try:
                self.driver.verify_connectivity()
                return
            except Exception:
                if time.time() > deadline:
                    raise
                time.sleep(3)

    def stats(self):
        r = self._run("MATCH (n:Tx) RETURN count(n) AS nodes")[0]
        e = self._run("MATCH (:Tx)-[r:SENDS_TO]->(:Tx) RETURN count(r) AS edges")[0]
        return {"nodes": r["nodes"], "edges": e["edges"]}

    # -- loading ---------------------------------------------------------------------------------
    def load_graph(self, graph, with_features=True, progress=print):
        """Idempotent: if the expected node and edge counts are already present, does nothing."""
        n_nodes, edge_index = graph.data.num_nodes, graph.data.edge_index.numpy()
        have = self.stats()
        if have == {"nodes": n_nodes, "edges": edge_index.shape[1]}:
            progress(f"Neo4j already holds the full graph ({have}); nothing to load.")
            return have
        self._run("MATCH (n:Tx) CALL { WITH n DETACH DELETE n } IN TRANSACTIONS OF 10000 ROWS")
        self._run("CREATE CONSTRAINT tx_id_unique IF NOT EXISTS FOR (n:Tx) REQUIRE n.id IS UNIQUE")
        self._run("CREATE INDEX tx_label IF NOT EXISTS FOR (n:Tx) ON (n.label)")
        self._run("CREATE INDEX tx_score IF NOT EXISTS FOR (n:Tx) ON (n.xgb_score)")

        t0 = time.time()
        with self.driver.session() as session:
            for start in range(0, n_nodes, BATCH):
                rows = []
                for i in range(start, min(start + BATCH, n_nodes)):
                    score = float(graph.scores[i])
                    row = {"id": i, "tx_id": int(graph.tx_id[i]), "time_step": int(graph.time_step[i]),
                           "label": LABEL_NAMES[int(graph.labels[i])], "split": split_of(graph.time_step[i]),
                           "xgb_score": score, "verdict": graph.verdict(score),
                           "degree": float(graph.tier2[i, 0]), "out_in_ratio": float(graph.tier2[i, 1]),
                           "pagerank": float(graph.tier2[i, 2]), "clustering": float(graph.tier2[i, 3])}
                    if with_features:
                        row["features"] = graph.data.x[i].tolist()
                    rows.append(row)
                session.run("UNWIND $rows AS r CREATE (n:Tx) SET n = r", rows=rows).consume()
                if (start // BATCH) % 10 == 0:
                    progress(f"  nodes {min(start + BATCH, n_nodes)}/{n_nodes}")
            progress(f"nodes loaded in {time.time() - t0:.0f}s; loading edges")
            src, dst = edge_index
            for start in range(0, edge_index.shape[1], BATCH * 2):
                pairs = [{"s": int(a), "d": int(b)} for a, b in zip(src[start:start + BATCH * 2],
                                                                     dst[start:start + BATCH * 2])]
                session.run("UNWIND $pairs AS p MATCH (a:Tx {id: p.s}), (b:Tx {id: p.d}) CREATE (a)-[:SENDS_TO]->(b)",
                            pairs=pairs).consume()
        progress(f"graph loaded in {time.time() - t0:.0f}s")
        return self.stats()

    # -- fraud-ring queries ----------------------------------------------------------------------
    @staticmethod
    def _hops(hops):
        hops = int(hops)
        if not 1 <= hops <= MAX_HOPS:
            raise ValueError(f"hops must be between 1 and {MAX_HOPS}")
        return hops

    def neighbors_within(self, node_id, hops=2, limit=100):
        """Every transaction within `hops` edges of a node (either direction), nearest first."""
        hops = self._hops(hops)
        q = f"""
        MATCH p = (n:Tx {{id: $id}})-[:SENDS_TO*1..{hops}]-(m:Tx)
        WITH m, min(length(p)) AS hops
        RETURN m.id AS node_id, hops, m.label AS ground_truth, m.xgb_score AS score, m.verdict AS verdict
        ORDER BY hops, score DESC LIMIT $limit"""
        return self._run(q, id=node_id, limit=int(limit))

    def illicit_within(self, node_id, hops=2):
        """Known-illicit transactions within `hops` of a node: is it sitting next to a confirmed
        fraud ring? (label comes from Elliptic's ground truth, not from the model.)"""
        hops = self._hops(hops)
        q = f"""
        MATCH p = (n:Tx {{id: $id}})-[:SENDS_TO*1..{hops}]-(m:Tx {{label: 'illicit'}})
        WHERE m.id <> $id
        WITH m, min(length(p)) AS hops
        RETURN m.id AS node_id, hops, m.xgb_score AS score ORDER BY hops, score DESC"""
        return self._run(q, id=node_id)

    def shortest_path(self, source, target, max_hops=8):
        """Shortest undirected path between two transactions (e.g. two flagged accounts)."""
        q = f"""
        MATCH (a:Tx {{id: $a}}), (b:Tx {{id: $b}})
        MATCH p = shortestPath((a)-[:SENDS_TO*..{int(max_hops)}]-(b))
        RETURN [n IN nodes(p) | {{node_id: n.id, ground_truth: n.label, score: n.xgb_score, verdict: n.verdict}}] AS path,
               length(p) AS hops"""
        rows = self._run(q, a=source, b=target)
        return rows[0] if rows else None

    def flagged_neighbor_counts(self, node_id, flag_threshold):
        """Direct neighbors, and how many are flagged by the model / known illicit."""
        q = """
        MATCH (n:Tx {id: $id})
        OPTIONAL MATCH (n)-[:SENDS_TO]-(m:Tx)
        WITH n, collect(DISTINCT m) AS ms
        RETURN n.id AS node_id, size(ms) AS neighbors,
               size([m IN ms WHERE m.xgb_score >= $thr]) AS flagged_neighbors,
               size([m IN ms WHERE m.label = 'illicit']) AS known_illicit_neighbors"""
        rows = self._run(q, id=node_id, thr=float(flag_threshold))
        return rows[0] if rows else None
