"""Load the Elliptic graph (with XGBoost Tier-1+2 scores) into Neo4j. Idempotent.

Run: python -m serving.load_neo4j [--no-features]
"""
import argparse
import sys

from .graph_service import GraphService
from .neo4j_store import Neo4jStore
from .preflight import report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-features", action="store_true",
                        help="skip storing the 165-feature vector on every node (much smaller database)")
    args = parser.parse_args()
    text, blocking = report("graph")
    if text:
        print(text, flush=True)
    if blocking:
        sys.exit(1)
    graph = GraphService()
    try:
        graph.load(with_gat=False)
    except Exception as exc:  # most often: the Elliptic download failed (no internet)
        print(f"\nTrustMesh: the graph could not be loaded ({type(exc).__name__}: {exc}).\n"
              "If the Elliptic dataset is not in data/elliptic/ it is downloaded on first start, which needs internet.\n"
              "See README.md, \"Files you must supply\".", flush=True)
        sys.exit(1)
    print(f"scores ready (test PR-AUC check: {graph.verification['test_pr_auc']:.4f})")
    store = Neo4jStore()
    store.wait_until_ready()
    print("loaded:", store.load_graph(graph, with_features=not args.no_features))
    store.close()


if __name__ == "__main__":
    main()
