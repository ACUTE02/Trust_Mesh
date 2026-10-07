"""Load the Elliptic graph (with XGBoost Tier-1+2 scores) into Neo4j. Idempotent.

Run: python -m serving.load_neo4j [--no-features]
"""
import argparse

from .graph_service import GraphService
from .neo4j_store import Neo4jStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-features", action="store_true",
                        help="skip storing the 165-feature vector on every node (much smaller database)")
    args = parser.parse_args()
    graph = GraphService()
    graph.load(with_gat=False)
    print(f"scores ready (test PR-AUC check: {graph.verification['test_pr_auc']:.4f})")
    store = Neo4jStore()
    store.wait_until_ready()
    print("loaded:", store.load_graph(graph, with_features=not args.no_features))
    store.close()


if __name__ == "__main__":
    main()
