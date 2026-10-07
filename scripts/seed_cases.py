"""Fill the `cases` table with real graph-only rows through the running API.

Scores the highest-risk transactions of the held-out test period (time steps 36-49) with the real
XGBoost model, so the frontend's "Flagged cases" table has real content on first launch. Voice and
combined fields stay NULL: these cases have no voice clip, and nothing is made up for them.

Run (API up): .venv\\Scripts\\python.exe scripts/seed_cases.py [--n 25] [--api http://localhost:8000]
"""
import argparse
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=25)
    parser.add_argument("--api", default="http://localhost:8000")
    args = parser.parse_args()

    from serving.graph_service import GraphService
    graph = GraphService()
    graph.load(with_gat=False)  # only the scores are needed to pick which nodes to submit
    for node in graph.top_scored(args.n, split="test"):
        r = requests.get(f"{args.api}/graph/score/{node}", timeout=30)
        r.raise_for_status()
        s = r.json()
        print(f"node {node}: score {s['score']:.3f} {s['verdict']} (ground truth: {s['ground_truth']})")


if __name__ == "__main__":
    main()
