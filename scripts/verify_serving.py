"""End-to-end check of a RUNNING stack: every endpoint, with real models and a real Neo4j.

Prints what came back so the output can be pasted into the project log as evidence. Exits non-zero
on the first failed expectation. Run (stack up): .venv\\Scripts\\python.exe scripts/verify_serving.py
"""
import argparse
import json
import sys

import requests

failures = []


def check(name, condition, detail=""):
    print(f"[{'ok' if condition else 'FAIL'}] {name} {detail}")
    if not condition:
        failures.append(name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--node", type=int, default=154134, help="an illicit test-period node")
    args = parser.parse_args()
    get = lambda p, **kw: requests.get(f"{args.api}{p}", timeout=300, **kw)  # noqa: E731
    quiet = {"record": "false"}  # this script must not leave test rows in the cases table
    cases_before = len(get("/cases", params={"limit": 1000}).json()["cases"])

    h = get("/health").json()
    print(json.dumps(h, indent=2))
    check("graph engine ready", h["graph"]["ready"])
    check("XGBoost reproduces Phase 1D test PR-AUC", abs(h["graph"]["test_pr_auc_check"]["test_pr_auc"] - 0.74743) < 1e-4,
          f"({h['graph']['test_pr_auc_check']['test_pr_auc']:.5f})")
    check("neo4j connected with the full graph", h["neo4j"].get("nodes") == 203769 and h["neo4j"].get("edges") == 234355,
          f"({h['neo4j']})")

    s = get(f"/graph/score/{args.node}", params=quiet).json()
    check("/graph/score is XGBoost Tier-1+2", s["model"] == "xgboost_tier1_tier2", f"(score {s['score']:.4f} {s['verdict']}, truth {s['ground_truth']})")
    e = get(f"/graph/explain/{args.node}").json()
    check("/graph/explain labels itself secondary", "NOT the source of the risk score" in e.get("label", ""))

    n = get(f"/graph/query/neighbors/{args.node}", params={"hops": 2}).json()
    check("Cypher: neighbors within 2 hops", n["count"] > 0, f"({n['count']} transactions)")
    ill = get(f"/graph/query/illicit/{args.node}", params={"hops": 3}).json()
    print("  known-illicit within 3 hops:", ill["known_illicit_within_hops"][:5])
    fn = get(f"/graph/query/flagged-neighbors/{args.node}").json()
    check("Cypher: flagged-neighbor counts", fn["neighbors"] >= 1, f"({fn})")
    if n["neighbors"]:
        far = n["neighbors"][-1]["node_id"]
        p = get("/graph/query/path", params={"source": args.node, "target": far}).json()
        check("Cypher: shortest path", p["connected"], f"({p['path']['hops'] if p['path'] else None} hops to {far})")

    clips = get("/voice/clips").json()["clips"]
    check("demo clips available", len(clips) > 0, f"({len(clips)})")
    for clip in clips:
        v = requests.post(f"{args.api}/voice/score", data={"clip": clip["clip"]}, params=quiet, timeout=300).json()
        print(f"  {clip['clip']:32s} true={clip.get('true_label', '?'):9s} spoof_prob={v['score']:.4f} "
              f"verdict={v['verdict']} model={v['model']}{' (fallback)' if v['fallback_used'] else ''}")
    if clips:
        r = requests.post(f"{args.api}/case/score", data={"node_id": args.node, "clip": clips[0]["clip"]}, params=quiet, timeout=300).json()
        f = r["fusion"]
        expected = f["weights"]["graph"] * r["graph"]["score"] + f["weights"]["voice"] * r["voice"]["score"]
        check("/case/score fuses both real scores", abs(f["combined_score"] - expected) < 1e-9,
              f"(graph {r['graph']['score']:.3f} + voice {r['voice']['score']:.3f} -> {f['combined_score']:.3f} {f['combined_verdict']})")

    flagged = get("/cases/flagged").json()["cases"]
    check("flagged cases, if any, are listed highest risk first",
          flagged == sorted(flagged, key=lambda c: -c["risk_score"]), f"({len(flagged)} rows)")
    cases_after = len(get("/cases", params={"limit": 1000}).json()["cases"])
    check("this script left no rows in the cases table", cases_after == cases_before, f"({cases_before} before, {cases_after} after)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
