"""Re-check every number in the frontend's single data module against its source.

frontend/src/data/evidence.json gives each entry a source: a repo file plus either `key` (a path into
that JSON file) or `text` (a string that must appear in that file, and must contain the value as shown).
This script resolves every source and fails if a value, spread or interval does not match at the
number of decimals the page displays. A derived entry (a plain-language figure such as "1.7 in 100")
must name the sourced entry it comes from and its formula, and is recomputed from that entry.

Run: .venv\\Scripts\\python.exe scripts/check_frontend_numbers.py
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "frontend" / "src" / "data" / "evidence.json"
_cache = {}


def load(rel):
    # TRUSTMESH_RESULTS_JSON=<file> checks against another copy of artifacts/graph/results.json
    # (for example the sanitised copy, which has no per-transaction block).
    if rel == "artifacts/graph/results.json" and os.environ.get("TRUSTMESH_RESULTS_JSON"):
        rel = os.environ["TRUSTMESH_RESULTS_JSON"]
    path = (ROOT / rel).resolve()
    if path not in _cache:
        text = path.read_text(encoding="utf-8")
        _cache[path] = (text, json.loads(text) if path.suffix == ".json" else None)
    return _cache[path]


def resolve(doc, key):
    node = doc
    for part in [p.strip() for p in key.split(">")]:
        if part.startswith("["):  # list filter: [field=value;field=value]
            want = dict(pair.split("=") for pair in part[1:-1].split(";"))
            matches = [row for row in node if all(str(row.get(k)) == v for k, v in want.items())]
            assert len(matches) == 1, f"{part}: {len(matches)} matches"
            node = matches[0]
        elif isinstance(node, list):
            node = node[int(part)]
        else:
            node = node[part]
    return node


def shown(value, decimals):
    return f"{value:.{decimals}f}"


def main():
    entries = json.loads(EVIDENCE.read_text(encoding="utf-8"))["entries"]
    problems = []
    for name, e in entries.items():
        src, dec = e.get("source") or {}, e["decimals"]
        for field in ("measures", "runDate", "phase"):
            if not e.get(field):
                problems.append(f"{name}: missing {field}")
        if src.get("derivedFrom"):  # a plain-language restatement of another sourced entry
            parent = entries.get(src["derivedFrom"])
            if parent is None or parent.get("source", {}).get("derivedFrom"):
                problems.append(f"{name}: derivedFrom must name a sourced (non-derived) entry")
            elif not src.get("formula"):
                problems.append(f"{name}: derived entry has no formula")
            elif shown(parent["value"] * src.get("scale", 1), dec) != shown(e["value"], dec):
                problems.append(f"{name}: page shows {shown(e['value'], dec)}, but "
                                f"{src['derivedFrom']} x {src.get('scale', 1)} = {parent['value'] * src.get('scale', 1)}")
            continue
        if not src.get("file") or not (src.get("key") or src.get("text")):
            problems.append(f"{name}: no source")
            continue
        try:
            text, doc = load(src["file"])
        except FileNotFoundError:
            problems.append(f"{name}: source file {src['file']} not found")
            continue
        if "key" in src:
            try:
                got = resolve(doc, src["key"]) * src.get("scale", 1)
            except (KeyError, IndexError, TypeError):
                problems.append(f"{name}: key not present in {src['file']}: {src['key']}")
                continue
            if shown(abs(got), dec) != shown(abs(e["value"]), dec):
                problems.append(f"{name}: page shows {shown(e['value'], dec)}, source has {got}")
            if "spread" in e:
                spread = resolve(doc, src["spreadKey"])
                if shown(spread, dec) != shown(e["spread"], dec):
                    problems.append(f"{name}: spread {e['spread']} vs source {spread}")
            if "interval" in e:
                for mine, key in zip(e["interval"], src["intervalKeys"]):
                    got_i = resolve(doc, key)
                    if abs(got_i - mine) > 0.6 * 10 ** -len(str(mine).split(".")[-1]):
                        problems.append(f"{name}: interval bound {mine} vs source {got_i}")
        else:
            if src["text"] not in text:
                problems.append(f"{name}: text not found in {src['file']}: {src['text']!r}")
            plain = src["text"].replace(",", "")
            candidates = {shown(abs(e["value"]), dec), str(abs(e["value"])), f"{abs(e['value']):g}"}
            if not any(c in plain for c in candidates):
                problems.append(f"{name}: source text does not contain the value {e['value']}")
            if "spread" in e and shown(e["spread"], dec) not in plain:
                problems.append(f"{name}: source text does not contain the spread {e['spread']}")
            for bound in e.get("interval", []):
                if f"{abs(bound):g}" not in plain and shown(abs(bound), dec) not in plain:
                    problems.append(f"{name}: source text does not contain interval bound {bound}")
    print(f"checked {len(entries)} entries against their sources")
    for p in problems:
        print("MISMATCH", p)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
