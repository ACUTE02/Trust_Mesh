"""Build notebooks/*.ipynb from their percent-format sources in notebooks/src/.

The sources use percent-format cells (`# %%`, `# %% [markdown]`). A `# %% inline <path>` cell is
replaced with that module's source, minus its single-line relative imports (`from .x import y`) --
those names are already defined by earlier inlined cells. This keeps Kaggle's one-file upload
self-contained while the real code lives, tested, in engines/.

Usage:
  python scripts/build_notebooks.py                 # build every notebook
  python scripts/build_notebooks.py --check         # fail if a committed .ipynb is out of date
  python scripts/build_notebooks.py 03_voice_baseline   # build just one
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "notebooks" / "src"
# Also matches a cross-engine import such as `from ..voice.evaluate import x` (the visual engine
# reuses the voice engine's bootstrap code; that module is inlined earlier in the same notebook).
RELATIVE_IMPORT = re.compile(r"^from \.+[\w.]* import .+$")


def parse_cells(text):
    cells, kind, lines = [], None, []

    def flush():
        if kind is None:
            return
        body = "\n".join(lines).strip("\n")
        if kind == "markdown":
            body = "\n".join(line[2:] if line.startswith("# ") else line.lstrip("#") for line in body.splitlines())
        if body.strip() or kind.startswith("inline:"):
            cells.append((kind, body))

    for line in text.splitlines():
        if line.startswith("# %%"):
            flush()
            header = line[4:].strip()
            if header == "[markdown]":
                kind = "markdown"
            elif header.startswith("inline "):
                kind = "inline:" + header[len("inline "):].strip()
            else:
                kind = "code"
            lines = []
        else:
            lines.append(line)
    flush()
    return cells


def inline_module(rel_path):
    src = (ROOT / rel_path).read_text(encoding="utf-8")
    for n, line in enumerate(src.splitlines(), 1):
        # A parenthesised import spans lines; only its first line would be stripped, leaving the
        # continuation in the notebook as a syntax error. Fail the build instead.
        if RELATIVE_IMPORT.match(line) and "(" in line and ")" not in line:
            raise SystemExit(f"{rel_path}:{n}: multi-line relative import -- split it into single-line imports")
    kept = [line for line in src.splitlines() if not RELATIVE_IMPORT.match(line)]
    return f"# ---- inlined from {rel_path} (edit the module, then rebuild) ----\n" + "\n".join(kept).strip("\n")


def build_notebook(source=None):
    source = Path(source) if source else SRC_DIR / "02_graph_full.py"
    nb_cells = []
    for kind, body in parse_cells(Path(source).read_text(encoding="utf-8")):
        if kind.startswith("inline:"):
            kind, body = "code", inline_module(kind[len("inline:"):])
        cell = {"cell_type": kind, "metadata": {}, "source": body.splitlines(keepends=True)}
        if kind == "code":
            cell.update({"execution_count": None, "outputs": []})
        nb_cells.append(cell)
    return {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


def code_cells(nb):
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def sources(only=None):
    for path in sorted(SRC_DIR.glob("*.py")):
        if only is None or path.stem == only:
            yield path


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    check = "--check" in sys.argv
    only = args[0] if args else None
    found = list(sources(only))
    if not found:
        sys.exit(f"no notebook source matching {only!r} in {SRC_DIR}")

    stale = []
    for src in found:
        target = ROOT / "notebooks" / f"{src.stem}.ipynb"
        text = json.dumps(build_notebook(src), indent=1, ensure_ascii=False) + "\n"
        if check:
            current = target.read_text(encoding="utf-8") if target.exists() else ""
            status = "up to date" if current == text else "OUT OF DATE"
            print(f"{target.name}: {status}")
            if current != text:
                stale.append(target.name)
        else:
            target.write_text(text, encoding="utf-8")
            print(f"wrote {target}")
    if stale:
        sys.exit(f"out of date: {', '.join(stale)} -- run: python scripts/build_notebooks.py")
