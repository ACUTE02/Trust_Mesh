"""WCAG 2.1 contrast check of the frontend's colour tokens, read from frontend/src/index.css.

Text pairs need 4.5:1; graphical marks (bars, dots, hatch, field borders) need 3:1 against what they sit on.
Run: .venv\\Scripts\\python.exe scripts/check_contrast.py
"""
import re
import sys
from pathlib import Path

CSS = Path(__file__).resolve().parents[1] / "frontend" / "src" / "index.css"
TEXT, MARK = 4.5, 3.0
PAIRS = [  # (foreground, background, minimum, what it is)
    ("ink", "slab", TEXT, "body text"), ("ink", "sheet", TEXT, "text on panels"),
    ("ink-soft", "slab", TEXT, "secondary text"), ("ink-soft", "sheet", TEXT, "secondary text on panels"),
    ("flag", "slab", TEXT, "shortcut label"), ("flag", "sheet", TEXT, "warning text on panels"),
    ("marker-ink", "marker", TEXT, "provenance tag text"), ("slab", "ink", TEXT, "button text"),
    ("ledger", "sheet", MARK, "graph dots"), ("ledger", "slab", MARK, "graph dots"), ("ledger", "track", MARK, "graph bar on its track"),
    ("spectrum", "sheet", MARK, "voice dots"), ("spectrum", "slab", MARK, "voice dots"), ("spectrum", "track", MARK, "voice bar on its track"),
    ("flag", "track", MARK, "shortcut hatch on its track"), ("ink", "track", MARK, "other-state tick on the track"),
    ("ink-soft", "sheet", MARK, "field border"),
]


def tokens(block):
    return dict(re.findall(r"--([\w-]+):\s*(#[0-9a-fA-F]{6})", block))


def luminance(hex_colour):
    channels = [int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    r, g, b = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def ratio(a, b):
    hi, lo = sorted((luminance(a), luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def main():
    css = CSS.read_text(encoding="utf-8")
    themes = {"light": tokens(css.split(":root {")[1].split("}")[0]),
              "dark": tokens(css.split(':root[data-theme="dark"] {')[1].split("}")[0])}
    failures = 0
    for theme, t in themes.items():
        for fg, bg, need, what in PAIRS:
            r = ratio(t[fg], t[bg])
            ok = r >= need
            failures += not ok
            print(f"{theme:5s} {fg:10s} on {bg:6s} {r:5.2f}:1 (needs {need}) {'ok  ' if ok else 'FAIL'} {what}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
