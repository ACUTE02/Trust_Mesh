"""WCAG 2.1 contrast check of the frontend's colour tokens: the base tokens in frontend/src/index.css and every
world palette (six worlds and the 404, light and dark) in frontend/src/worlds.css.

Text pairs need 4.5:1; graphical marks (bars, dots, hatch, field borders) need 3:1 against what they sit on.
Text that sits over a 3D scene sits on a scrim (the slab colour at 90 percent opacity) or a panel (98 percent).
Whatever the canvas draws behind it, the background is then at worst opacity x slab + (1 - opacity) x (black or
white), so those text pairs are also checked against both extremes. This is arithmetic on the palette, not a measurement of the rendered pixels.

Run: .venv\\Scripts\\python.exe scripts/check_contrast.py
"""
import re
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "frontend" / "src"
TEXT, MARK = 4.5, 3.0
SURFACES = {"scrim": 0.9, "panel": 0.98}  # opacity of .scrim (text cards over a scene) and .panel (data panels) in worlds.css
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
OVER_SCENE = {"scrim": [("ink", TEXT), ("ink-soft", TEXT)], "panel": [("ink", TEXT), ("ink-soft", TEXT), ("flag", TEXT)]}


def tokens(block):
    return dict(re.findall(r"--([\w-]+):\s*(#[0-9a-fA-F]{6})", block))


def rgb(hex_colour):
    return [int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5)]


def luminance(channels):
    r, g, b = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def ratio(a, b):
    hi, lo = sorted((luminance(a), luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def world_palettes(css):
    """(world, theme) -> tokens, read from the selectors of worlds.css. A rule may list several worlds."""
    found = {}
    for selectors, body in re.findall(r"((?:html[^{}]*?\{)|(?:[^{}]*html\[[^{}]*?)\s*)\{([^{}]*)\}", css):
        for world, theme in re.findall(r'html(?::not\(\[data-world\]\)|\[data-world="(\w+)"\])\[data-theme="(\w+)"\]', selectors):
            found.setdefault((world or "home", theme), {}).update(tokens(body))
    return found


def main():
    index = (SRC / "index.css").read_text(encoding="utf-8")
    base = {"dark": tokens(index.split(":root {")[1].split("}")[0])}  # the site is dark only
    palettes = {(f"base", theme): t for theme, t in base.items()}
    for key, t in world_palettes((SRC / "worlds.css").read_text(encoding="utf-8")).items():
        palettes[key] = {**base[key[1]], **t}  # a world overrides the base tokens it names
    failures = 0
    for (world, theme), t in sorted(palettes.items()):
        for fg, bg, need, what in PAIRS:
            r = ratio(rgb(t[fg]), rgb(t[bg]))
            ok = r >= need
            failures += not ok
            print(f"{world:8s} {theme:5s} {fg:10s} on {bg:6s} {r:5.2f}:1 (needs {need}) {'ok  ' if ok else 'FAIL'} {what}")
        slab = rgb(t["slab"])
        for surface, opacity in SURFACES.items():
            for extreme, name in (([0.0] * 3, "black"), ([1.0] * 3, "white")):
                behind = [opacity * s + (1 - opacity) * e for s, e in zip(slab, extreme)]
                for fg, need in OVER_SCENE[surface]:
                    r = ratio(rgb(t[fg]), behind)
                    ok = r >= need
                    failures += not ok
                    print(f"{world:8s} {theme:5s} {fg:10s} on {surface} over {name:5s} {r:5.2f}:1 (needs {need}) {'ok  ' if ok else 'FAIL'}")
    print(f"\n{len(palettes)} palettes checked, {failures} failures")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
