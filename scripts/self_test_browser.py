"""Self-test, browser part (ST6 of PHASE_2_SELF_TEST_RUNBOOK.md). Headless Edge/Chrome over the DevTools protocol.

Checks, against the running site:
  1. Keyboard only: Tab through every page, list anything interactive that Tab never reaches, then run
     each desk with the keyboard (Enter / Space), not the mouse.
  2. Reduced motion: with prefers-reduced-motion emulated, the instrument's load animation is off.
  3. axe-core on each page, light and dark, plain layer and engineer layer.
  4. Microphone flow with Chromium's fake capture device fed by a synthetic tone (NOT real speech).
  5. No horizontal overflow at 390 / 768 / 1440 in both themes.
Aggregates are written to data/self_test/browser.json (local, not committed).

Run (stack up): .venv\\Scripts\\python.exe scripts/self_test_browser.py [--base http://localhost:8080]
Needs `websockets`, `requests`, Edge or Chrome, and axe-core installed in frontend/node_modules.
"""
import argparse
import asyncio
import json
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import requests
import soundfile as sf
import websockets

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "self_test"
BROWSERS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe"]
PAGES = {"landing": "/", "graph": "/graph", "voice": "/voice", "case": "/case", "evidence": "/evidence", "about": "/about", "lost": "/not-a-page"}
AXE = ROOT / "frontend" / "node_modules" / "axe-core" / "axe.min.js"

DESCRIBE = """(() => { const e = document.activeElement; if (!e || e === document.body) return null;
  const label = (e.getAttribute('aria-label') || e.innerText || e.value || e.name || '').trim().replace(/\\s+/g, ' ').slice(0, 40);
  return e.tagName.toLowerCase() + (e.type ? '[' + e.type + ']' : '') + ':' + label; })()"""
ALL_FOCUSABLE = """(() => [...document.querySelectorAll('a[href], button:not([disabled]), input:not([disabled]), select, textarea, summary, [tabindex]')]
  .filter(e => e.tabIndex >= 0 && e.getClientRects().length > 0 || (e.tagName === 'INPUT' && e.type === 'radio') || (e.tagName === 'INPUT' && e.type === 'file'))
  .map(e => { const label = (e.getAttribute('aria-label') || e.innerText || e.value || e.name || '').trim().replace(/\\s+/g, ' ').slice(0, 40);
              return e.tagName.toLowerCase() + (e.type ? '[' + e.type + ']' : '') + ':' + label; }))()"""


class Page:
    def __init__(self, ws):
        self.ws, self.n = ws, 0

    async def send(self, method, **params):
        self.n += 1
        await self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            message = json.loads(await self.ws.recv())
            if message.get("id") == self.n:
                return message.get("result", {})

    async def js(self, expression, await_promise=False):
        r = await self.send("Runtime.evaluate", expression=expression, returnByValue=True, awaitPromise=await_promise)
        return r.get("result", {}).get("value")

    async def key(self, key, code=None, vk=None, shift=False):
        common = {"key": key, "code": code or key, "windowsVirtualKeyCode": vk or 0, "modifiers": 8 if shift else 0}
        text = {"Enter": "\r", " ": " "}.get(key)
        await self.send("Input.dispatchKeyEvent", type="keyDown", **common, **({"text": text} if text else {}))
        await self.send("Input.dispatchKeyEvent", type="keyUp", **common)

    async def goto(self, url, wait=2.5):
        await self.send("Page.navigate", url=url)
        await asyncio.sleep(wait)

    async def wait_for(self, expression, seconds=40):
        for _ in range(int(seconds * 2)):
            if await self.js(expression):
                return True
            await asyncio.sleep(0.5)
        return False

    async def tab_to(self, wanted, limit=120):
        """Press Tab until the focused element's description starts with `wanted`."""
        for _ in range(limit):
            await self.key("Tab", "Tab", 9)
            if (await self.js(DESCRIBE) or "").startswith(wanted):
                return True
        return False


async def keyboard_checks(page, base):
    result = {}
    await page.send("Emulation.setDeviceMetricsOverride", width=1280, height=900, deviceScaleFactor=1, mobile=False)
    for name, path in PAGES.items():
        await page.goto(base + path)
        await page.js("sessionStorage.setItem('tm-engineers-%s', '1'); location.reload()" % name)
        await asyncio.sleep(2.5)
        expected = await page.js(ALL_FOCUSABLE)
        seen, order = [], []
        for _ in range(len(expected) + 25):
            await page.key("Tab", "Tab", 9)
            d = await page.js(DESCRIBE)
            if d is None:
                break
            order.append(d)
            if d not in seen:
                seen.append(d)
        # a radio group is one tab stop; file inputs are reached through their visible button
        unreachable = [e for e in dict.fromkeys(expected)
                       if e not in seen and not e.startswith("input[radio]") and not e.startswith("input[file]")]
        radios_reached = any(d.startswith("input[radio]") for d in seen) or not any(e.startswith("input[radio]") for e in expected)
        in_dom_order = [d for d in dict.fromkeys(expected) if d in seen] == [d for d in seen if d in expected]
        result[name] = {"interactive_elements": len(set(expected)), "reached_by_tab": len(seen), "unreachable": unreachable,
                        "radio_group_reachable": radios_reached, "focus_follows_page_order": in_dom_order,
                        "first_stops": order[:4]}

    # instrument: arrow keys switch the radio group
    await page.goto(base + "/")
    ok = await page.tab_to("input[radio]")
    before = await page.js("document.querySelector('[aria-live] [data-evidence]').textContent")
    await page.key("ArrowRight", "ArrowRight", 39)
    await asyncio.sleep(1.2)
    after = await page.js("document.querySelector('[aria-live] [data-evidence]').textContent")
    result["instrument_arrow_keys"] = {"reached": ok, "before": before, "after": after, "pass": ok and before != after}
    outline = await page.js("(() => { const s = getComputedStyle(document.activeElement.closest('label') || document.activeElement); return s.outlineStyle + ' ' + s.outlineWidth; })()")
    result["focus_outline_on_instrument"] = outline

    # landing desk: Tab to the button, press Enter
    await page.goto(base + "/")
    ok = await page.tab_to("button[submit]:Score this case")
    await page.key("Enter", "Enter", 13)
    done = await page.wait_for("!!document.querySelector('[data-testid=case-card]')")
    result["landing_desk_by_keyboard"] = {"reached": ok, "scored": done, "pass": ok and done}
    ok = await page.tab_to("button[button]:Show the technical view")
    await page.key(" ", "Space", 32)
    opened = await page.wait_for("!!document.querySelector('[data-testid=tech-view]')", 5)
    result["technical_view_by_keyboard"] = {"reached": ok, "opened": opened, "pass": ok and opened}

    # graph desk: Tab to the number box, press Enter to submit the form
    await page.goto(base + "/graph")
    ok = await page.tab_to("input[text]")
    await page.key("Enter", "Enter", 13)
    done = await page.wait_for("!!document.querySelector('[data-testid=graph-result]')")
    result["graph_desk_by_keyboard"] = {"reached": ok, "scored": done, "pass": ok and done}

    # voice desk: the file is attached through the protocol (a file picker cannot be driven by keys in
    # a headless browser); everything after that is keyboard only.
    await page.goto(base + "/voice")
    doc = await page.send("DOM.getDocument")
    node = await page.send("DOM.querySelector", nodeId=doc["root"]["nodeId"], selector="input[type=file]")
    await page.send("DOM.setFileInputFiles", nodeId=node["nodeId"], files=[str(OUT / "synthetic_tone.wav")])
    ready = await page.wait_for("!!document.querySelector('[data-testid=clip-ready]')", 15)
    picker = await page.tab_to("button[button]:Choose an audio file")
    ok = await page.tab_to("button[button]:Score this clip")
    await page.key("Enter", "Enter", 13)
    done = await page.wait_for("!!document.querySelector('[data-testid=voice-result]')", 60)
    result["voice_desk_by_keyboard"] = {"file_button_reachable": picker, "clip_loaded": ready, "reached": ok, "scored": done,
                                        "pass": picker and ready and ok and done,
                                        "note": "the file itself was attached by the test tool, not typed"}
    return result


async def reduced_motion(page, base):
    await page.send("Emulation.setEmulatedMedia", features=[{"name": "prefers-reduced-motion", "value": "reduce"}])
    await page.goto(base + "/")
    reduced = await page.js("getComputedStyle(document.querySelector('.bar-anim')).transitionDuration")
    await asyncio.sleep(1.0)
    width = await page.js("document.querySelector('.bar-anim').style.width")
    await page.send("Emulation.setEmulatedMedia", features=[{"name": "prefers-reduced-motion", "value": "no-preference"}])
    await page.goto(base + "/")
    normal = await page.js("getComputedStyle(document.querySelector('.bar-anim')).transitionDuration")
    return {"transition_with_reduced_motion": reduced, "transition_normally": normal, "bar_at_final_width": width,
            "pass": reduced in ("0s", "") and normal not in ("0s", "")}


async def axe(page, base):
    source = AXE.read_text(encoding="utf-8")
    runs = []
    for theme in ("light", "dark"):
        await page.send("Emulation.setEmulatedMedia", features=[{"name": "prefers-color-scheme", "value": theme}])
        for name, path in PAGES.items():
            for layer in ("plain", "engineers"):
                await page.goto(base + path, wait=1.0)
                await page.js("localStorage.removeItem('tm-theme'); sessionStorage.setItem('tm-engineers-%s', '%s'); location.reload()"
                              % (name, "1" if layer == "engineers" else "0"))
                await asyncio.sleep(2.5)
                await page.js(source)
                assert await page.js("typeof axe.run") == "function", "axe-core did not load"
                found = await page.js(
                    "axe.run(document, {resultTypes: ['violations']}).then(r => r.violations.map(v => "
                    "({id: v.id, impact: v.impact, help: v.help, nodes: v.nodes.length, example: v.nodes[0].target.join(' ')})))",
                    await_promise=True)
                runs.append({"page": name, "theme": theme, "layer": layer, "violations": found or []})
    by_impact = {}
    for run in runs:
        for v in run["violations"]:
            by_impact[v["impact"]] = by_impact.get(v["impact"], 0) + 1
    distinct = {}
    for run in runs:
        for v in run["violations"]:
            d = distinct.setdefault(v["id"], {"impact": v["impact"], "help": v["help"], "where": [], "example": v["example"]})
            d["where"].append(f"{run['page']}/{run['theme']}/{run['layer']}")
    return {"runs": len(runs), "violations_by_impact": by_impact, "distinct_rules": distinct}


async def overflow(page, base):
    worst, checked = 0, 0
    for theme in ("light", "dark"):
        await page.send("Emulation.setEmulatedMedia", features=[{"name": "prefers-color-scheme", "value": theme}])
        for width in (390, 768, 1440):
            await page.send("Emulation.setDeviceMetricsOverride", width=width, height=900, deviceScaleFactor=1, mobile=False)
            for name, path in PAGES.items():
                for layer in ("0", "1"):
                    await page.goto(base + path, wait=0.8)
                    await page.js("sessionStorage.setItem('tm-engineers-%s', '%s'); location.reload()" % (name, layer))
                    await asyncio.sleep(1.5)
                    worst = max(worst, await page.js("document.documentElement.scrollWidth - document.documentElement.clientWidth") or 0)
                    checked += 1
    return {"combinations_checked": checked, "worst_overflow_px": worst, "pass": worst == 0}


async def microphone(page, base):
    await page.goto(base + "/voice")
    clicked = await page.js("(() => { const b = [...document.querySelectorAll('button')].find(b => b.textContent.trim() === 'Record from the microphone'); if (!b) return false; b.click(); return true; })()")
    recording = await page.wait_for("[...document.querySelectorAll('button')].some(b => b.textContent.trim() === 'Stop recording')", 10)
    problem = await page.js("document.querySelector('[role=alert]')?.innerText || null")
    await asyncio.sleep(4)
    await page.js("[...document.querySelectorAll('button')].find(b => b.textContent.trim() === 'Stop recording')?.click()")
    ready = await page.wait_for("!!document.querySelector('[data-testid=clip-ready]')", 15)
    seconds = await page.js("document.querySelector('[data-testid=clip-ready]')?.innerText || null")
    await page.js("[...document.querySelectorAll('button')].find(b => b.textContent.trim() === 'Score this clip')?.click()")
    scored = await page.wait_for("!!document.querySelector('[data-testid=voice-result]')", 60)
    return {"record_button_found": clicked, "recording_started": recording, "error_shown": problem, "clip_produced": ready,
            "clip_line": seconds, "scored": scored, "pass": bool(clicked and recording and ready and scored),
            "note": "fake capture device fed by a synthetic tone; this is not real speech and not a real microphone"}


def launch(extra, port):
    browser = next(b for b in BROWSERS if Path(b).exists())
    profile = tempfile.mkdtemp(prefix="trustmesh_selftest_")
    process = subprocess.Popen([browser, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--remote-debugging-port={port}",
                                f"--user-data-dir={profile}", *extra, "about:blank"])
    for _ in range(40):
        try:
            tabs = requests.get(f"http://127.0.0.1:{port}/json", timeout=1).json()
            return process, next(t for t in tabs if t["type"] == "page")["webSocketDebuggerUrl"]
        except requests.RequestException:
            time.sleep(0.5)
    raise RuntimeError("browser did not start")


async def run(base):
    OUT.mkdir(parents=True, exist_ok=True)
    t = np.arange(6 * 48000) / 48000
    sf.write(OUT / "synthetic_tone.wav", (0.3 * np.sin(2 * np.pi * 170 * t) * (1 + 0.6 * np.sin(2 * np.pi * 4 * t))).astype(np.float32), 48000, subtype="PCM_16")
    result = {}
    process, ws_url = launch([], 9334)
    try:
        async with websockets.connect(ws_url, max_size=None) as ws:
            page = Page(ws)
            await page.send("Page.enable")
            await page.send("DOM.enable")
            result["keyboard"] = await keyboard_checks(page, base)
            result["reduced_motion"] = await reduced_motion(page, base)
            result["axe"] = await axe(page, base)
            result["overflow"] = await overflow(page, base)
    finally:
        process.terminate()
    process, ws_url = launch(["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
                              f"--use-file-for-fake-audio-capture={OUT / 'synthetic_tone.wav'}"], 9335)
    try:
        async with websockets.connect(ws_url, max_size=None) as ws:
            page = Page(ws)
            await page.send("Page.enable")
            result["microphone"] = await microphone(page, base)
    finally:
        process.terminate()
    (OUT / "browser.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://localhost:8080")
    asyncio.run(run(parser.parse_args().base))
