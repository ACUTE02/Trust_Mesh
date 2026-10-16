"""Browser checks for the six worlds (Revision 2), with headless Edge/Chrome driven over the DevTools protocol.

Needs the production build served (for example `npx vite preview --port 4173` in frontend/) and axe-core in
frontend/node_modules. WebGL runs on SwiftShader (software), so frame rates here are NOT real GPU speed: they are
only useful as a comparison between two worlds or two builds. Counts (draw calls, triangles, geometries) are exact.

  python scripts/check_worlds_browser.py --base http://localhost:4173 --what console,keyboard,axe,overflow,fallbacks,tiers,throttled,leaks,screens

Results go to data/self_test/worlds_browser.json (git-ignored). Screenshots go to --shots (outside the repository).
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

import websockets

sys.path.insert(0, str(Path(__file__).resolve().parent))
import self_test_browser as stb  # noqa: E402

ROUTES = {"home": "/", "graph": "/graph", "voice": "/voice", "case": "/case", "evidence": "/evidence", "about": "/about", "lost": "/not-a-page"}
NAV_LABEL = {"home": "Overview", "graph": "Graph Engine", "voice": "Voice Engine", "case": "Case desk", "evidence": "Evidence", "about": "About"}
GL_FLAGS = ["--enable-unsafe-swiftshader", "--use-angle=swiftshader", "--ignore-gpu-blocklist"]
OUT = stb.OUT


class Rec(stb.Page):
    """A Page that keeps the events (console messages, exceptions) it sees while waiting for answers."""

    def __init__(self, ws):
        super().__init__(ws)
        self.events = []

    async def send(self, method, **params):
        self.n += 1
        await self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            message = json.loads(await self.ws.recv())
            if message.get("id") == self.n:
                return message.get("result", {})
            self.events.append(message)

    async def drain(self, seconds=0.2):
        """Give the browser a moment, collecting events."""
        try:
            while True:
                self.events.append(json.loads(await asyncio.wait_for(self.ws.recv(), timeout=seconds)))
        except asyncio.TimeoutError:
            return


async def new_page(ws, width=1440, height=900, mobile=False, scheme=None):
    page = Rec(ws)
    for domain in ("Page", "Runtime", "Log", "Network"):
        await page.send(f"{domain}.enable")
    await page.send("Emulation.setDeviceMetricsOverride", width=width, height=height, deviceScaleFactor=1, mobile=mobile)
    if scheme:
        await page.send("Emulation.setEmulatedMedia", features=[{"name": "prefers-color-scheme", "value": scheme}])
    return page


async def with_browser(fn, port=9360, flags=None):
    process, ws_url = stb.launch(flags if flags is not None else GL_FLAGS, port)
    try:
        async with websockets.connect(ws_url, max_size=None) as ws:
            return await fn(ws)
    finally:
        process.kill()


# --- console errors --------------------------------------------------------------------------------------------
async def console_errors(base):
    async def run(ws):
        page = await new_page(ws)
        found = {}
        for name, path in ROUTES.items():
            page.events.clear()
            await page.goto(base + path, wait=0)
            await asyncio.sleep(4.5)
            await page.drain(0.3)
            errors = []
            for e in page.events:
                if e.get("method") == "Runtime.exceptionThrown":
                    errors.append(e["params"]["exceptionDetails"].get("text", "exception") + ": " + str(e["params"]["exceptionDetails"].get("exception", {}).get("description", ""))[:200])
                elif e.get("method") == "Runtime.consoleAPICalled" and e["params"]["type"] in ("error", "assert"):
                    errors.append(" ".join(str(a.get("value", a.get("description", ""))) for a in e["params"]["args"])[:200])
                elif e.get("method") == "Log.entryAdded" and e["params"]["entry"]["level"] == "error":
                    errors.append(e["params"]["entry"]["text"][:200])
            found[name] = errors
        return {"errors_by_route": found, "pass": not any(found.values())}
    return await with_browser(run)


console = console_errors


# --- keyboard, axe, overflow (the repository's own checks, now over every route) ---------------------------------
async def keyboard(base):
    async def run(ws):
        page = await new_page(ws, 1280, 900)
        return await stb.keyboard_checks(page, base)
    return await with_browser(run)


async def axe(base):
    async def run(ws):
        page = await new_page(ws, 1280, 900)
        return await stb.axe(page, base)
    return await with_browser(run)


async def overflow(base):
    async def run(ws):
        page = await new_page(ws, 1280, 900)
        return await stb.overflow(page, base)
    return await with_browser(run)


# --- keyboard by identity: every interactive element of every route receives focus from the Tab key ----------------
TAG_ALL = """(() => { const els = [...document.querySelectorAll('a[href], button, input, select, textarea, [tabindex]')]
  .filter(e => e.tabIndex >= 0 && e.getClientRects().length > 0 && !e.disabled);
  els.forEach((e, i) => e.dataset.kb = String(i)); window.__kbSeen = new Set(); window.__kbTotal = els.length;
  document.addEventListener('focusin', ev => { if (ev.target.dataset && ev.target.dataset.kb) window.__kbSeen.add(ev.target.dataset.kb); });
  return els.length })()"""


async def keyboard_identity(base):
    async def run(ws):
        page = await new_page(ws, 1280, 900)
        out = {}
        for name, path in ROUTES.items():
            for layer in ("0", "1"):
                await page.goto(base + path, wait=0)
                await page.js("sessionStorage.setItem('tm-engineers-%s', '%s'); 1" % ({"home": "landing", "lost": "x"}.get(name, name), layer))
                await page.goto(base + path, wait=3)
                total = await page.js(TAG_ALL)
                for _ in range(total + 12):
                    await page.key("Tab", "Tab", 9)
                seen = await page.js("window.__kbSeen.size")
                missed = await page.js("[...document.querySelectorAll('[data-kb]')].filter(e => !window.__kbSeen.has(e.dataset.kb)).map(e => (e.tagName + ':' + (e.type||'') + ':' + (e.getAttribute('aria-label')||e.innerText||e.name||'').trim().slice(0,30)))")
                radios = [m for m in missed if m.startswith("INPUT:radio")]  # a radio group is one tab stop
                files = [m for m in missed if m.startswith("INPUT:file")]    # reached through its visible button
                real = [m for m in missed if m not in radios and m not in files]
                out[f"{name}/{'engineers' if layer == '1' else 'plain'}"] = {"interactive": total, "focused": seen, "not_focused": real, "pass": not real}
        out["pass"] = all(v["pass"] for v in out.values())
        return out
    return await with_browser(run, 9370)


# --- fallbacks: reduced motion, data saving, no WebGL, lost context ---------------------------------------------
async def fallbacks(base):
    async def reduced(ws):
        page = await new_page(ws)
        await page.send("Emulation.setEmulatedMedia", features=[{"name": "prefers-reduced-motion", "value": "reduce"}])
        await page.goto(base + "/graph?perf=1", wait=5)
        a = await page.js("JSON.stringify({canvas: !!document.querySelector('#stage canvas'), poster: !!document.querySelector('[data-poster]')})")
        await asyncio.sleep(3)
        await page.js("window.__before = window.__tmPerf ? window.__tmPerf.fps : null; 1")
        await page.js("[...document.querySelectorAll('nav[aria-label=Pages] a')].find(a => a.textContent.trim()==='Voice Engine').click(); 1")
        await asyncio.sleep(1.2)
        b = await page.js("JSON.stringify({curtain: !!document.querySelector('.curtain'), world: document.documentElement.dataset.world, h1: document.activeElement && document.activeElement.tagName})")
        idle_fps = await page.js("(async()=>{const t0=performance.now();let n=0;const s=window.__tmPerf?window.__tmPerf.fps:0;await new Promise(r=>setTimeout(r,2500));return window.__tmPerf?window.__tmPerf.fps:null})()", await_promise=True)
        return {"static_pose_drawn": json.loads(a), "after_route_change": json.loads(b), "fps_when_idle": idle_fps}

    async def saver(ws):
        page = await new_page(ws)
        await page.send("Page.addScriptToEvaluateOnNewDocument", source="Object.defineProperty(navigator,'connection',{get:()=>({saveData:true})});")
        await page.goto(base + "/voice", wait=3)
        return json.loads(await page.js("JSON.stringify({canvas: !!document.querySelector('#stage canvas'), poster: !!document.querySelector('[data-poster]'), text: document.querySelector('h1') && document.querySelector('h1').innerText.slice(0,40)})"))

    async def nowebgl(ws):
        page = await new_page(ws)
        await page.send("Page.addScriptToEvaluateOnNewDocument", source="HTMLCanvasElement.prototype.getContext=function(){return null};")
        out = {}
        for name in ROUTES:
            await page.goto(base + ROUTES[name], wait=2.5)
            out[name] = json.loads(await page.js("JSON.stringify({canvas: !!document.querySelector('#stage canvas'), poster: !!document.querySelector('[data-poster]'), h1: !!document.querySelector('h1')})"))
        return out

    async def lost(ws):
        page = await new_page(ws)
        await page.goto(base + "/graph?perf=1", wait=5)
        before = await page.js("!!document.querySelector('#stage canvas')")
        await page.js("(()=>{const c=document.querySelector('#stage canvas');const gl=c.getContext('webgl2');const e=gl.getExtension('WEBGL_lose_context');e.loseContext();return 1})()")
        await asyncio.sleep(2)
        return {"canvas_before": before, "poster_after_loss": await page.js("!!document.querySelector('[data-poster]')"), "text_still_there": await page.js("!!document.querySelector('h1')")}

    r = {"reduced_motion": await with_browser(reduced, 9361), "save_data": await with_browser(saver, 9362),
         "no_webgl": await with_browser(nowebgl, 9363, flags=["--disable-gpu", "--disable-webgl", "--disable-3d-apis"]),
         "context_lost": await with_browser(lost, 9364)}
    r["pass"] = bool(r["reduced_motion"]["static_pose_drawn"]["canvas"] and not r["reduced_motion"]["after_route_change"]["curtain"]
                     and r["save_data"]["poster"] and not r["save_data"]["canvas"]
                     and all(v["poster"] and not v["canvas"] and v["h1"] for v in r["no_webgl"].values())
                     and r["context_lost"]["poster_after_loss"])
    return r


# --- the tiers: exact draw calls, triangles, geometries per world and tier ---------------------------------------
async def tiers(base):
    async def run(ws):
        page = await new_page(ws)
        table = {}
        for tier in ("low", "medium", "high"):
            for name, path in ROUTES.items():
                await page.goto(f"{base}{path}?perf=1&tier={tier}", wait=0)
                await asyncio.sleep(5.0)
                snap = json.loads(await page.js("JSON.stringify(window.__tmPerf || null)") or "null")
                table.setdefault(name, {})[tier] = snap and {k: snap[k] for k in ("calls", "triangles", "geometries", "textures", "frameMs", "fps", "tier")}
        return table
    return await with_browser(run, 9365)


# --- governor: which tier a world settles on when left alone ------------------------------------------------------
async def governor(base):
    async def run(ws):
        page = await new_page(ws)
        out = {}
        for name in ("home", "graph", "voice"):
            await page.goto(f"{base}{ROUTES[name]}?perf=1", wait=0)
            await asyncio.sleep(14)
            out[name] = await page.js("window.__tmPerf && window.__tmPerf.tier")
        return out
    return await with_browser(run, 9366)


# --- throttled run: 4x CPU, slow network, mobile size; long tasks per route change; LCP and layout shift ---------------
OBSERVE = """
window.__long = []; window.__cls = 0; window.__lcp = 0;
try { new PerformanceObserver(l => l.getEntries().forEach(e => window.__long.push([Math.round(e.startTime), Math.round(e.duration)]))).observe({type:'longtask', buffered:true}); } catch(e) {}
try { new PerformanceObserver(l => l.getEntries().forEach(e => { if (!e.hadRecentInput) window.__cls += e.value; })).observe({type:'layout-shift', buffered:true}); } catch(e) {}
try { new PerformanceObserver(l => l.getEntries().forEach(e => window.__lcp = Math.round(e.startTime))).observe({type:'largest-contentful-paint', buffered:true}); } catch(e) {}
"""


async def throttled(base):
    async def run(ws):
        page = await new_page(ws, 390, 800, mobile=True)
        await page.send("Emulation.setCPUThrottlingRate", rate=4)
        await page.send("Network.emulateNetworkConditions", offline=False, latency=150, downloadThroughput=int(1.6 * 1024 * 1024 / 8), uploadThroughput=int(750 * 1024 / 8))
        await page.send("Page.addScriptToEvaluateOnNewDocument", source=OBSERVE)
        await page.goto(base + "/?perf=1", wait=0)
        await asyncio.sleep(12)
        first = json.loads(await page.js("JSON.stringify({lcp: window.__lcp, cls: window.__cls, long: window.__long})"))
        report = {"first_load": first, "route_changes": {}}
        for name in ("graph", "voice", "case", "evidence", "about", "home"):
            await page.js("window.__long.length = 0; 1")
            await page.js("(()=>{const l=[...document.querySelectorAll('#mobile-menu a, nav a')].find(a=>a.textContent.trim()===%s);if(l){l.click();return 1}const b=[...document.querySelectorAll('header button')].find(b=>b.textContent.trim()==='Menu');b&&b.click();return 0})()" % json.dumps(NAV_LABEL[name]))
            await asyncio.sleep(0.4)
            await page.js("(()=>{const l=[...document.querySelectorAll('#mobile-menu a')].find(a=>a.textContent.trim()===%s);l&&l.click();return 1})()" % json.dumps(NAV_LABEL[name]))
            await asyncio.sleep(3.5)
            report["route_changes"][name] = {"world": await page.js("document.documentElement.dataset.world"), "long_tasks_ms": await page.js("window.__long.map(x=>x[1])")}
        report["final_cls"] = await page.js("window.__cls")
        # input to visible response: a click on the theme switch until the next frame after it is painted
        report["theme_toggle_response_ms"] = await page.js("(async()=>{const b=[...document.querySelectorAll('button')].find(b=>/Switch to/.test(b.textContent));const t=performance.now();b.click();await new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)));return Math.round(performance.now()-t)})()", await_promise=True)
        return report
    return await with_browser(run, 9367)


# --- memory: all six worlds twice, then home again; geometry and texture counts return to the baseline -----------------
async def leaks(base):
    async def run(ws):
        page = await new_page(ws)
        await page.goto(base + "/?perf=1&tier=low", wait=0)
        await asyncio.sleep(5)
        snap = lambda: page.js("JSON.stringify({geometries: window.__tmPerf.geometries, textures: window.__tmPerf.textures, calls: window.__tmPerf.calls})")
        baseline = json.loads(await snap())
        visits = []
        for _ in range(2):
            for name in ("graph", "voice", "case", "evidence", "about", "home"):
                await page.js("[...document.querySelectorAll('nav[aria-label=Pages] a')].find(a => a.textContent.trim()===%s).click(); 1" % json.dumps(NAV_LABEL[name]))
                await asyncio.sleep(4)
                visits.append((name, json.loads(await snap())))
        after = json.loads(await snap())
        heap = await page.js("performance.memory ? Math.round(performance.memory.usedJSHeapSize/1048576) : null")
        return {"baseline_home": baseline, "after_two_rounds_home": after, "visits": visits, "js_heap_mb_after": heap,
                "pass": baseline["geometries"] == after["geometries"] and baseline["textures"] == after["textures"]}
    return await with_browser(run, 9368)


# --- screenshots ----------------------------------------------------------------------------------------------------
SIZES = {"1440": (1440, 900), "768": (768, 1024), "390": (390, 800)}


async def shoot(page, path):
    shot = await page.send("Page.captureScreenshot", format="png")
    import base64
    Path(path).write_bytes(base64.b64decode(shot["data"]))


async def screens(base, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    async def run(ws):
        page = await new_page(ws)
        for size, (w, h) in SIZES.items():
            await page.send("Emulation.setDeviceMetricsOverride", width=w, height=h, deviceScaleFactor=1, mobile=w < 600)
            for mode in ("light", "dark", "reduced"):
                scheme = "light" if mode == "light" else "dark"
                await page.send("Emulation.setEmulatedMedia", features=[{"name": "prefers-color-scheme", "value": scheme},
                                                                         {"name": "prefers-reduced-motion", "value": "reduce" if mode == "reduced" else "no-preference"}])
                for name, path in ROUTES.items():
                    await page.goto(base + path, wait=0)
                    await page.js("localStorage.removeItem('tm-theme'); 1")
                    await page.goto(base + path, wait=0)
                    await asyncio.sleep(5.5)
                    await shoot(page, out / f"{name}_{size}_{mode}_top.png")
                    mid = await page.js("(()=>{const c=[...document.querySelectorAll('[data-stage]')].filter(e=>+e.dataset.stage>=1)[0];if(!c)return null;window.scrollTo(0,c.getBoundingClientRect().top+scrollY-60);return 1})()")
                    if mid:
                        await asyncio.sleep(5.5)
                        await shoot(page, out / f"{name}_{size}_{mode}_mid.png")
        return sorted(p.name for p in out.glob("*.png"))
    return await with_browser(run, 9369)


def contact_sheet(shots, out_file):
    from PIL import Image
    shots = Path(shots)
    cols = [("light", "top"), ("dark", "top"), ("light", "mid"), ("dark", "mid")]
    names = [n for n in ROUTES]
    tw, th = 360, 225
    sheet = Image.new("RGB", (tw * len(cols) + 5 * (len(cols) + 1), th * len(names) + 5 * (len(names) + 1)), (128, 128, 128))
    for r, name in enumerate(names):
        for c, (mode, part) in enumerate(cols):
            f = shots / f"{name}_1440_{mode}_{part}.png"
            if f.exists():
                im = Image.open(f).convert("RGB").resize((tw, th))
                sheet.paste(im, (5 + c * (tw + 5), 5 + r * (th + 5)))
    sheet.save(out_file)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://localhost:4173")
    parser.add_argument("--what", default="console,keyboard,axe,overflow,fallbacks,tiers,governor,throttled,leaks")
    parser.add_argument("--shots", default=r"E:\TrustMesh\design_research\shots")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    file = OUT / "worlds_browser.json"
    results = json.loads(file.read_text()) if file.exists() else {}
    for key in [k for k in args.what.split(",") if k]:
        print("running", key, flush=True)
        if key == "screens":
            results[key] = await screens(args.base, args.shots)
            contact_sheet(args.shots, Path(args.shots) / "contact_sheet.png")
        else:
            results[key] = await globals()[key](args.base)
        file.write_text(json.dumps(results, indent=1))
        print(json.dumps(results[key], indent=1)[:1800], flush=True)


if __name__ == "__main__":
    asyncio.run(main())
