"""Live checks of the worlds against the running stack (the API and the site), driven over the DevTools protocol.

1. voice : your recording becomes a terrain before any upload; the guard's two outcomes reach the scene
           (a tone is refused and the terrain collapses; band-limited speech is scored with a warning and fog).
2. recording : with TRUSTMESH_RECORDING_MODE=1 on the API, no real transaction number, label or score reaches any
           3D scene or the page (case balance, neighbourhood beads, register).

Usage (stack up, site on 8080 or the preview on 4173 with the API proxied):
  python scripts/check_worlds_live.py --base http://localhost:8080 --what voice
  TRUSTMESH_RECORDING_MODE=1 docker compose up -d api   (wait for health), then:
  python scripts/check_worlds_live.py --base http://localhost:8080 --what recording
The scene data is readable only with ?perf=1 (window.__tmLive). Writes data/self_test/worlds_live.json (git-ignored).
"""
import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import check_worlds_browser as cwb  # noqa: E402
import audio_fixtures as fx  # noqa: E402

EXAMPLE_IDS = ("154134", "146797", "151474")


async def set_file(page, path):
    doc = await page.send("DOM.getDocument", depth=1)
    node = await page.send("DOM.querySelector", nodeId=doc["root"]["nodeId"], selector="input[type=file]")
    await page.send("DOM.setFileInputFiles", files=[str(path)], nodeId=node["nodeId"])


async def voice(base):
    tmp = Path(tempfile.mkdtemp(prefix="tm_worlds_"))
    sf.write(tmp / "tone.wav", fx.tone(), fx.SR, subtype="PCM_16")
    sf.write(tmp / "speech.wav", fx.speech_like(), fx.SR, subtype="PCM_16")
    sf.write(tmp / "phone.wav", fx.telephone_band(fx.speech_like()), fx.SR, subtype="PCM_16")

    async def run(ws):
        page = await cwb.new_page(ws)
        out = {}
        for name, expect in (("tone", "notSpeech"), ("speech", "scored"), ("phone", "lowQuality")):
            await page.goto(base + "/voice?perf=1", wait=0)
            await asyncio.sleep(5)
            await set_file(page, tmp / f"{name}.wav")
            ready = await page.wait_for("!!document.querySelector('[data-testid=clip-ready]')", 20)
            sample_in_scene = await page.js("!!(window.__tmLive && window.__tmLive.data.voice && window.__tmLive.data.voice.samples)")
            await page.js("[...document.querySelectorAll('button')].find(b => b.textContent.trim() === 'Score this clip').click(); 1")
            await asyncio.sleep(1)
            done = await page.wait_for("!!document.querySelector('[data-testid=voice-result]') || !!document.querySelector('[role=alert]')", 90)
            await asyncio.sleep(1.5)
            data = json.loads(await page.js("JSON.stringify((window.__tmLive && window.__tmLive.data.voice) ? {verdict: window.__tmLive.data.voice.verdict, notSpeech: window.__tmLive.data.voice.notSpeech, lowQuality: window.__tmLive.data.voice.lowQuality, cutoffHz: window.__tmLive.data.voice.cutoffHz, hasSamples: !!window.__tmLive.data.voice.samples} : null)"))
            alert = await page.js("(document.querySelector('[role=alert]')||{}).innerText || null")
            out[name] = {"clip_ready": ready, "terrain_samples_given_before_upload": sample_in_scene, "answered": done, "scene": data, "page_message": alert and alert[:140]}
        out["pass"] = bool(out["tone"]["scene"]["notSpeech"] and out["speech"]["scene"]["verdict"] and not out["speech"]["scene"]["notSpeech"]
                           and out["phone"]["scene"]["lowQuality"] and all(out[k]["terrain_samples_given_before_upload"] for k in ("tone", "speech", "phone")))
        return out
    return await cwb.with_browser(run, 9380)


async def recording(base):
    async def run(ws):
        page = await cwb.new_page(ws)
        out = {}
        await page.goto(base + "/case?perf=1", wait=0)
        await asyncio.sleep(5)
        await page.js("(async()=>{await new Promise(r=>setTimeout(r,1500));[...document.querySelectorAll('button')].find(b => b.textContent.trim()==='Score this case').click();return 1})()", await_promise=True)
        await page.wait_for("!!document.querySelector('[data-testid=case-card]')", 60)
        await asyncio.sleep(2)
        case = json.loads(await page.js("JSON.stringify(window.__tmLive.data.case || null)"))
        text = await page.js("document.body.innerText")
        html = await page.js("document.documentElement.outerHTML")
        out["case"] = {"scene_data": case, "banner": "Recording mode is on" in text, "register": [l for l in text.splitlines() if "Details hidden" in l][:3],
                       "example_ids_in_page_text": [i for i in EXAMPLE_IDS if i in text], "pass": bool(case and case["graph"] is None and case["masked"] and "Recording mode is on" in text and not any(i in text for i in EXAMPLE_IDS))}
        await page.goto(base + "/graph?perf=1", wait=0)
        await asyncio.sleep(5)
        await page.js("[...document.querySelectorAll('button')].find(b => /^example 1$/.test(b.textContent.trim())).click(); 1")
        await page.wait_for("!!document.querySelector('[data-testid=graph-result]')", 60)
        await asyncio.sleep(2)
        graph = json.loads(await page.js("JSON.stringify(window.__tmLive.data.graph ? {centre: window.__tmLive.data.graph.centre, neighbours: window.__tmLive.data.graph.neighbours.length, flagged: window.__tmLive.data.graph.neighbours.filter(n => n.flagged).length} : null)"))
        text = await page.js("document.body.innerText")
        out["graph"] = {"scene_data": graph, "example_ids_in_page_text": [i for i in EXAMPLE_IDS if i in text],
                        "pass": bool(graph and graph["centre"] and not graph["centre"]["flagged"] and graph["flagged"] == 0 and not any(i in text for i in EXAMPLE_IDS))}
        out["pass"] = out["case"]["pass"] and out["graph"]["pass"]
        return out
    return await cwb.with_browser(run, 9381)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://localhost:8080")
    parser.add_argument("--what", default="voice")
    args = parser.parse_args()
    file = cwb.OUT / "worlds_live.json"
    results = json.loads(file.read_text()) if file.exists() else {}
    for key in args.what.split(","):
        results[key] = await globals()[key](args.base)
        file.write_text(json.dumps(results, indent=1))
        print(json.dumps(results[key], indent=1)[:2500], flush=True)


if __name__ == "__main__":
    asyncio.run(main())
