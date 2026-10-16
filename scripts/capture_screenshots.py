"""Full-page screenshots of every page at 390, 768 and 1440 px in both themes, with headless Edge/Chrome
driven over the DevTools protocol (exact viewport widths and an emulated colour scheme).

Only the empty state of each page is captured: no case is scored, so the images hold aggregate results
and no per-sample prediction or dataset content.

Run (stack up): .venv\\Scripts\\python.exe scripts/capture_screenshots.py [--base http://localhost:8080]
                [--out docs/frontend_screenshots]
Needs the `websockets` and `requests` packages (already in the dev environment) and Edge or Chrome.
"""
import argparse
import asyncio
import base64
import json
import subprocess
import tempfile
import time
from pathlib import Path

import requests
import websockets

ROOT = Path(__file__).resolve().parents[1]
BROWSERS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe"]
PAGES = {"landing": "/", "graph": "/graph", "voice": "/voice", "case": "/case", "evidence": "/evidence", "about": "/about", "lost": "/not-a-page"}
WIDTHS = (390, 768, 1440)
PORT = 9333


async def capture(ws_url, base, out):
    async with websockets.connect(ws_url, max_size=None) as ws:
        counter = 0

        async def send(method, **params):
            nonlocal counter
            counter += 1
            await ws.send(json.dumps({"id": counter, "method": method, "params": params}))
            while True:
                message = json.loads(await ws.recv())
                if message.get("id") == counter:
                    return message.get("result", {})

        await send("Page.enable")
        for theme in ("light", "dark"):
            await send("Emulation.setEmulatedMedia", features=[{"name": "prefers-color-scheme", "value": theme}])
            for width in WIDTHS:
                await send("Emulation.setDeviceMetricsOverride", width=width, height=900, deviceScaleFactor=1, mobile=False)
                for name, path in PAGES.items():
                    await send("Page.navigate", url=base + path)
                    await asyncio.sleep(2.5)  # fonts, /health, and the instrument's one load animation
                    overflow = await send("Runtime.evaluate", returnByValue=True, expression=(
                        "document.documentElement.scrollWidth - document.documentElement.clientWidth"))
                    size = (await send("Page.getLayoutMetrics"))["cssContentSize"]
                    shot = await send("Page.captureScreenshot", format="png", captureBeyondViewport=True,
                                      clip={"x": 0, "y": 0, "width": width, "height": size["height"], "scale": 1})
                    target = out / f"{name}_{width}_{theme}.png"
                    target.write_bytes(base64.b64decode(shot["data"]))
                    print(f"{target.name}: {width}x{int(size['height'])}, horizontal overflow {overflow['result']['value']} px")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://localhost:8080")
    parser.add_argument("--out", default="docs/frontend_screenshots")
    args = parser.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    browser = next(b for b in BROWSERS if Path(b).exists())
    profile = tempfile.mkdtemp(prefix="trustmesh_shots_")
    process = subprocess.Popen([browser, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                                f"--remote-debugging-port={PORT}", f"--user-data-dir={profile}", "about:blank"])
    try:
        for _ in range(40):
            try:
                tabs = requests.get(f"http://127.0.0.1:{PORT}/json", timeout=1).json()
                break
            except requests.RequestException:
                time.sleep(0.5)
        page = next(t for t in tabs if t["type"] == "page")
        asyncio.run(capture(page["webSocketDebuggerUrl"], args.base, out))
    finally:
        process.terminate()


if __name__ == "__main__":
    main()
