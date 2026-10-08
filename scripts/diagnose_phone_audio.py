"""Why do genuine phone recordings score as fake? Diagnosis only: no training, no model change, no threshold change.

Scores audio in this process with the SAME code and weights the service uses (Model D head on the frozen
XLS-R encoder, `serving.voice_service`), but exposes the steps the service hides: level, trimming, resampling,
codec, noise, reverb. Prints AGGREGATES only. Per-clip detail from the owner's own recordings is written next to
the recordings (outside the repository), never into the repository.

Checks (numbers as in the owner's request):
  1  level, rate, duration, trimming of each clip           5  distribution comparison (aggregates)
  2  the clips in several versions                          4  codec / level / noise / resampling / reverb on dataset clips
  3  browser microphone path (headless Edge, fake capture device playing the owner's real clip)

Run: .venv\\Scripts\\python.exe scripts/diagnose_phone_audio.py "E:\\TrustMesh\\my_voice_tests"
Needs the stack NOT to be running Docker (it only needs the weights on disk) and, for check 3, frontend/dist built.
"""
import asyncio
import base64
import io
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import fftconvolve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from engines.voice.features import quantize  # noqa: E402
from engines.voice.preprocess import fix_length, frame_energy_db, resample_linear, trim_silence  # noqa: E402
from serving.voice_service import VoiceService  # noqa: E402

SR = 16000
CLIP_SAMPLES = 4 * SR
FF = None


def ffmpeg(*args):
    done = subprocess.run([FF, "-y", "-loglevel", "error", *map(str, args)], capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(done.stderr.strip()[:300])


def native_info(path):
    out = subprocess.run([FF, "-hide_banner", "-i", str(path)], capture_output=True, text=True).stderr
    line = next(l for l in out.splitlines() if "Audio:" in l)
    parts = [p.strip() for p in line.split("Audio:")[1].split(",")]
    kbps = next((p for p in parts if "kb/s" in p), "")
    return {"codec": parts[0].split(" (")[0], "rate_hz": int(parts[1].split()[0]), "bitrate_kbps": float(kbps.split()[0]) if kbps else None}


def load16(path):
    """Proper (anti-aliased) conversion to 16 kHz mono float, as ffmpeg does it."""
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "x.wav"
        ffmpeg("-i", path, "-ac", 1, "-ar", SR, "-c:a", "pcm_s16le", out)
        w, sr = sf.read(out, dtype="float32")
    assert sr == SR
    return w


def load_native(path):
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "x.wav"
        ffmpeg("-i", path, "-ac", 1, "-c:a", "pcm_s16le", out)
        return sf.read(out, dtype="float32")


def db(x):
    return 20 * np.log10(max(float(x), 1e-9))


def rms(w):
    return float(np.sqrt(np.mean(w.astype(np.float64) ** 2) + 1e-18))


class Scorer:
    """The service's scoring path with the steps exposed. `base()` must equal VoiceService.score_waveform."""

    def __init__(self):
        self.svc = VoiceService()
        self.svc.load()
        self.head = self.svc.heads["D"]
        self.thr = self.svc.thresholds["D"]
        self.feats = {}

    def prepare(self, wave, top_db=30.0, mode="trim_start"):
        if mode == "none_start":
            seg = wave
        elif mode == "none_middle":
            mid = len(wave) // 2
            seg = wave[max(0, mid - CLIP_SAMPLES // 2): mid + CLIP_SAMPLES // 2]
        else:
            seg, _ = trim_silence(wave, SR, top_db)
        return quantize(fix_length(seg, CLIP_SAMPLES))

    @torch.no_grad()
    def score(self, wave, top_db=30.0, mode="trim_start", keep=None):
        clip = self.prepare(wave, top_db, mode)
        pooled = self.svc.encoder.pooled(torch.from_numpy(clip).unsqueeze(0))
        feats = self.svc.encoder.mean_only(pooled)
        logit = float(self.head(feats)[0])
        if keep is not None:
            w = torch.softmax(self.head.layer_logits.detach(), 0).view(1, -1, 1)
            self.feats[keep] = (w * feats).sum(1)[0].numpy()
        p = float(1 / (1 + np.exp(logit)))
        return {"logit": logit, "p": p, "band": "high" if p >= self.thr["high"] else "medium" if p >= self.thr["medium"] else "low"}


def describe(wave16, native_rate, label_extra=None):
    """Check 1: level, rate, duration, trimming."""
    d = {"native_rate_hz": native_rate, "duration_s": len(wave16) / SR, "peak_dbfs": db(np.max(np.abs(wave16))), "rms_dbfs": db(rms(wave16))}
    frames, _, _ = frame_energy_db(wave16, SR)
    d["noise_floor_dbfs"] = float(np.percentile(frames, 10))
    d["loud_frames_dbfs"] = float(np.percentile(frames, 90))
    d["dynamic_range_db"] = d["loud_frames_dbfs"] - d["noise_floor_dbfs"]
    for top_db in (30.0,):
        trimmed, st = trim_silence(wave16, SR, top_db)
        d["after_trim_s"] = len(trimmed) / SR
        d["kept_fraction_of_duration"] = len(trimmed) / len(wave16)
        d["kept_fraction_of_energy"] = float(np.sum(trimmed.astype(np.float64) ** 2) / (np.sum(wave16.astype(np.float64) ** 2) + 1e-18))
        d["lead_silence_s"] = st["lead_silence_s"]
        d["trail_silence_s"] = st["trail_silence_s"]
        d["trimmed_rms_dbfs"] = db(rms(trimmed))
    spec = np.abs(np.fft.rfft(wave16.astype(np.float64))) ** 2
    freqs = np.fft.rfftfreq(len(wave16), 1 / SR)
    d["energy_above_4khz_fraction"] = float(spec[freqs >= 4000].sum() / (spec.sum() + 1e-18))
    d["clipped_samples_fraction"] = float(np.mean(np.abs(wave16) >= 0.999))
    return d


def reencode(src_wav, codec, via48, tmp):
    """Round trip a 16 kHz WAV through a codec (and optionally through 48 kHz, like a phone chain)."""
    stem = f"{src_wav.stem}_{codec}_{'48' if via48 else '16'}"
    work = src_wav
    if via48:
        work = tmp / f"{stem}_up.wav"
        ffmpeg("-i", src_wav, "-ar", 48000, "-c:a", "pcm_s16le", work)
    if codec == "resample_only":
        coded = work
    else:
        enc = {"mp3_128": (".mp3", ["-c:a", "libmp3lame", "-b:a", "128k"]), "aac_128": (".m4a", ["-c:a", "aac", "-b:a", "128k"]),
               "aac_32": (".m4a", ["-c:a", "aac", "-b:a", "32k"]), "opus_64": (".opus", ["-c:a", "libopus", "-b:a", "64k"]),
               "opus_24": (".opus", ["-c:a", "libopus", "-b:a", "24k"])}[codec]
        coded = tmp / f"{stem}{enc[0]}"
        ffmpeg("-i", work, *enc[1], coded)
    back = tmp / f"{stem}_back.wav"
    ffmpeg("-i", coded, "-ac", 1, "-ar", SR, "-c:a", "pcm_s16le", back)
    return sf.read(back, dtype="float32")[0]


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def pct(values, flt):
    values = np.asarray(values)
    return float(np.mean(flt(values))) * 100


# ---------------------------------------------------------------------------------------------------
def browser_capture(folder, work):
    """Check 3: push the owner's clips through the site's own browser pipeline. Returns {name: wav bytes}."""
    import requests
    import websockets
    from self_test_browser import BROWSERS, Page

    dist = ROOT / "frontend" / "dist"
    if not dist.exists():
        return None, "frontend/dist not built"
    import functools
    import http.server
    import threading

    class SpaHandler(http.server.SimpleHTTPRequestHandler):
        """Static files, with unknown paths (client-side routes such as /voice) answered by index.html."""
        def send_head(self):
            path = self.translate_path(self.path)
            if not Path(path).exists() and "." not in Path(self.path).name:
                self.path = "/index.html"
            return super().send_head()

        def log_message(self, *a):
            pass

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 5199), functools.partial(SpaHandler, directory=str(dist)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = "http://127.0.0.1:5199"
    out = {}
    INSTALL = """window.__up = null; const of = window.fetch.bind(window);
      window.fetch = async (u, init) => { if (String(u).includes('/voice/score')) { const f = init.body.get('file');
        const b = new Uint8Array(await f.arrayBuffer()); let s = ''; for (let i = 0; i < b.length; i += 8192) s += String.fromCharCode.apply(null, b.subarray(i, i + 8192));
        window.__up = { name: f.name, size: b.length, b64: btoa(s) };
        return new Response(JSON.stringify({ detail: 'blocked by the test: nothing was scored by the service' }), { status: 500, headers: { 'content-type': 'application/json' } }); }
        return of(u, init); }; 'ok'"""

    async def one_session(capture_wav, jobs, port):
        browser = next(b for b in BROWSERS if Path(b).exists())
        profile = tempfile.mkdtemp(prefix="tm_phone_")
        proc = subprocess.Popen([browser, "--headless=new", "--disable-gpu", f"--remote-debugging-port={port}", f"--user-data-dir={profile}",
                                 "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
                                 f"--use-file-for-fake-audio-capture={capture_wav}", "about:blank"])
        try:
            for _ in range(40):
                try:
                    tabs = requests.get(f"http://127.0.0.1:{port}/json", timeout=1).json()
                    break
                except requests.RequestException:
                    time.sleep(0.5)
            async with websockets.connect(next(t for t in tabs if t["type"] == "page")["webSocketDebuggerUrl"], max_size=None) as ws:
                page = Page(ws)
                await page.send("Page.enable")
                await page.send("DOM.enable")
                for key, kind, path, seconds in jobs:
                    await page.goto(base + "/voice", wait=2.0)
                    await page.js(INSTALL)
                    if kind == "mic":
                        await page.js("[...document.querySelectorAll('button')].find(b => b.textContent.trim() === 'Record from the microphone').click()")
                        await page.wait_for("[...document.querySelectorAll('button')].some(b => b.textContent.trim() === 'Stop recording')", 10)
                        await asyncio.sleep(seconds + 0.4)
                        await page.js("[...document.querySelectorAll('button')].find(b => b.textContent.trim() === 'Stop recording').click()")
                    else:
                        doc = await page.send("DOM.getDocument")
                        node = await page.send("DOM.querySelector", nodeId=doc["root"]["nodeId"], selector="input[type=file]")
                        await page.send("DOM.setFileInputFiles", nodeId=node["nodeId"], files=[str(path)])
                    ok = await page.wait_for("!!document.querySelector('[data-testid=clip-ready]')", 20)
                    problem = await page.js("document.querySelector('[role=alert]')?.innerText || null")
                    if not ok:
                        out[key] = {"error": problem or "clip was not produced"}
                        continue
                    await page.js("[...document.querySelectorAll('button')].find(b => b.textContent.trim() === 'Score this clip').click()")
                    got = await page.wait_for("!!window.__up", 15)
                    up = await page.js("window.__up") if got else None
                    out[key] = {"wav": base64.b64decode(up["b64"]), "name": up["name"]} if up else {"error": "no upload captured"}
        finally:
            proc.terminate()

    async def run():
        real = sorted((folder / "real").iterdir())
        fake = sorted((folder / "fake").iterdir())
        port = 9440
        for i, src in enumerate(real, 1):
            cap = work / f"capture_{i}.wav"
            ffmpeg("-i", src, "-ac", 1, "-ar", 48000, "-c:a", "pcm_s16le", cap)
            seconds = sf.info(cap).frames / 48000
            await one_session(cap, [(f"real{i}_mic", "mic", None, seconds), (f"real{i}_file", "file", src, 0)], port + i)
        cap = work / "capture_any.wav"
        ffmpeg("-i", fake[0], "-ac", 1, "-ar", 48000, "-c:a", "pcm_s16le", cap)
        await one_session(cap, [(f"fake{i}_file", "file", src, 0) for i, src in enumerate(fake, 1)], port + 9)

    try:
        asyncio.run(run())
    finally:
        httpd.shutdown()
    return out, None


def check3(folder, work, scorer, versions, names):
    """Check 3 as a function: returns (aggregates, per-clip private detail)."""
    caps, why = browser_capture(folder, work)
    if caps is None:
        return {"not_run": why}, {}
    rows = {}
    for key, cap in caps.items():
        if "wav" not in cap:
            rows[key] = {"error": cap["error"]}
            continue
        (work / f"browser_{key}.wav").write_bytes(cap["wav"])
        w, sr = sf.read(io.BytesIO(cap["wav"]), dtype="float32")
        base_key = key.split("_")[0]
        rows[key] = {"browser_wav_rate": sr, "browser_seconds": len(w) / sr, "browser": scorer.score(w),
                     "ffmpeg_path": versions[base_key][names[0]], "path": key.split("_")[1]}
    ok = {k: v for k, v in rows.items() if "browser" in v}
    mic = {k: v for k, v in ok.items() if v["path"] == "mic"}
    fil = {k: v for k, v in ok.items() if v["path"] == "file"}
    agg = {
        "captured": len(ok), "failed": len(rows) - len(ok),
        "microphone_path_real_n": len(mic), "microphone_path_flagged_high": sum(v["browser"]["band"] == "high" for v in mic.values()),
        "file_path_n": len(fil), "file_path_flagged_high": sum(v["browser"]["band"] == "high" for v in fil.values()),
        "mean_abs_logit_diff_browser_vs_ffmpeg_file_path": float(np.mean([abs(v["browser"]["logit"] - v["ffmpeg_path"]["logit"]) for v in fil.values()])) if fil else None,
        "mean_abs_logit_diff_mic_vs_ffmpeg": float(np.mean([abs(v["browser"]["logit"] - v["ffmpeg_path"]["logit"]) for v in mic.values()])) if mic else None,
        "mean_logit_browser_real": float(np.mean([v["browser"]["logit"] for k, v in ok.items() if k.startswith("real")])) if any(k.startswith("real") for k in ok) else None,
        "mean_logit_browser_clone": float(np.mean([v["browser"]["logit"] for k, v in ok.items() if k.startswith("fake")])) if any(k.startswith("fake") for k in ok) else None,
        "note": "the 'microphone' is Chromium's fake device playing the owner's real clip: it exercises the site's MediaRecorder/WebM/resample path, not a real microphone or room",
    }
    return agg, rows


# ---------------------------------------------------------------------------------------------------
def main():
    global FF
    import imageio_ffmpeg
    FF = imageio_ffmpeg.get_ffmpeg_exe()
    folder = Path(sys.argv[1])
    work = folder / "_diagnosis"          # derived audio of the owner's voice: outside the repository
    work.mkdir(exist_ok=True)
    tmp = ROOT / "data" / "self_test" / "tmp"   # dataset-derived audio: data/ is git-ignored
    tmp.mkdir(parents=True, exist_ok=True)
    scorer = Scorer()
    result, private = {}, {}
    only_browser = len(sys.argv) > 2 and sys.argv[2] == "browser"

    real_files = sorted((folder / "real").iterdir())
    fake_files = sorted((folder / "fake").iterdir())
    ds_dir = ROOT / "data" / "demo_clips"
    ds_bona = sorted(ds_dir.glob("asvspoof_bonafide_*.wav"))
    ds_spoof = sorted(ds_dir.glob("asvspoof_spoof_*.wav"))
    clips = {f"real{i}": f for i, f in enumerate(real_files, 1)} | {f"fake{i}": f for i, f in enumerate(fake_files, 1)}
    wave16 = {k: load16(f) for k, f in clips.items()}
    ds16 = {f.stem: sf.read(f, dtype="float32")[0] for f in ds_bona + ds_spoof}

    # ---- sanity: this process reproduces the service ------------------------------------------------
    svc_check = scorer.svc.score_waveform(wave16["real1"], SR)
    mine = scorer.score(wave16["real1"])
    result["replication"] = {"service_score": svc_check["score"], "this_script_score": mine["p"], "abs_diff": abs(svc_check["score"] - mine["p"])}
    assert result["replication"]["abs_diff"] < 1e-6, "this script does not reproduce the service"

    if only_browser:  # re-run check 3 alone and merge it into the saved results
        names = ["A straight resample (ffmpeg), trim 30 dB"]
        versions = {k: {names[0]: scorer.score(w)} for k, w in wave16.items()}
        out_json = ROOT / "data" / "self_test" / "phone_audio.json"
        saved = json.loads(out_json.read_text())
        saved["check3"], priv3 = check3(folder, work, scorer, versions, names)
        out_json.write_text(json.dumps(saved, indent=2))
        pf = folder / "PHONE_AUDIO_DIAGNOSIS_PRIVATE.json"
        priv = json.loads(pf.read_text(encoding="utf-8"))
        priv["check3"] = priv3
        pf.write_text(json.dumps(priv, indent=2, default=lambda o: str(o)), encoding="utf-8")
        print(json.dumps(saved["check3"], indent=1))
        return

    # ---- (1) describe --------------------------------------------------------------------------------
    info = {k: native_info(f) for k, f in clips.items()}
    desc = {k: describe(wave16[k], info[k]["rate_hz"]) for k in clips}
    ds_desc = {k: describe(w, SR) for k, w in ds16.items()}
    private["check1"] = {k: {**info[k], **desc[k]} for k in clips}

    def agg(group, key):
        vals = np.array([group[k][key] for k in group], dtype=float)
        return {"min": float(vals.min()), "mean": float(vals.mean()), "max": float(vals.max())}

    mine_real = {k: v for k, v in desc.items() if k.startswith("real")}
    mine_fake = {k: v for k, v in desc.items() if k.startswith("fake")}
    ds_b = {k: v for k, v in ds_desc.items() if "bonafide" in k}
    ds_s = {k: v for k, v in ds_desc.items() if "spoof" in k}
    keys = ["duration_s", "peak_dbfs", "rms_dbfs", "trimmed_rms_dbfs", "noise_floor_dbfs", "dynamic_range_db", "after_trim_s",
            "kept_fraction_of_duration", "kept_fraction_of_energy", "lead_silence_s", "energy_above_4khz_fraction", "clipped_samples_fraction"]
    result["check1"] = {"groups": {"owner_real_n3": {k: agg(mine_real, k) for k in keys}, "owner_clone_n3": {k: agg(mine_fake, k) for k in keys},
                                   "dataset_genuine_n4": {k: agg(ds_b, k) for k in keys}, "dataset_spoof_n4": {k: agg(ds_s, k) for k in keys}},
                        "owner_real_codec": sorted({f"{v['codec']} {v['rate_hz']} Hz {v['bitrate_kbps']} kb/s" for v in (info[k] for k in info if k.startswith('real'))}),
                        "owner_clone_codec": sorted({f"{v['codec']} {v['rate_hz']} Hz {v['bitrate_kbps']} kb/s" for v in (info[k] for k in info if k.startswith('fake'))})}
    ds_rms = float(np.mean([v["trimmed_rms_dbfs"] for v in ds_desc.values()]))
    ds_peak = float(np.mean([v["peak_dbfs"] for v in ds_desc.values()]))
    result["check1"]["dataset_typical_trimmed_rms_dbfs"] = ds_rms
    result["check1"]["dataset_typical_peak_dbfs"] = ds_peak

    # ---- (2) versions of the owner's clips ------------------------------------------------------------
    def version_waves(k, f):
        w16 = wave16[k]
        native, nsr = load_native(f)
        trimmed, _ = trim_silence(w16, SR, 30.0)
        gain_rms = 10 ** ((ds_rms - db(rms(trimmed))) / 20)
        gain_peak = 10 ** (-1.0 / 20) / max(float(np.max(np.abs(w16))), 1e-9)
        return {
            "A straight resample (ffmpeg), trim 30 dB": (w16, 30.0, "trim_start"),
            "B service-style linear resample, trim 30 dB": (resample_linear(native, nsr)[0] if nsr != SR else w16, 30.0, "trim_start"),
            "C peak-normalised to -1 dBFS": (np.clip(w16 * gain_peak, -1, 1), 30.0, "trim_start"),
            "D RMS-normalised to dataset level": (np.clip(w16 * gain_rms, -1, 1), 30.0, "trim_start"),
            "E trim at 20 dB": (w16, 20.0, "trim_start"), "F trim at 40 dB": (w16, 40.0, "trim_start"),
            "G trim at 50 dB": (w16, 50.0, "trim_start"), "H trim at 60 dB": (w16, 60.0, "trim_start"),
            "I no trim, first 4 s": (w16, 30.0, "none_start"), "J no trim, middle 4 s": (w16, 30.0, "none_middle"),
        }

    versions = {}
    for k, f in clips.items():
        versions[k] = {name: scorer.score(w, db_, mode) for name, (w, db_, mode) in version_waves(k, f).items()}
    private["check2"] = versions
    names = list(next(iter(versions.values())))
    result["check2"] = {"rate_native_equals_16k_for_owner_real": all(info[k]["rate_hz"] == SR for k in info if k.startswith("real")),
                        "versions": {n: {"real_n3": {"flagged_high": sum(versions[k][n]["band"] == "high" for k in versions if k.startswith("real")),
                                                     "flagged_medium": sum(versions[k][n]["band"] == "medium" for k in versions if k.startswith("real")),
                                                     "passed_low": sum(versions[k][n]["band"] == "low" for k in versions if k.startswith("real")),
                                                     "mean_logit": float(np.mean([versions[k][n]["logit"] for k in versions if k.startswith("real")]))},
                                         "clone_n3": {"flagged_high": sum(versions[k][n]["band"] == "high" for k in versions if k.startswith("fake")),
                                                      "mean_logit": float(np.mean([versions[k][n]["logit"] for k in versions if k.startswith("fake")]))}}
                                     for n in names}}
    # ranking inside each pair on the raw (unsaturated) logit: a diagnostic, not a decision rule
    pairs_ranked = sum(versions[f"fake{i}"][names[0]]["logit"] < versions[f"real{i}"][names[0]]["logit"] for i in (1, 2, 3))
    result["check2"]["pairs_where_clone_looks_more_fake_than_real_on_raw_logit"] = int(pairs_ranked)
    # control: the same version changes on dataset clips (level steps should do nothing; trims may)
    ds_versions = {}
    for k, w in ds16.items():
        trimmed, _ = trim_silence(w, SR, 30.0)
        ds_versions[k] = {"baseline": scorer.score(w), "rms_to_owner_level": scorer.score(np.clip(w * 10 ** ((float(np.mean([v['trimmed_rms_dbfs'] for v in mine_real.values()])) - db(rms(trimmed))) / 20), -1, 1)),
                          "trim_60": scorer.score(w, 60.0), "no_trim_first_4s": scorer.score(w, mode="none_start")}
    result["check2"]["dataset_control_max_abs_logit_change"] = {
        n: float(max(abs(ds_versions[k][n]["logit"] - ds_versions[k]["baseline"]["logit"]) for k in ds_versions)) for n in ("rms_to_owner_level", "trim_60", "no_trim_first_4s")}
    result["check2"]["dataset_control_bands_unchanged_for_level"] = all(ds_versions[k]["rms_to_owner_level"]["band"] == ds_versions[k]["baseline"]["band"] for k in ds_versions)

    result["check3"], private["check3"] = check3(folder, work, scorer, versions, names)

    # ---- (4) codec, level, noise, resampling, reverb on dataset clips ----------------------------------
    bona_names = [f.stem for f in ds_bona]
    spoof_names = [f.stem for f in ds_spoof]
    cond = {}
    rng = np.random.default_rng(7)
    for name in bona_names + spoof_names:
        w = ds16[name]
        src = tmp / f"{name}.wav"
        sf.write(src, w, SR, subtype="PCM_16")
        res = {"original": scorer.score(w)}
        for codec in ("resample_only", "mp3_128", "aac_128", "aac_32", "opus_64", "opus_24"):
            for via48 in (False, True):
                if codec == "resample_only" and not via48:
                    continue
                res[f"{codec}{' via 48 kHz' if via48 else ''}"] = scorer.score(reencode(src, codec, via48, tmp), keep=f"{name}|{codec}|{int(via48)}")
        trimmed, _ = trim_silence(w, SR, 30.0)
        speech_rms = rms(trimmed)
        for gain_db in (-20, -40, -60):
            res[f"level {gain_db} dB"] = scorer.score(w * 10 ** (gain_db / 20))
        res["level +12 dB (clipped)"] = scorer.score(np.clip(w * 10 ** (12 / 20), -1, 1))
        for snr in (30, 20, 10):
            res[f"white noise {snr} dB SNR"] = scorer.score(np.clip(w + rng.normal(0, speech_rms / 10 ** (snr / 20), len(w)).astype(np.float32), -1, 1))
        t = np.arange(int(0.5 * SR)) / SR
        ir = rng.normal(0, 1, len(t)) * np.exp(-6.9 * t / 0.4)
        ir /= np.sqrt(np.sum(ir ** 2))
        res["room reverb (RT60 0.4 s)"] = scorer.score(np.clip(fftconvolve(w, ir)[:len(w)], -1, 1).astype(np.float32))
        scorer.feats[f"{name}|original"] = None
        cond[name] = res
    conditions = list(cond[bona_names[0]])
    result["check4"] = {"dataset_genuine_n": len(bona_names), "dataset_spoof_n": len(spoof_names), "conditions": {}}
    for c in conditions:
        result["check4"]["conditions"][c] = {
            "genuine_stay_low": sum(cond[n][c]["band"] == "low" for n in bona_names),
            "genuine_medium": sum(cond[n][c]["band"] == "medium" for n in bona_names),
            "genuine_high": sum(cond[n][c]["band"] == "high" for n in bona_names),
            "spoof_stay_high": sum(cond[n][c]["band"] == "high" for n in spoof_names),
            "spoof_become_low": sum(cond[n][c]["band"] == "low" for n in spoof_names),
            "genuine_mean_logit": float(np.mean([cond[n][c]["logit"] for n in bona_names])),
            "spoof_mean_logit": float(np.mean([cond[n][c]["logit"] for n in spoof_names]))}

    # ---- (5) distributions, aggregates only -------------------------------------------------------------
    z = np.load(ROOT / "artifacts" / "voice" / "phase2c_test_scores.npz")
    lab, logits = z["D_ssl_mean_only_P1_s42__labels"], z["D_ssl_mean_only_P1_s42__scores"].astype(float)
    owner_real_logits = np.array([versions[f"real{i}"][names[0]]["logit"] for i in (1, 2, 3)])
    owner_fake_logits = np.array([versions[f"fake{i}"][names[0]]["logit"] for i in (1, 2, 3)])
    result["check5"] = {
        "test_set_logit_percentiles": {"genuine_p5_p50_p95": [float(np.percentile(logits[lab == 1], q)) for q in (5, 50, 95)],
                                       "spoof_p5_p50_p95": [float(np.percentile(logits[lab == 0], q)) for q in (5, 50, 95)]},
        "owner_real_logit_min_mean_max": [float(owner_real_logits.min()), float(owner_real_logits.mean()), float(owner_real_logits.max())],
        "owner_clone_logit_min_mean_max": [float(owner_fake_logits.min()), float(owner_fake_logits.mean()), float(owner_fake_logits.max())],
        "share_of_test_set_spoof_clips_that_look_LESS_fake_than_owner_real_clips_mean_pct": float(np.mean([pct(logits[lab == 0], lambda a, v=v: a > v) for v in owner_real_logits])),
        "share_of_test_set_genuine_clips_that_look_LESS_genuine_than_owner_real_clips_mean_pct": float(np.mean([pct(logits[lab == 1], lambda a, v=v: a < v) for v in owner_real_logits])),
    }
    # the feature vector that goes into the head, for the owner's clips and the dataset clips
    for k in clips:
        scorer.score(wave16[k], keep=k)
    for n, w in ds16.items():
        scorer.score(w, keep=f"{n}|original")
    F = scorer.feats
    cen_b = np.mean([F[f"{n}|original"] for n in bona_names], 0)
    cen_s = np.mean([F[f"{n}|original"] for n in spoof_names], 0)
    own_r = [F[f"real{i}"] for i in (1, 2, 3)]
    own_f = [F[f"fake{i}"] for i in (1, 2, 3)]
    aac32 = [F[f"{n}|aac_32|0"] for n in bona_names]
    aac128 = [F[f"{n}|aac_128|0"] for n in bona_names]
    summary = lambda vs: {"cosine_to_dataset_genuine_centroid": float(np.mean([cosine(v, cen_b) for v in vs])),  # noqa: E731
                          "cosine_to_dataset_spoof_centroid": float(np.mean([cosine(v, cen_s) for v in vs])),
                          "closer_to_spoof_centroid": int(sum(cosine(v, cen_s) > cosine(v, cen_b) for v in vs)), "n": len(vs)}
    result["check5"]["feature_vector_groups"] = {
        "owner_real": summary(own_r), "owner_clone": summary(own_f),
        "dataset_genuine_re-encoded_AAC_32k": summary(aac32), "dataset_genuine_re-encoded_AAC_128k": summary(aac128),
        "dataset_genuine_original_(each_vs_the_other_three_left_out)": {
            "closer_to_spoof_centroid": int(sum(cosine(F[f"{n}|original"], cen_s) > cosine(F[f"{n}|original"], np.mean([F[f"{m}|original"] for m in bona_names if m != n], 0)) for n in bona_names)), "n": len(bona_names)},
        "cosine_owner_real_to_owner_clone_same_sentence_mean": float(np.mean([cosine(F[f"real{i}"], F[f"fake{i}"]) for i in (1, 2, 3)])),
        "cosine_owner_real_to_each_other_mean": float(np.mean([cosine(F["real1"], F["real2"]), cosine(F["real1"], F["real3"]), cosine(F["real2"], F["real3"])])),
        "cosine_dataset_genuine_to_each_other_mean": float(np.mean([cosine(F[f"{a}|original"], F[f"{b}|original"]) for i, a in enumerate(bona_names) for b in bona_names[i + 1:]])),
    }

    out_json = ROOT / "data" / "self_test" / "phone_audio.json"
    out_json.write_text(json.dumps(result, indent=2))
    (folder / "PHONE_AUDIO_DIAGNOSIS_PRIVATE.json").write_text(json.dumps(private, indent=2, default=lambda o: str(o)), encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
