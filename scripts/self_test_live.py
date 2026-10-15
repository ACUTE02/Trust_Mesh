"""Self-test, live part (ST2 and ST5 of PHASE_2_SELF_TEST_RUNBOOK.md). Needs the stack running.

ST2: 200 random labelled test-period nodes scored through the API, compared with the offline scorer.
ST5: bad and awkward inputs; every one must give a clear error or a flagged result, never a crash.
All audio here is generated in this script (silence, noise, a tone); nothing comes from a dataset.
Aggregates only are printed and written to data/self_test/live.json (local, not committed).

Run: .venv\\Scripts\\python.exe scripts/self_test_live.py [--api http://localhost:8000] [--skip-rate-limit]
"""
import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import requests
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data" / "self_test"
SR = 16000


def wav(samples, sr=SR):
    buf = io.BytesIO()
    sf.write(buf, np.asarray(samples, dtype=np.float32), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def tone(seconds):
    t = np.arange(int(seconds * SR)) / SR
    return 0.3 * np.sin(2 * np.pi * 170 * t) * (1 + 0.6 * np.sin(2 * np.pi * 4 * t))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--skip-rate-limit", action="store_true")
    args = parser.parse_args()
    api = args.api
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261008)
    result = {}
    quiet = {"record": "false"}  # do not write self-test rows into the cases table

    # --- ST2 graph ---------------------------------------------------------------------------------
    from serving.graph_service import GraphService
    g = GraphService()
    g.load(with_gat=False)
    test = np.flatnonzero((g.labels != 2) & (g.time_step >= 36))
    sample = rng.choice(test, 200, replace=False)
    diffs, verdict_mismatch = [], 0
    for node in sample:
        r = requests.get(f"{api}/graph/score/{int(node)}", params=quiet, timeout=30).json()
        diffs.append(abs(r["score"] - float(g.scores[node])))
        verdict_mismatch += r["verdict"] != g.verdict(float(g.scores[node]))
    result["st2_graph"] = {"n": 200, "max_abs_diff": float(max(diffs)), "mean_abs_diff": float(np.mean(diffs)),
                           "verdict_mismatches": int(verdict_mismatch), "pass": max(diffs) <= 1e-3}

    # --- ST5 fail-safe -----------------------------------------------------------------------------
    cases = []

    def record(name, response, expect, note=""):
        try:
            body = response.json()
        except ValueError:
            body = response.text[:200]
        detail = body.get("detail") if isinstance(body, dict) else body
        summary = detail if detail else {k: body[k] for k in ("score", "verdict", "low_confidence", "warnings") if k in body}
        ok = expect(response, body)
        cases.append({"case": name, "status": response.status_code, "answer": summary, "pass": bool(ok), "note": note})
        print(("PASS " if ok else "FAIL ") + name, response.status_code, str(summary)[:160])

    def post(blob, name="clip.wav", ctype="audio/wav"):
        return requests.post(f"{api}/voice/score", files={"file": (name, blob, ctype)}, params=quiet, timeout=120)

    is_error = lambda code: (lambda r, b: r.status_code == code and isinstance(b, dict) and bool(b.get("detail")))  # noqa: E731
    flagged = lambda r, b: r.status_code == 422 or (r.status_code == 200 and b.get("low_confidence") is True and b.get("warnings"))  # noqa: E731

    record("transaction number that does not exist", requests.get(f"{api}/graph/score/999999", timeout=30), is_error(404))
    record("transaction number that is not a number", requests.get(f"{api}/graph/score/abc", timeout=30),
           lambda r, b: r.status_code == 422)
    record("empty file", post(b""), is_error(422))
    record("wrong file type (a text file)", post(b"hello, not audio", "notes.txt", "text/plain"), is_error(415))
    record("61-second clip", post(wav(tone(61))), is_error(422))
    record("file over 10 MB", post(wav(tone(350))), is_error(413))
    record("0.1-second clip", post(wav(tone(0.1))), flagged)
    record("pure silence, 3 seconds", post(wav(np.zeros(3 * SR))), flagged)
    not_speech = lambda r, b: r.status_code == 422 and "does not look like speech" in str(b.get("detail", ""))  # noqa: E731
    record("loud white noise, 3 seconds", post(wav(np.clip(rng.normal(0, 0.5, 3 * SR), -1, 1))), not_speech,
           "Refused as not speech (the audio guard, added 2026-10-15). Before it, noise got an ordinary score.")
    record("synthetic tone, 3 seconds", post(wav(tone(3))), not_speech,
           "Refused as not speech (the audio guard). Before it, a tone got an ordinary score.")

    if not args.skip_rate_limit:
        # Sent at once, not one after another: one scoring call takes over a second on a CPU, so a single
        # patient client cannot reach 30 a minute. The limit counts requests as they arrive.
        from concurrent.futures import ThreadPoolExecutor
        time.sleep(61)  # start from an empty minute
        blob = wav(tone(1.5))
        with ThreadPoolExecutor(40) as pool:
            codes = list(pool.map(lambda _: post(blob).status_code, range(40)))
        refused, served = codes.count(429), len(codes) - codes.count(429)  # a tone is now a 422, not a 200
        ok = refused == 10 and served == 30
        cases.append({"case": "40 scoring requests sent at once (limit is 30 a minute)", "status": 429 if refused else 200,
                      "answer": f"{served} scored, {refused} refused with a 'too many requests' message",
                      "pass": ok, "note": "One request after another cannot reach the limit: a scoring call takes over a second."})
        print(("PASS " if ok else "FAIL ") + "rate limit", refused, "of 40 refused,", served, "scored")
        alive = requests.get(f"{api}/health", timeout=30).status_code
        cases.append({"case": "service still answering after all of the above", "status": alive, "answer": "", "pass": alive == 200, "note": ""})

    result["st5"] = cases
    (OUT / "live.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result["st2_graph"]))
    time.sleep(0)


if __name__ == "__main__":
    main()
