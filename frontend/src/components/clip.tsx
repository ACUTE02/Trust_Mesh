import { useEffect, useRef, useState } from "react";
import { AudioProblem, TARGET_RATE, clipFromBlob, spectrogram, type Clip } from "../lib/audio";
import type { Health } from "../lib/api";
import { Live } from "./evidence";

/** Limits come from the API (/health). If it is unreachable they are unknown, and we say so. */
export function limitsText(health: Health | null | "down"): string {
  if (health && health !== "down" && health.limits) {
    return `Up to ${health.limits.max_audio_seconds} seconds and ${health.limits.max_upload_mb} MB. WAV, FLAC, MP3 or OGG.`;
  }
  return "WAV, FLAC, MP3 or OGG. The size and length limits are set by the scoring service.";
}

/**
 * Choose a file or record from the microphone. Either way the audio is decoded in the browser and
 * converted to 16 kHz mono WAV before anything is sent. No dataset clips are offered: bring your own.
 */
export function ClipInput({
  clip,
  onClip,
  health,
  optional = false,
}: {
  clip: Clip | null;
  onClip: (clip: Clip | null) => void;
  health: Health | null | "down";
  optional?: boolean;
}) {
  const [problem, setProblem] = useState<string | null>(null);
  const [recording, setRecording] = useState(false);
  const [busy, setBusy] = useState(false);
  const recorder = useRef<MediaRecorder | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const maxSeconds = health && health !== "down" && health.limits ? health.limits.max_audio_seconds : null;
  const maxBytes = health && health !== "down" && health.limits ? health.limits.max_upload_mb * 1024 * 1024 : null;

  async function accept(blob: Blob, name: string, type: string) {
    setProblem(null);
    if (type && !type.startsWith("audio/") && !type.startsWith("video/webm") && type !== "application/octet-stream") {
      setProblem(`"${name}" is not an audio file. Choose a WAV, FLAC, MP3 or OGG recording.`);
      return;
    }
    if (maxBytes !== null && blob.size > maxBytes) {
      setProblem(`"${name}" is larger than the ${health && health !== "down" ? health.limits?.max_upload_mb : ""} MB limit.`);
      return;
    }
    setBusy(true);
    try {
      onClip(await clipFromBlob(blob, name, maxSeconds));
    } catch (e) {
      onClip(null);
      setProblem(e instanceof AudioProblem ? e.message : "That file could not be read as audio.");
    } finally {
      setBusy(false);
    }
  }

  async function startRecording() {
    setProblem(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const rec = new MediaRecorder(stream);
      const chunks: Blob[] = [];
      rec.ondataavailable = (ev) => chunks.push(ev.data);
      rec.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        setRecording(false);
        void accept(new Blob(chunks, { type: rec.mimeType }), "recording", "audio/webm");
      };
      recorder.current = rec;
      rec.start();
      setRecording(true);
    } catch {
      setProblem("The microphone isn't available. Allow microphone access for this page, or choose a file instead.");
    }
  }

  return (
    <div>
      <div className="flex flex-wrap items-center gap-3">
        <input
          ref={fileInput}
          type="file"
          accept="audio/*,.wav,.flac,.mp3,.ogg"
          className="sr-only"
          aria-label="Choose an audio file"
          onChange={(ev) => {
            const f = ev.target.files?.[0];
            if (f) void accept(f, f.name, f.type);
            ev.target.value = "";
          }}
        />
        <button type="button" className="btn btn-quiet" onClick={() => fileInput.current?.click()} disabled={recording || busy}>
          Choose an audio file
        </button>
        {recording ? (
          <button type="button" className="btn" onClick={() => recorder.current?.stop()}>
            Stop recording
          </button>
        ) : (
          <button type="button" className="btn btn-quiet" onClick={() => void startRecording()} disabled={busy}>
            Record from the microphone
          </button>
        )}
        {clip && (
          <button type="button" className="underline" onClick={() => onClip(null)}>
            Remove the clip
          </button>
        )}
      </div>
      <p className="mt-2 mb-0 text-xs text-ink-soft">
        {optional ? "Optional. " : ""}
        Use your own recording. {limitsText(health)}
      </p>
      {recording && (
        <p role="status" className="mt-2 mb-0 text-sm">
          Recording. Speak for a few seconds, then stop.
        </p>
      )}
      {busy && (
        <p role="status" className="mt-2 mb-0 text-sm text-ink-soft">
          Reading the audio.
        </p>
      )}
      {clip && !busy && (
        <p className="mt-2 mb-0 text-sm" data-testid="clip-ready">
          Clip ready: {clip.name}, <Live>{clip.seconds.toFixed(1)}</Live> seconds.
        </p>
      )}
      {problem && (
        <p role="alert" className="measure mt-2 mb-0 border-l-4 border-flag bg-sheet px-3 py-2 text-sm">
          {problem}
        </p>
      )}
    </div>
  );
}

function css(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

/**
 * Waveform and spectrogram of one clip, with the silence the scorer trims marked once the API has
 * reported it. The text under the figure says the same thing in words.
 */
export function ClipView({ clip, leadSilence, trailSilence }: { clip: Clip; leadSilence?: number; trailSilence?: number }) {
  const wave = useRef<HTMLCanvasElement>(null);
  const spec = useRef<HTMLCanvasElement>(null);
  const [theme, setTheme] = useState(document.documentElement.dataset.theme);

  useEffect(() => {
    const observer = new MutationObserver(() => setTheme(document.documentElement.dataset.theme));
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const ink = css("--ink");
    const hue = css("--spectrum");
    const flag = css("--flag");
    const sheet = css("--sheet");
    const w = wave.current;
    const s = spec.current;
    if (!w || !s) return;
    const width = (w.width = s.width = 900);
    const markSilence = (ctx: CanvasRenderingContext2D, height: number) => {
      ctx.fillStyle = flag;
      ctx.globalAlpha = 0.28;
      if (leadSilence) ctx.fillRect(0, 0, (leadSilence / clip.seconds) * width, height);
      if (trailSilence) ctx.fillRect(width - (trailSilence / clip.seconds) * width, 0, (trailSilence / clip.seconds) * width, height);
      ctx.globalAlpha = 1;
    };

    w.height = 110;
    const wc = w.getContext("2d");
    if (wc) {
      wc.fillStyle = sheet;
      wc.fillRect(0, 0, width, w.height);
      wc.fillStyle = ink;
      const per = clip.samples.length / width;
      for (let x = 0; x < width; x++) {
        let lo = 1;
        let hi = -1;
        for (let i = Math.floor(x * per); i < Math.floor((x + 1) * per); i++) {
          const v = clip.samples[i];
          if (v < lo) lo = v;
          if (v > hi) hi = v;
        }
        if (hi < lo) continue;
        wc.fillRect(x, (1 - hi) * 55, 1, Math.max(1, (hi - lo) * 55));
      }
      markSilence(wc, w.height);
    }

    s.height = 128;
    const sc = s.getContext("2d");
    if (sc) {
      sc.fillStyle = sheet;
      sc.fillRect(0, 0, width, s.height);
      const frames = spectrogram(clip.samples);
      if (frames.length) {
        let top = -Infinity;
        for (const f of frames) for (const v of f) if (v > top) top = v;
        const floor = top - 70;
        const bins = frames[0].length;
        const colWidth = width / frames.length;
        sc.fillStyle = hue;
        frames.forEach((frame, fi) => {
          for (let y = 0; y < s.height; y++) {
            const v = frame[Math.floor(((s.height - 1 - y) / s.height) * bins)];
            const a = Math.max(0, Math.min(1, (v - floor) / 70));
            if (a < 0.04) continue;
            sc.globalAlpha = a;
            sc.fillRect(fi * colWidth, y, Math.ceil(colWidth), 1);
          }
        });
        sc.globalAlpha = 1;
      }
      markSilence(sc, s.height);
    }
  }, [clip, leadSilence, trailSilence, theme]);

  const hasSilence = leadSilence !== undefined;
  return (
    <figure className="m-0">
      <canvas ref={wave} className="block h-[5.5rem] w-full" role="img" aria-label="Waveform of the clip" />
      <canvas ref={spec} className="mt-1 block h-28 w-full" role="img" aria-label={`Spectrogram of the clip, 0 to ${TARGET_RATE / 2000} kHz`} />
      <figcaption className="mt-2 text-xs text-ink-soft">
        Waveform (top) and spectrogram (bottom, low frequencies at the bottom) of {clip.name},{" "}
        <Live>{clip.seconds.toFixed(1)}</Live> seconds.{" "}
        {hasSilence ? (
          <>
            Shaded: silence the scorer trimmed,{" "}
            <Live>{(leadSilence ?? 0).toFixed(2)}</Live> s at the start and{" "}
            <Live>{(trailSilence ?? 0).toFixed(2)}</Live> s at the end.
          </>
        ) : (
          "Score the clip to see which silence the scorer trims."
        )}
      </figcaption>
    </figure>
  );
}
