// Browser-side audio handling for the voice desk: decode whatever the reader gives us (a file or a
// microphone recording), convert it to 16 kHz mono WAV for the API, and compute a spectrogram to draw.
// Nothing here scores anything; scoring happens only in the API.

export const TARGET_RATE = 16000;

export interface Clip {
  name: string;
  samples: Float32Array; // mono, TARGET_RATE
  seconds: number;
  wav: Blob;
}

export class AudioProblem extends Error {}

export async function clipFromBlob(blob: Blob, name: string, maxSeconds: number | null): Promise<Clip> {
  const Ctx: typeof AudioContext | undefined =
    window.AudioContext ?? (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
  if (!Ctx) throw new AudioProblem("This browser cannot decode audio. Try a current Chrome, Edge or Firefox.");
  const ctx = new Ctx();
  let decoded: AudioBuffer;
  try {
    decoded = await ctx.decodeAudioData(await blob.arrayBuffer());
  } catch {
    throw new AudioProblem("That file isn't audio this browser can read. Use a WAV, FLAC, MP3 or OGG recording.");
  } finally {
    void ctx.close();
  }
  if (maxSeconds !== null && decoded.duration > maxSeconds) {
    throw new AudioProblem(`That clip is ${Math.round(decoded.duration)} seconds long. The limit is ${maxSeconds} seconds.`);
  }
  if (decoded.duration < 0.1) throw new AudioProblem("That clip is too short to score. Record at least a second of speech.");
  const frames = Math.max(1, Math.round(decoded.duration * TARGET_RATE));
  const offline = new OfflineAudioContext(1, frames, TARGET_RATE);
  const source = offline.createBufferSource();
  source.buffer = decoded;
  source.connect(offline.destination);
  source.start();
  const samples = (await offline.startRendering()).getChannelData(0).slice();
  return { name: name.replace(/\.[^.]+$/, "") + ".wav", samples, seconds: samples.length / TARGET_RATE, wav: encodeWav(samples) };
}

export function encodeWav(samples: Float32Array, rate = TARGET_RATE): Blob {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const text = (offset: number, s: string) => [...s].forEach((c, i) => view.setUint8(offset + i, c.charCodeAt(0)));
  text(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  text(8, "WAVEfmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, rate, true);
  view.setUint32(28, rate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  text(36, "data");
  view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, Math.round(s * 32767), true);
  }
  return new Blob([buffer], { type: "audio/wav" });
}

/** Magnitude spectrogram in dB, [frame][bin], Hann window. Small radix-2 FFT, no dependency. */
export function spectrogram(samples: Float32Array, size = 512, maxFrames = 600): Float32Array[] {
  const hop = Math.max(size / 2, Math.ceil((samples.length - size) / maxFrames));
  const window = Float32Array.from({ length: size }, (_, i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (size - 1)));
  const frames: Float32Array[] = [];
  const re = new Float32Array(size);
  const im = new Float32Array(size);
  for (let start = 0; start + size <= samples.length; start += hop) {
    for (let i = 0; i < size; i++) {
      re[i] = samples[start + i] * window[i];
      im[i] = 0;
    }
    fft(re, im);
    const mags = new Float32Array(size / 2);
    for (let k = 0; k < size / 2; k++) mags[k] = 10 * Math.log10(re[k] * re[k] + im[k] * im[k] + 1e-10);
    frames.push(mags);
  }
  return frames;
}

function fft(re: Float32Array, im: Float32Array): void {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const angle = (-2 * Math.PI) / len;
    for (let i = 0; i < n; i += len) {
      for (let k = 0; k < len / 2; k++) {
        const wr = Math.cos(angle * k);
        const wi = Math.sin(angle * k);
        const a = i + k;
        const b = a + len / 2;
        const xr = re[b] * wr - im[b] * wi;
        const xi = re[b] * wi + im[b] * wr;
        re[b] = re[a] - xr;
        im[b] = im[a] - xi;
        re[a] += xr;
        im[a] += xi;
      }
    }
  }
}
