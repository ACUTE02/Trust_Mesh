// Turns audio samples into a height grid for the Voice world's terrain: time across, frequency along the depth
// axis, loudness as height. Pure maths on the samples already decoded in the browser; nothing is sent anywhere.
import { spectrogram } from "./audio";

export const TERRAIN_RATE = 16000; // the same rate the API receives
const ABSOLUTE_LOW_DB = -30; // spectrogram level (in the units of audio.spectrogram) below which the terrain is flat

/** Height grid in 0..1, row-major [row][col]: row 0 is the lowest frequency, the last row is the highest (the Nyquist frequency). */
export function terrainHeights(samples: Float32Array, cols: number, rows: number, floorDb = 60): Float32Array {
  const frames = spectrogram(samples, 512, cols);
  const out = new Float32Array(cols * rows);
  if (!frames.length) return out;
  const bins = frames[0].length;
  let peak = -Infinity;
  for (const f of frames) for (let k = 0; k < bins; k++) if (f[k] > peak) peak = f[k];
  if (!Number.isFinite(peak)) return out;
  // Silence must stay flat: the floor is also absolute, so a clip with nothing in it is not stretched up to full height.
  const low = Math.max(peak - floorDb, ABSOLUTE_LOW_DB);
  const span = peak - low;
  if (span <= 3) return out;
  for (let c = 0; c < cols; c++) {
    const frame = frames[Math.min(frames.length - 1, Math.floor((c * frames.length) / cols))];
    for (let r = 0; r < rows; r++) {
      const k0 = Math.floor((r * bins) / rows);
      const k1 = Math.max(k0 + 1, Math.floor(((r + 1) * bins) / rows));
      let m = -Infinity;
      for (let k = k0; k < k1; k++) if (frame[k] > m) m = frame[k];
      const h = Math.min(1, Math.max(0, (m - low) / Math.max(span, floorDb / 2)));
      out[r * cols + c] = Math.pow(h, 1.35);
    }
  }
  return out;
}

/**
 * A short stretch of vowel-like sound made from scratch (a voice-pitch sweep through two formants, in syllables), so
 * the idle terrain is built the same way as a real clip's. Deterministic: the same call gives the same samples.
 */
export function synthSpeech(seconds = 2.6, rate = TERRAIN_RATE): Float32Array {
  const n = Math.floor(seconds * rate);
  const out = new Float32Array(n);
  const formants: [number, number][] = [[700, 1200], [400, 2200], [550, 1700], [300, 2500], [650, 1100]];
  let phase = 0;
  for (let i = 0; i < n; i++) {
    const t = i / rate;
    const syllable = Math.floor(t * 2.4);
    const inSyl = (t * 2.4) % 1;
    const env = Math.sin(Math.PI * Math.min(1, inSyl * 1.15)) ** 1.5;
    const f0 = 118 + 38 * Math.sin(t * 1.9) + syllable * 6;
    phase += (2 * Math.PI * f0) / rate;
    const [f1, f2] = formants[syllable % formants.length];
    let v = 0;
    for (let h = 1; h <= 28; h++) {
      const f = h * f0;
      if (f > 7600) break;
      const g = Math.exp(-Math.pow((f - f1) / 160, 2)) + 0.6 * Math.exp(-Math.pow((f - f2) / 230, 2)) + 0.05 / h;
      v += (g / h ** 0.35) * Math.sin(h * phase);
    }
    out[i] = 0.2 * env * v;
  }
  return out;
}
