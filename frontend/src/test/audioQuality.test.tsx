// The audio-quality notice: plain words by default, measures under "For engineers", no hard-coded colours.
import { fireEvent, render, screen } from "@testing-library/react";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import App from "../App";
import { AudioQualityNotice, otherWarnings } from "../components/caseParts";
import { EngineerLayer, EngineerToggle } from "../components/engineers";
import type { VoiceScore } from "../lib/api";

const BASE: VoiceScore = {
  clip: "x.wav", score: 0.9, verdict: "high", model: "D", model_requested: "D", fallback_used: false,
  thresholds: { medium: 0.02, high: 0.66, source: "" }, duration_s: 5, speech_s: 4,
  low_confidence: false, warnings: [], low_quality_audio: false, audio_quality: null,
};
const QUALITY = {
  analysed: true, loud_frames: 300, speech_like: true, low_quality_audio: true, reasons: ["very little sound above 5 kHz"],
  spectral_movement: 0.712, hf_to_mid_db: -61.4, cutoff_hz: 5200,
  thresholds: { spectral_movement_min: 0.3, hf_to_mid_min_db: -55, cutoff_min_hz: 7600 },
};
const LOW: VoiceScore = {
  ...BASE, low_quality_audio: true, low_confidence: true, audio_quality: QUALITY,
  warnings: ["This audio looks phone-quality or heavily compressed. This model was trained on clean recordings, so the result is unreliable.", "Only 0.8 s of sound after trimming silence."],
};

beforeEach(() => {
  sessionStorage.clear();
  vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("no network in tests")));
});

function withToggle(voice: VoiceScore) {
  return render(
    <EngineerLayer page="t">
      <EngineerToggle />
      <AudioQualityNotice voice={voice} />
    </EngineerLayer>,
  );
}

describe("audio quality notice", () => {
  it("says in plain words that the result may be wrong, and calls it a known weakness", () => {
    withToggle(LOW);
    const text = screen.getByTestId("low-quality-notice").textContent ?? "";
    expect(text).toContain("The result may be wrong for this recording.");
    expect(text).toContain("This audio looks phone-quality or heavily compressed.");
    expect(text).toContain("so the result is unreliable");
    expect(text).toContain("known weakness");
  });

  it("is a live-region alert, so it is announced", () => {
    withToggle(LOW);
    expect(screen.getByTestId("low-quality-notice").getAttribute("role")).toBe("alert");
  });

  it("shows nothing for clean audio", () => {
    withToggle({ ...BASE, audio_quality: { ...QUALITY, low_quality_audio: false, reasons: [] } });
    expect(screen.queryByTestId("low-quality-notice")).toBeNull();
  });

  it("keeps the measures out of sight until the engineer detail is opened", () => {
    withToggle(LOW);
    expect(screen.queryByTestId("audio-quality-table")).toBeNull();
    fireEvent.click(screen.getByText("Show the detail for engineers"));
    const table = screen.getByTestId("audio-quality-table").textContent ?? "";
    expect(table).toContain("0.712");
    expect(table).toContain("-61.4 dB");
    expect(table).toContain("5200 Hz");
    expect(table).toContain("7600 Hz");
    expect(document.body.textContent).toContain("they never change the score");
  });

  it("separates the low-quality warning from other warnings", () => {
    expect(otherWarnings(LOW)).toEqual(["Only 0.8 s of sound after trimming silence."]);
    expect(otherWarnings(BASE)).toEqual([]);
  });

  it("uses theme tokens only, so it reads in light and dark", () => {
    const source = readFileSync(join(process.cwd(), "src", "components", "caseParts.tsx"), "utf-8");
    const notice = source.slice(source.indexOf("export function AudioQualityNotice"));
    expect(notice).not.toMatch(/#[0-9a-fA-F]{3,8}\b/);
    expect(notice).not.toMatch(/rgb\(|hsl\(/);
    expect(notice).toContain("border-flag bg-sheet");
  });
});

describe("the voice page", () => {
  it("states the compressed-phone-audio weakness plainly in the default layer", () => {
    render(
      <MemoryRouter initialEntries={["/voice"]}>
        <App />
      </MemoryRouter>,
    );
    const text = document.querySelector("main")!.textContent ?? "";
    expect(text).toContain("Compressed phone audio is a known weakness.");
    expect(text).toContain("refuses a sound that is not speech");
  });
});
