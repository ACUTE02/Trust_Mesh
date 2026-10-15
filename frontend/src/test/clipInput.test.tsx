// Regression (2026-10-14 final audit, BF-06): choosing a rejected file after a good one left the old clip
// loaded, so "Score" silently scored the previous recording while an error was on screen.
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ClipInput } from "../components/clip";
import type { Health } from "../lib/api";
import type { Clip } from "../lib/audio";

const oldClip: Clip = { name: "good.wav", samples: new Float32Array(16000), seconds: 1, wav: new Blob() };
const health = {
  graph: { ready: true, model: "x" }, voice: { ready: true }, neo4j: { connected: false },
  limits: { max_upload_mb: 1, max_audio_seconds: 60, scoring_requests_per_minute: 30 },
} as Health;

function pick(file: File) {
  const input = screen.getByLabelText("Choose an audio file") as HTMLInputElement;
  fireEvent.change(input, { target: { files: [file] } });
}

describe("ClipInput with a clip already loaded", () => {
  it("drops the old clip when a non-audio file is chosen", async () => {
    const onClip = vi.fn();
    render(<ClipInput clip={oldClip} onClip={onClip} health={health} />);
    pick(new File(["hello"], "notes.txt", { type: "text/plain" }));
    await waitFor(() => expect(screen.getByText(/is not an audio file/)).toBeTruthy());
    expect(onClip).toHaveBeenCalledWith(null);
  });

  it("drops the old clip when the file is over the size limit", async () => {
    const onClip = vi.fn();
    render(<ClipInput clip={oldClip} onClip={onClip} health={health} />);
    pick(new File([new Uint8Array(2 * 1024 * 1024)], "big.wav", { type: "audio/wav" }));
    await waitFor(() => expect(screen.getByText(/larger than the 1 MB limit/)).toBeTruthy());
    expect(onClip).toHaveBeenCalledWith(null);
  });
});
