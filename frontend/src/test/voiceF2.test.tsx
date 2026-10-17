// The opt-in voice model F2: hidden unless the service says it is installed, D stays the default, the wording and the
// numbers come from the data module, and the case desk never offers it.
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "../App";
import { F2_WORDING, VoiceModelChoice, f2Available } from "../components/voiceModel";
import { EngineerLayer, EngineerToggle } from "../components/engineers";
import { ModelInUse } from "../components/caseParts";
import evidence from "../data/evidence.json";
import { api, type Health } from "../lib/api";

const HEALTH_WITH_F2: Health = {
  graph: { ready: true, model: "xgboost_tier1_tier2" },
  voice: { ready: true, serving_default: "D", loaded: ["D", "F2"], opt_in: { F2: true } },
  neo4j: { connected: false },
};
const HEALTH_WITHOUT_F2: Health = {
  graph: { ready: true, model: "xgboost_tier1_tier2" },
  voice: { ready: true, serving_default: "D", loaded: ["D"], opt_in: { F2: false } },
  neo4j: { connected: false },
};

function stubHealth(health: Health | "down") {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockImplementation(async (url: string) => {
      if (health === "down" || !String(url).endsWith("/health")) throw new Error("no network in tests");
      return new Response(JSON.stringify(health), { status: 200, headers: { "content-type": "application/json" } });
    }),
  );
}

beforeEach(() => sessionStorage.clear());
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

function value(id: string): number {
  return (evidence as unknown as { entries: Record<string, { value: number }> }).entries[id].value;
}

describe("F2 availability", () => {
  it("is available only when the service reports its weights as loaded", () => {
    expect(f2Available(HEALTH_WITH_F2)).toBe(true);
    expect(f2Available(HEALTH_WITHOUT_F2)).toBe(false);
    expect(f2Available("down")).toBe(false);
    expect(f2Available(null)).toBe(false);
    expect(f2Available({ ...HEALTH_WITH_F2, voice: { ready: true } })).toBe(false);
  });

  it("hides the choice completely when F2 is not installed", () => {
    render(<VoiceModelChoice health={HEALTH_WITHOUT_F2} value="D" onChange={() => {}} />);
    expect(screen.queryByTestId("voice-model-choice")).toBeNull();
    expect(screen.queryByText(/F2/)).toBeNull();
  });
});

describe("the voice page", () => {
  const open = () =>
    render(
      <MemoryRouter initialEntries={["/voice"]}>
        <App />
      </MemoryRouter>,
    );

  it("has no F2 choice when the service does not list it, and still renders the page", async () => {
    stubHealth(HEALTH_WITHOUT_F2);
    open();
    await screen.findByText(/Try it: your voice, as a landscape/);
    await new Promise((r) => setTimeout(r, 50));
    expect(screen.queryByTestId("voice-model-select")).toBeNull();
  });

  it("has no F2 choice when the service is down", async () => {
    stubHealth("down");
    open();
    await screen.findByText(/Try it: your voice, as a landscape/);
    expect(screen.queryByTestId("voice-model-select")).toBeNull();
  });

  it("offers F2 as a small secondary choice with D selected by default", async () => {
    stubHealth(HEALTH_WITH_F2);
    open();
    const select = (await screen.findByTestId("voice-model-select")) as HTMLSelectElement;
    expect(select.value).toBe("D");
    expect(Array.from(select.options).map((o) => o.value)).toEqual(["D", "F2"]);
    expect(select.options[0].textContent).toContain("default");
    expect(screen.queryByTestId("f2-wording")).toBeNull();
  });

  it("shows the F2 wording and the engineer numbers only after F2 is chosen", async () => {
    stubHealth(HEALTH_WITH_F2);
    open();
    const select = await screen.findByTestId("voice-model-select");
    fireEvent.change(select, { target: { value: "F2" } });
    expect(screen.getByTestId("f2-wording").textContent).toBe(F2_WORDING);
    expect(F2_WORDING).toContain("reduced, not removed");
    expect(F2_WORDING).toContain("not tested on real phone calls");
    expect(F2_WORDING).toContain("D remains the main model");
    // numbers are inside "For engineers", so they appear once the engineer detail is opened
    expect(screen.queryByTestId("f2-numbers")).toBeNull();
    fireEvent.click(screen.getAllByText("Show the detail for engineers")[0]);
    const text = screen.getByTestId("f2-numbers").textContent ?? "";
    for (const id of ["p2d.flag.aac32.control", "p2d.flag.aac32.f2", "p2d.flag.via48.control", "p2d.flag.via48.f2", "p2d.flag.clean.f2"]) {
      expect(text).toContain(value(id).toFixed(4));
    }
    expect(text).toContain("0.5982");
    expect(text).toContain("0.0596");
    expect(text).toContain("0.9095");
    expect(text).toContain("0.1012");
    expect(text).toContain("0.0319");
    expect(text).toMatch(/better than the clean-only head on 1 of 4 unseen codec settings/);
    expect(text).toContain("Still weak on synthetic room echo");
    expect(text).toContain("no medium band");
  });

  it("makes no validated claim and no light-theme claim around F2", async () => {
    stubHealth(HEALTH_WITH_F2);
    open();
    fireEvent.change(await screen.findByTestId("voice-model-select"), { target: { value: "F2" } });
    fireEvent.click(screen.getAllByText("Show the detail for engineers")[0]);
    const text = (screen.getByTestId("voice-model-choice").textContent ?? "").toLowerCase();
    expect(text).not.toMatch(/\bvalidated\b/);
    expect(text).not.toMatch(/light (theme|mode)/);
  });

  it("sends ?model=F2 only for the opt-in choice; the default sends no model at all", async () => {
    const calls: string[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation(async (url: string) => {
        calls.push(String(url));
        return new Response(JSON.stringify({ clip: "x", score: 0.1, verdict: "low", model: "D" }), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      }),
    );
    const wav = new Blob([new Uint8Array(10)], { type: "audio/wav" });
    await api.voiceScore(wav, "a.wav");
    await api.voiceScore(wav, "a.wav", "D");
    await api.voiceScore(wav, "a.wav", "F2");
    expect(calls).toEqual(["/api/voice/score", "/api/voice/score", "/api/voice/score?model=F2"]);
  });
});

describe("naming the model that scored", () => {
  it("shows the label with the seeds for F2 and for D", () => {
    const { rerender } = render(
      <EngineerLayer page="t">
        <EngineerToggle />
        <ModelInUse model="F2" label="F2 (mean of seeds 42, 43 and 44)" />
      </EngineerLayer>,
    );
    expect(screen.getByTestId("model-in-use").textContent).toContain("F2 (mean of seeds 42, 43 and 44)");
    rerender(
      <EngineerLayer page="t">
        <ModelInUse model="D" label="D (seed 42)" />
      </EngineerLayer>,
    );
    expect(screen.getByTestId("model-in-use").textContent).toContain("D (seed 42)");
  });
});

describe("the case desk", () => {
  it("never offers F2", async () => {
    stubHealth(HEALTH_WITH_F2);
    render(
      <MemoryRouter initialEntries={["/case"]}>
        <App />
      </MemoryRouter>,
    );
    await waitFor(() => expect(document.querySelector("main")).not.toBeNull());
    await new Promise((r) => setTimeout(r, 100));
    expect(screen.queryByTestId("voice-model-select")).toBeNull();
    expect(document.body.textContent ?? "").not.toContain("F2 (trained on compressed audio)");
  });
});
