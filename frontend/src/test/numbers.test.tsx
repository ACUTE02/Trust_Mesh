// Build-time honesty checks. `npm run build` runs these first, so a number without a source fails the build.
import { fireEvent, render, screen } from "@testing-library/react";
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import App from "../App";
import { CaseCard, ModelInUse } from "../components/caseParts";
import { Num } from "../components/evidence";
import { allEvidence, formatValue } from "../data/evidence";
import { TechView } from "../components/TechView";
import { SplitInstrument } from "../pages/Landing";

// A "result-looking" number: any decimal, an "N in 100" rate, a thousands separator, or four or more digits.
const RESULT_NUMBER = /\d\.\d|\d+ in 100|\d{1,3},\d{3}|\d{5,}|(?!(?:19|20)\d\d)\d{4}/; // years are allowed

beforeEach(() => {
  // No API in tests: every request fails, which also exercises the "service didn't answer" states.
  vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("no network in tests")));
});

function renderPage(path: string, engineers = false) {
  sessionStorage.clear();
  if (engineers) for (const page of ["landing", "graph", "voice"]) sessionStorage.setItem(`tm-engineers-${page}`, "1");
  return render(
    <MemoryRouter initialEntries={[path]}>
      <App />
    </MemoryRouter>,
  );
}

describe("the data module", () => {
  it("gives every entry a value, what it measures, a run date, a phase and a source", () => {
    for (const [id, e] of allEvidence()) {
      expect(e.measures, id).toBeTruthy();
      expect(e.runDate, id).toMatch(/^\d{4}-\d{2}-\d{2}$/);
      expect(e.phase, id).toBeTruthy();
      expect(e.source?.file, id).toBeTruthy();
      expect(e.source.key ?? e.source.text ?? e.source.derivedFrom, id).toBeTruthy();
      if (e.derived) expect(e.source.formula, `${id} is derived and needs a formula`).toBeTruthy();
    }
  });
});

describe("no number without a source", () => {
  it.each([
    ["/", false], ["/graph", false], ["/voice", false], ["/", true], ["/graph", true], ["/voice", true],
  ] as const)("page %s (engineer detail open: %s) renders result numbers only through the data module", (path, engineers) => {
    const { container } = renderPage(path, engineers);
    expect(container.querySelectorAll("[data-layer=engineers]").length > 0).toBe(engineers);
    expect(container.querySelectorAll("[data-evidence]").length).toBeGreaterThan(5);
    const copy = container.cloneNode(true) as HTMLElement;
    // Remove what is allowed to carry numbers: data-module values, live API values, axis furniture,
    // example transaction identifiers, and the citations table (publication years and DOIs).
    copy.querySelectorAll("[data-evidence], [data-live], [data-axis], [data-id], footer, option").forEach((n) => n.remove());
    const stray = (copy.textContent ?? "").match(new RegExp(RESULT_NUMBER, "g"));
    expect(stray, `numbers typed outside the data module on ${path}`).toBeNull();
  });

  it("no component or page source types a result number inline", () => {
    const files: string[] = [];
    const walk = (dir: string) => {
      for (const name of readdirSync(dir)) {
        const full = join(dir, name);
        if (statSync(full).isDirectory()) {
          if (!["data", "test", "lib"].includes(name)) walk(full);
        } else if (full.endsWith(".tsx")) files.push(full);
      }
    };
    walk(join(process.cwd(), "src"));
    expect(files.length).toBeGreaterThan(5);
    for (const file of files) {
      const text = readFileSync(file, "utf-8")
        .replace(/\/\/.*$/gm, "")
        .replace(/\/\*[\s\S]*?\*\//g, "")
        .replace(/className=(\{`[^`]*`\}|"[^"]*")/g, "") // utility classes like w-[4.25rem]
        .replace(/style=\{\{[^}]*\}\}/g, "");
      const hits = text.match(/\b0\.\d{3,4}\b|\b\d{1,3},\d{3}\b|\b\d{5,}\b/g);
      expect(hits, `result-looking literal in ${file}`).toBeNull();
    }
  });

  it("renders a count from the data module with separators", () => {
    const { container } = render(<Num id={"graph.nodes"} />);
    expect(container.textContent).toBe("203,769");
  });

  it("an entry that is not yet run reads 'not yet run' and does not crash", () => {
    const pending = { value: null, decimals: 4, measures: "x", runDate: "2026-10-08", phase: "P2C" as const, source: { file: "f", text: "t" } };
    expect(formatValue(pending)).toBe("not yet run");
  });
});

describe("the hero instrument", () => {
  it("shows the random-split score first, then the time-split score and the difference", () => {
    render(<SplitInstrument />);
    const readout = () => document.querySelector("[aria-live] [data-evidence]")?.textContent;
    expect(readout()).toBe("0.7730");
    expect(screen.queryByText(/a difference of/)).toBeNull();
    fireEvent.click(screen.getByLabelText("Time split"));
    expect(readout()).toBe("0.5115");
    expect(screen.getByText(/a difference of/).textContent).toContain("0.2616");
    expect(screen.getByText("Same model, same settings. Only the split changed.")).toBeTruthy();
  });

  it("is a keyboard-operable radio group", () => {
    render(<SplitInstrument />);
    const radios = screen.getAllByRole("radio") as HTMLInputElement[];
    expect(radios).toHaveLength(2);
    expect(radios[0].checked).toBe(true);
    expect(radios[0].name).toBe(radios[1].name);
  });
});

const GRAPH = {
  node_id: 1, tx_id: 1, time_step: 40, score: 0.9, verdict: "high" as const, model: "xgboost_tier1_tier2",
  thresholds: { medium: 0.1, high: 0.5 }, ground_truth: "illicit" as const, split: "test" as const, in_sample: false,
};
const VOICE = {
  clip: "mine.wav", score: 0.2, verdict: "medium" as const, model: "D", model_requested: "D", fallback_used: false,
  thresholds: { medium: 0.1, high: 0.5, source: "" }, duration_s: 3, speech_s: 2,
};

describe("the case card", () => {
  it("shows empty fields as empty, never as zero, and keeps the visual slot dashed", () => {
    render(<CaseCard graph={GRAPH} voice={null} combined={null} />);
    expect(screen.getByTestId("graph-score").textContent).toBe("0.9000");
    expect(screen.getByTestId("voice-score").textContent).toBe("empty");
    expect(screen.getByTestId("combined-score").textContent).toBe("empty");
    expect(screen.getByText(/not connected/).closest(".planned")).not.toBeNull();
    expect(screen.getByTestId("case-card").textContent).not.toMatch(/0\.0000/);
  });

  it("always carries the no-measured-accuracy note beside the combined score", () => {
    render(<CaseCard graph={GRAPH} voice={VOICE} combined={{ combined_score: 0.55, combined_verdict: "medium", weights: { graph: 0.5, voice: 0.5 } }} />);
    expect(screen.getByTestId("combined-score").textContent).toBe("0.5500");
    expect(screen.getByRole("note").textContent).toContain("It has no measured accuracy");
    expect(screen.getByRole("note").textContent).toContain("simply averaged (0.5 and 0.5)");
  });
});

describe("ModelInUse", () => {
  it("shows model D when the API reports D", () => {
    render(<ModelInUse model="D" requested="D" fallback={false} weightsFile="phase2c_D_ssl_mean_only_P1_s42.pt" />);
    const text = screen.getByTestId("model-in-use").textContent ?? "";
    expect(text).toContain("Voice model D");
    expect(text).toContain("phase2c_D_ssl_mean_only_P1_s42.pt");
    expect(text).not.toContain("Fallback");
  });

  it("says so when the API reports a fallback to model C", () => {
    render(<ModelInUse model="C" requested="D" fallback />);
    const text = screen.getByTestId("model-in-use").textContent ?? "";
    expect(text).toContain("Voice model C");
    expect(text).toContain("Fallback in use: model D was requested");
  });

  it("does not invent a name when the API reports none", () => {
    const { container } = render(<ModelInUse model={undefined} />);
    expect(container.textContent).toContain("voice model unknown");
  });
});

describe("pages without the API", () => {
  it("the landing page explains how to start the stack instead of going blank", async () => {
    renderPage("/");
    expect(await screen.findByText(/The scoring service is not running/)).toBeTruthy();
    expect(screen.getAllByText(/Designed, not built. No results yet./).length).toBeGreaterThan(0);
    fireEvent.click(screen.getByText("Score this case"));
    expect((await screen.findByRole("alert")).textContent).toContain("Start the stack with docker compose up");
  });

  it("the visual engine appears only as a dashed row, with no number", () => {
    const { container } = renderPage("/");
    const rows = [...container.querySelectorAll(".planned")].filter((n) => /Visual/.test(n.textContent ?? ""));
    expect(rows.length).toBeGreaterThan(0);
    for (const row of rows) expect(row.textContent).not.toMatch(/\d\.\d/);
  });
});

// Revision 1: easy language by default, the technical terms only under "For engineers".
const JARGON = /PR-AUC|\bEER\b|\bFPR\b|XGBoost|\bGNN\b|wav2vec|temporal split|attack-disjoint|calibrat/i;

describe("two layers", () => {
  it.each(["/", "/graph", "/voice"])("page %s uses no jargon until the engineer detail is opened", (path) => {
    const { container } = renderPage(path);
    const main = container.querySelector("main")!;
    expect(main.querySelector("[data-layer=engineers]")).toBeNull();
    const found = (main.textContent ?? "").match(new RegExp(JARGON, "gi"));
    expect(found, `jargon in the plain layer of ${path}`).toBeNull();
  });

  it("one toggle opens every engineer block on the page, and the choice is kept for the session", () => {
    const { container, unmount } = renderPage("/graph");
    fireEvent.click(screen.getByText("Show the detail for engineers"));
    expect(container.querySelectorAll("[data-layer=engineers]").length).toBeGreaterThan(4);
    expect(container.querySelector("main")!.textContent).toMatch(/PR-AUC on the illicit class/);
    expect(container.querySelector("main")!.textContent).toMatch(/Recall @ 5% FPR \(n = 686\)/);
    expect(sessionStorage.getItem("tm-engineers-graph")).toBe("1");
    unmount();
    const again = render(
      <MemoryRouter initialEntries={["/graph"]}>
        <App />
      </MemoryRouter>,
    );
    expect(again.container.querySelectorAll("[data-layer=engineers]").length).toBeGreaterThan(4);
    fireEvent.click(screen.getByText("Hide the detail for engineers"));
    expect(again.container.querySelector("[data-layer=engineers]")).toBeNull();
  });

  it("plain figures are derived entries with a formula, shown through the data module", () => {
    const derived = allEvidence().filter(([, e]) => e.derived);
    expect(derived.length).toBeGreaterThan(5);
    for (const [id, e] of derived) {
      expect(e.source.derivedFrom, id).toBeTruthy();
      expect(e.source.formula, id).toBeTruthy();
    }
    const { container } = renderPage("/voice");
    expect(container.querySelector('[data-evidence="voice.2c.D.per100"]')?.textContent).toBe("1.7");
  });
});

describe("the technical view", () => {
  const HEALTH = {
    graph: { ready: true, model: "xgboost_tier1_tier2" },
    voice: { ready: true, serving_default: "D", requested_default: "D", fallback_active: false, weights: { D: "phase2c_D_ssl_mean_only_P1_s42.pt" } },
    neo4j: { connected: true },
  };

  it("is closed by default and shows only what the service returned", () => {
    render(<TechView calls={[{ method: "GET", path: "/graph/score/7", status: 200, ms: 12.4, body: { score: 0.5 } }]} health={HEALTH} cypher={{ neighbors: "MATCH (n) RETURN n" }} />);
    expect(screen.queryByTestId("tech-view")).toBeNull();
    fireEvent.click(screen.getByText("Show the technical view"));
    const text = screen.getByTestId("tech-view").textContent ?? "";
    expect(text).toContain("GET /graph/score/7");
    expect(text).toContain("round trip 12 ms");
    expect(text).toContain("weights phase2c_D_ssl_mean_only_P1_s42.pt, seed 42");
    expect(text).toContain("MATCH (n) RETURN n");
  });

  it("says so when nothing has been requested or the service is down", () => {
    render(<TechView calls={[]} health="down" />);
    fireEvent.click(screen.getByText("Show the technical view"));
    const text = screen.getByTestId("tech-view").textContent ?? "";
    expect(text).toContain("The service did not answer");
    expect(text).toContain("No request yet");
  });
});
