// Revision 2: the six worlds and the 404. Honesty, behaviour and accessibility checks that must hold on every route.
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "../App";
import { CaseDesk } from "../pages/Landing";
import { terrainHeights, synthSpeech } from "../lib/terrain";
import { buildBeads, fillBars, fillNeighbours } from "../world/graphLayout";
import { CHAPTERS, buildEdges, buildMeshLayout } from "../world/layouts";
import { live } from "../world/store";
import { WORLDS, worldFor } from "../world/worlds";

const ROUTES = ["/", "/graph", "/voice", "/case", "/evidence", "/about", "/not-a-page"] as const;
const RESULT_NUMBER = /\d\.\d|\d+ in 100|\d{1,3},\d{3}|\d{5,}|(?!(?:19|20)\d\d)\d{4}/;
const JARGON = /PR-AUC|\bEER\b|\bFPR\b|XGBoost|\bGNN\b|wav2vec|temporal split|attack-disjoint|calibrat/i;
const TOOL_WORDS = /Claude|Anthropic|ChatGPT|OpenAI|Copilot|generated with|AI-generated|AI generated|built with (three|react|ai)/i;

function reduceMotion(on: boolean) {
  window.matchMedia = ((query: string) => ({
    matches: on && query.includes("prefers-reduced-motion"),
    media: query,
    addEventListener() {},
    removeEventListener() {},
  })) as unknown as typeof window.matchMedia;
}

async function renderRoute(path: string, engineers = false) {
  sessionStorage.clear();
  if (engineers) for (const page of ["landing", "graph", "voice", "case", "evidence", "about"]) sessionStorage.setItem(`tm-engineers-${page}`, "1");
  const view = render(
    <MemoryRouter initialEntries={[path]}>
      <App />
    </MemoryRouter>,
  );
  await screen.findByRole("heading", { level: 1 });
  return view;
}

beforeEach(() => {
  vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("no network in tests")));
  reduceMotion(false);
});
afterEach(() => {
  document.documentElement.removeAttribute("data-world");
  live.data = {};
});

describe("every route", () => {
  it.each(ROUTES)("%s has one h1, sets its world and its title, and has a skip link and a main landmark", async (path) => {
    const { container } = await renderRoute(path);
    expect(container.querySelectorAll("h1")).toHaveLength(1);
    const world = worldFor(path);
    await waitFor(() => expect(document.documentElement.dataset.world).toBe(world));
    expect(document.title).toBe(WORLDS[world].title);
    expect(container.querySelector("a[href='#main']")).not.toBeNull();
    expect(container.querySelector("main#main")).not.toBeNull();
  });

  it.each(ROUTES)("%s: nothing drawn on a canvas is exposed to assistive technology, and the stage holds no text", async (path) => {
    const { container } = await renderRoute(path);
    for (const el of container.querySelectorAll("canvas, #stage, [data-poster]")) expect(el.closest("[aria-hidden=true]") ?? el.getAttribute("aria-hidden")).toBeTruthy();
    expect(container.querySelector("#stage")?.textContent ?? "").toBe("");
  });

  it.each(ROUTES)("%s renders result numbers only through the data module or the live API (plain and engineer layers)", async (path) => {
    for (const engineers of [false, true]) {
      const { container, unmount } = await renderRoute(path, engineers);
      const copy = container.cloneNode(true) as HTMLElement;
      copy.querySelectorAll("[data-evidence], [data-live], [data-axis], [data-id], [data-citations], footer, option, code").forEach((n) => n.remove());
      const stray = (copy.textContent ?? "").match(new RegExp(RESULT_NUMBER, "g"));
      expect(stray, `numbers typed outside the data module on ${path} (engineers ${engineers})`).toBeNull();
      unmount();
    }
  });

  it.each(ROUTES)("%s says nothing about how it was built or which tools were used", async (path) => {
    const { container } = await renderRoute(path, true);
    expect(container.textContent ?? "").not.toMatch(TOOL_WORDS);
    expect(container.innerHTML).not.toMatch(TOOL_WORDS);
  });

  it.each(["/case", "/evidence", "/about", "/not-a-page"])("%s keeps jargon out of the plain layer", async (path) => {
    const { container } = await renderRoute(path);
    expect(container.querySelector("main")!.textContent ?? "").not.toMatch(JARGON);
  });
});

describe("honesty rules", () => {
  it("the Visual Engine appears only as a dashed row with no number, on every page that mentions it", async () => {
    for (const path of ROUTES) {
      const { container, unmount } = await renderRoute(path, true);
      // a diagram and its caption describe the empty slot in words; every other mention must sit in a dashed box
      const mentions = [...container.querySelectorAll("main *")].filter((n) => n.children.length === 0 && /Visual (Engine|slot)/.test(n.textContent ?? "") && !n.closest("svg, figcaption"));
      for (const m of mentions) {
        const box = m.closest(".planned");
        expect(box, `Visual Engine mention outside a dashed box on ${path}: ${m.textContent}`).not.toBeNull();
        expect(box!.textContent).not.toMatch(/\d\.\d/);
      }
      unmount();
    }
  });

  it("the Home page says the Visual Engine is designed, not built, with no results", async () => {
    const { container } = await renderRoute("/");
    expect(container.textContent).toContain("Designed, not built. No results yet.");
  });

  it("the combined score is shown as a mechanism: placeholder weights, no measured accuracy", async () => {
    const { container } = await renderRoute("/case");
    const text = container.querySelector("main")!.textContent ?? "";
    expect(text).toMatch(/The weights are placeholders/);
    expect(text).toMatch(/no measured accuracy/);
    expect(container.querySelector('[data-evidence="fusion.weight"]')).not.toBeNull();
  });

  it("the known compressed-audio weakness is on the Voice and Evidence pages, with the codec counts from the data module", async () => {
    for (const path of ["/voice", "/evidence"]) {
      const { container, unmount } = await renderRoute(path);
      if (path === "/evidence") fireEvent.click(screen.getAllByText("Pull the drawer and turn the card").at(-1)!);
      expect(container.textContent).toMatch(/AAC/);
      expect(container.querySelector('[data-evidence="guard.aac.warned"]')).not.toBeNull();
      expect(container.querySelector('[data-evidence="guard.aac.n"]')).not.toBeNull();
      unmount();
    }
  });

  it("Phase 2D is sealed until opened, and opens as 'reduced, not removed' with every figure from the data module", async () => {
    const { container } = await renderRoute("/evidence");
    expect(screen.queryByTestId("envelope-open")).toBeNull();
    fireEvent.click(screen.getByText("Break the seal and read the result"));
    const open = screen.getByTestId("envelope-open");
    expect(open.textContent).toMatch(/Reduced, not removed/);
    expect(open.textContent).toMatch(/not real phone calls/);
    expect(open.textContent).toMatch(/still serves the original voice model/);
    expect(within(open).getAllByText(/\d/).length).toBeGreaterThan(5);
    const copy = container.cloneNode(true) as HTMLElement;
    copy.querySelectorAll("[data-evidence], [data-citations], footer").forEach((n) => n.remove());
    expect((copy.textContent ?? "").match(new RegExp(RESULT_NUMBER, "g"))).toBeNull();
  });

  it("the untested things are drawn as dashed drawers", async () => {
    const { container } = await renderRoute("/evidence");
    const items = [...container.querySelectorAll("#untested li")];
    expect(items.length).toBeGreaterThan(5);
    for (const li of items) expect(li.className).toContain("planned");
  });

  it("the owner's open items are marked in the source and shown as plain 'coming soon', never as invented content", async () => {
    const { container } = await renderRoute("/about");
    expect(container.querySelector("[data-todo]")).not.toBeNull();
    expect(container.textContent).toMatch(/Resume: coming soon/);
    expect(container.textContent).not.toMatch(/\+?\d{10}|CGPA/i);
  });
});

describe("navigation and accessibility", () => {
  it("moves focus to the new page's h1 and updates the title on a route change (reduced motion: no curtain)", async () => {
    reduceMotion(true);
    const { container } = await renderRoute("/");
    const link = within(container.querySelector("nav[aria-label=Pages]") as HTMLElement).getByText("Voice Engine");
    fireEvent.click(link);
    await waitFor(() => expect(document.title).toBe(WORLDS.voice.title));
    await waitFor(() => expect(document.activeElement?.tagName).toBe("H1"));
    expect(document.activeElement?.id).toBe("voice-h");
    expect(container.querySelector(".curtain")).toBeNull();
  });

  it("runs an exit curtain between pages when motion is allowed, then swaps the page", async () => {
    const { container } = await renderRoute("/");
    fireEvent.click(within(container.querySelector("nav[aria-label=Pages]") as HTMLElement).getByText("Case desk"));
    expect(container.querySelector(".curtain")).not.toBeNull();
    await waitFor(() => expect(document.title).toBe(WORLDS.case.title), { timeout: 3000 });
  });

  it("the mobile menu opens, traps focus, and Escape closes it and returns focus to its button", async () => {
    const { container } = await renderRoute("/");
    const button = within(container.querySelector("header") as HTMLElement).getByText("Menu");
    fireEvent.click(button);
    const menu = container.querySelector("#mobile-menu") as HTMLElement;
    expect(menu).not.toBeNull();
    expect(button.getAttribute("aria-expanded")).toBe("true");
    const first = menu.querySelector("a") as HTMLElement;
    expect(document.activeElement).toBe(first);
    const links = menu.querySelectorAll("a");
    (links[links.length - 1] as HTMLElement).focus();
    fireEvent.keyDown(document, { key: "Tab" });
    expect(document.activeElement).toBe(button); // Tab from the last link wraps to the toggle, still inside the trap
    fireEvent.keyDown(document, { key: "Escape" });
    expect(container.querySelector("#mobile-menu")).toBeNull();
    expect(document.activeElement).toBe(button);
  });

  it("the theme toggle switches the theme, remembers the choice, and never rebuilds the page", async () => {
    document.documentElement.dataset.theme = "light";
    const { container } = await renderRoute("/graph");
    const main = container.querySelector("main");
    fireEvent.click(screen.getByText("Switch to dark theme"));
    expect(document.documentElement.dataset.theme).toBe("dark");
    expect(localStorage.getItem("tm-theme")).toBe("dark");
    expect(container.querySelector("main")).toBe(main);
    fireEvent.click(screen.getByText("Switch to light theme"));
    expect(localStorage.getItem("tm-theme")).toBe("light");
  });

  it("the footer carries the thread to the next page", async () => {
    const { container } = await renderRoute("/voice");
    expect(container.querySelector("footer a[href='/case']")?.textContent).toContain("The Balance");
  });
});

describe("the data module and the pages agree", () => {
  it("every World route is registered once and has a distinct idea and title", () => {
    const titles = new Set(Object.values(WORLDS).map((w) => w.title));
    expect(titles.size).toBe(Object.keys(WORLDS).length);
    expect(worldFor("/graph/")).toBe("graph");
    expect(worldFor("/nope")).toBe("lost");
  });

  it("index.html carries no tool or assistant wording", () => {
    const html = readFileSync(join(process.cwd(), "index.html"), "utf-8");
    expect(html).not.toMatch(TOOL_WORDS);
  });
});

describe("recording mode hides what belongs to a real transaction in every 3D view", () => {
  const health = { graph: { ready: true, model: "xgboost_tier1_tier2" }, voice: { ready: false }, neo4j: { connected: false }, recording_mode: true };
  const graph = {
    node_id: 154134, tx_id: 1, time_step: 40, score: 0.9, verdict: "high", model: "xgboost_tier1_tier2",
    thresholds: { medium: 0.1, high: 0.5 }, ground_truth: "illicit", split: "test", in_sample: false,
  };
  const json = (body: unknown) => Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(body) });

  it("the balance receives no transaction score and the register shows no scores or verdicts", async () => {
    const fetchMock = vi.fn((url: string) => (url.includes("/health") ? json(health) : url.includes("/graph/score") ? json(graph) : Promise.reject(new Error("unexpected"))));
    vi.stubGlobal("fetch", fetchMock);
    const seen: boolean[] = [];
    render(
      <MemoryRouter>
        <CaseDesk onScored={(_g, _f, masked) => seen.push(masked)} />
      </MemoryRouter>,
    );
    await waitFor(() => expect(fetchMock).toHaveBeenCalled());
    await act(async () => {
      await new Promise((r) => setTimeout(r, 50)); // let the health answer land: recording mode is read from it
    });
    fireEvent.click(screen.getByText("Score this case"));
    await waitFor(() => expect(seen.length).toBeGreaterThan(0));
    expect(seen[0]).toBe(true);
    await waitFor(() => expect((live.data.case as { masked: boolean } | undefined)?.masked).toBe(true));
    const data = live.data.case as { graph: number | null; masked: boolean } | undefined;
    expect(data?.graph).toBeNull();
    expect(data?.masked).toBe(true);
  });
});

describe("scene layouts", () => {
  it("the Home mesh is deterministic, finite, and ready for every chapter", () => {
    const a = buildMeshLayout(300), b = buildMeshLayout(300);
    expect(a.targets).toHaveLength(CHAPTERS);
    for (let c = 0; c < CHAPTERS; c++) {
      expect(a.targets[c]).toEqual(b.targets[c]);
      expect(a.targets[c].every(Number.isFinite)).toBe(true);
    }
    expect(new Set(a.kind)).toEqual(new Set([0, 1, 2, 3]));
    const edges = buildEdges(a);
    expect(edges.length % 2).toBe(0);
    expect(edges.length).toBeGreaterThan(a.count / 2);
    expect(Math.max(...edges)).toBeLessThan(a.count);
  });

  it("the Graph beads stack bars in proportion to their values and lay neighbours on rings", () => {
    const b = buildBeads(400);
    fillBars(b, [{ value: 0.75, tint: 1 }, { value: 0.5, tint: 0 }]);
    let shown = 0;
    for (let i = 0; i < b.count; i++) shown += b.vis[i * 4 + 2];
    expect(shown).toBe(Math.round(0.75 * 56) + Math.round(0.5 * 56));
    fillNeighbours(b, { flagged: false }, [{ hops: 1, flagged: false }, { hops: 2, flagged: true }]);
    expect(b.vis[3]).toBe(1);
    expect(b.vis[1 * 4 + 3]).toBe(1);
    expect(b.vis[3 * 4 + 3]).toBe(0);
    fillNeighbours(b, null, []);
    expect(b.vis[3]).toBeCloseTo(0.3);
  });

  it("the Voice terrain puts a pure tone on the right frequency row and stays flat for silence", () => {
    const rate = 16000, n = rate;
    const tone = Float32Array.from({ length: n }, (_, i) => 0.5 * Math.sin((2 * Math.PI * 2000 * i) / rate));
    const cols = 32, rows = 32;
    const h = terrainHeights(tone, cols, rows);
    const rowMean = Array.from({ length: rows }, (_, r) => h.slice(r * cols, (r + 1) * cols).reduce((s, v) => s + v, 0) / cols);
    const peak = rowMean.indexOf(Math.max(...rowMean));
    expect(Math.abs(peak - Math.round((2000 / 8000) * rows))).toBeLessThanOrEqual(1);
    expect(h.every((v) => v >= 0 && v <= 1)).toBe(true);
    expect(terrainHeights(new Float32Array(n), cols, rows).every((v) => v === 0)).toBe(true);
    const s = synthSpeech(0.5);
    expect(s.every(Number.isFinite)).toBe(true);
    expect(synthSpeech(0.5)).toEqual(s);
  });
});
