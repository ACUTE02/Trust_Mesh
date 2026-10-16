// The six worlds and the 404. Colours live in index.css (one place); this file holds what is not colour:
// route, title, the one idea each world shows, and its motion character.
import { ease } from "./ticker";

export type WorldId = "home" | "graph" | "voice" | "case" | "evidence" | "about" | "lost";

export interface World {
  id: WorldId;
  path: string;
  nav: string;
  title: string; // document.title
  idea: string;
  easing: (t: number) => number;
  transitionMs: number;
}

export const WORLDS: Record<WorldId, World> = {
  home: { id: "home", path: "/", nav: "The Mesh", title: "TrustMesh: two fraud checks and the evidence for each", idea: "trust is a thread between things", easing: ease.drift, transitionMs: 900 },
  graph: { id: "graph", path: "/graph", nav: "The Ledger", title: "Graph Engine: the transaction check", idea: "a transaction is a bead on a thread", easing: ease.settle, transitionMs: 800 },
  voice: { id: "voice", path: "/voice", nav: "The Terrain", title: "Voice Engine: the voice check", idea: "a voice, as a landscape", easing: ease.elastic, transitionMs: 800 },
  case: { id: "case", path: "/case", nav: "The Balance", title: "Case desk: two scores, one decision", idea: "two scores, honestly weighed", easing: ease.damped, transitionMs: 700 },
  evidence: { id: "evidence", path: "/evidence", nav: "The Archive", title: "Evidence: what was checked and what was not", idea: "what we checked, what we could not", easing: ease.drawer, transitionMs: 800 },
  about: { id: "about", path: "/about", nav: "The Workbench", title: "About: how to rebuild TrustMesh", idea: "how to rebuild it yourself", easing: ease.draw, transitionMs: 700 },
  lost: { id: "lost", path: "*", nav: "", title: "Page not found: TrustMesh", idea: "an unlinked node", easing: ease.drift, transitionMs: 600 },
};

export function worldFor(pathname: string): WorldId {
  const p = pathname.replace(/\/+$/, "") || "/";
  const hit = (Object.values(WORLDS) as World[]).find((w) => w.path === p);
  return hit ? hit.id : "lost";
}

export const NAV_ORDER: WorldId[] = ["home", "graph", "voice", "case", "evidence", "about"];
