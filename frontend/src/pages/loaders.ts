// The pages other than the Overview are loaded when first visited, so the first page does not pay for them.
// (Tests swap this module for loaders.eager.ts, so they can render any page without waiting.)
import { lazy } from "react";

export const GraphPage = lazy(() => import("./GraphPage").then((m) => ({ default: m.GraphPage })));
export const VoicePage = lazy(() => import("./VoicePage").then((m) => ({ default: m.VoicePage })));
export const CasePage = lazy(() => import("./CasePage"));
export const EvidencePage = lazy(() => import("./EvidencePage"));
export const AboutPage = lazy(() => import("./AboutPage"));
