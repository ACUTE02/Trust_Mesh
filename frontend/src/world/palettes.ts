// Scenes read their colours from the same CSS variables the page uses (index.css), so there is one palette
// per world and theme, and a theme toggle only updates colour uniforms: the canvas is never rebuilt.
import { Color } from "three";

export interface ScenePalette {
  bg: Color;
  ink: Color;
  accent: Color;
  accent2: Color;
  thread: Color;
  glow: Color;
  flag: Color;
}

export const palette: ScenePalette = {
  bg: new Color(), ink: new Color(), accent: new Color(), accent2: new Color(),
  thread: new Color(), glow: new Color(), flag: new Color(),
};

const MAP: [keyof ScenePalette, string][] = [
  ["bg", "--slab"], ["ink", "--ink"], ["accent", "--accent"], ["accent2", "--accent-2"],
  ["thread", "--thread"], ["glow", "--glow"], ["flag", "--flag"],
];

/** Copy the current CSS variables into the shared colours. Call after the theme or the world changes. */
export function refreshPalette(): void {
  const css = getComputedStyle(document.documentElement);
  for (const [key, name] of MAP) {
    const v = css.getPropertyValue(name).trim();
    if (v) palette[key].set(v);
  }
}
