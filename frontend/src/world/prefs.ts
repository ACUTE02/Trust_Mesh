// What this device and this person allow. Read once; the pieces that can change (reduced motion) are re-read on change.
import { useEffect, useState } from "react";
import { live, markDirty } from "./store";

export interface MotionPrefs {
  reduced: boolean; // prefers-reduced-motion
  saveData: boolean; // data saver or prefers-reduced-data: no WebGL at all
  lowPower: boolean; // few cores or little memory: start and stay on the low tier
  isTouch: boolean; // coarse pointer: lighter scenes, no cursor effects
}

function query(q: string): boolean {
  return typeof window !== "undefined" && !!window.matchMedia?.(q).matches;
}

export function readPrefs(): MotionPrefs {
  const nav = (typeof navigator === "undefined" ? {} : navigator) as Navigator & {
    deviceMemory?: number;
    connection?: { saveData?: boolean };
  };
  return {
    reduced: query("(prefers-reduced-motion: reduce)"),
    saveData: !!nav.connection?.saveData || query("(prefers-reduced-data: reduce)"),
    lowPower: (nav.hardwareConcurrency ?? 8) <= 4 || (nav.deviceMemory ?? 8) <= 4,
    isTouch: query("(pointer: coarse)"),
  };
}

export function useMotionPrefs(): MotionPrefs {
  const [prefs, setPrefs] = useState(readPrefs);
  useEffect(() => {
    const mq = window.matchMedia?.("(prefers-reduced-motion: reduce)");
    if (!mq) return;
    const onChange = () => {
      setPrefs(readPrefs());
      live.reduced = mq.matches;
      markDirty();
    };
    mq.addEventListener?.("change", onChange);
    return () => mq.removeEventListener?.("change", onChange);
  }, []);
  return prefs;
}

/** Can this browser create a WebGL context at all? (jsdom, old phones and blocked GPUs say no.) */
export function webglAvailable(): boolean {
  try {
    const c = document.createElement("canvas");
    return !!(c.getContext("webgl2") || c.getContext("webgl"));
  } catch {
    return false;
  }
}
