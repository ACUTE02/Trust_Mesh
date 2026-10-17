import type { ReactNode } from "react";
import type { Health } from "../lib/api";
import { ForEngineers } from "./engineers";
import { Num } from "./evidence";

/** The wording shown whenever the opt-in model F2 is chosen. Easy language; no claim beyond the recorded test. */
export const F2_WORDING =
  "F2 was trained on copies of the training audio that had been compressed like phone audio. In a synthetic test it wrongly flagged far fewer genuine compressed clips, but the problem is reduced, not removed. It was not tested on real phone calls. D remains the main model.";

export type VoiceChoice = "D" | "F2";

/** F2 is offered only when the service reports its weights as loaded. Anything else (down, loading, old service) hides it. */
export function f2Available(health: Health | null | "down"): boolean {
  return !!health && health !== "down" && health.voice?.opt_in?.F2 === true;
}

/**
 * A small, secondary choice under the clip controls. D is the default and stays selected until the visitor
 * changes it. Hidden when F2 is not installed on the server.
 */
export function VoiceModelChoice({
  health,
  value,
  onChange,
}: {
  health: Health | null | "down";
  value: VoiceChoice;
  onChange: (v: VoiceChoice) => void;
}) {
  if (!f2Available(health)) return null;
  return (
    <div className="grid gap-2 text-sm" data-testid="voice-model-choice">
      <label className="flex flex-wrap items-baseline gap-x-3 gap-y-1 text-ink-soft">
        <span>Voice model</span>
        <select
          className="border border-rule bg-sheet px-2 py-1 text-ink"
          value={value}
          onChange={(e) => onChange(e.target.value as VoiceChoice)}
          data-testid="voice-model-select"
        >
          <option value="D">D (default)</option>
          <option value="F2">F2 (trained on compressed audio)</option>
        </select>
      </label>
      {value === "F2" && (
        <>
          <p className="measure m-0" data-testid="f2-wording">
            {F2_WORDING}
          </p>
          <F2Numbers />
        </>
      )}
    </div>
  );
}

function F2Numbers(): ReactNode {
  return (
    <ForEngineers title="model F2 (Phase 2D)">
      <ul className="m-0 grid list-none gap-1.5 p-0" data-testid="f2-numbers">
        <li>
          Genuine clips flagged after AAC 32 kb/s: <Num id="p2d.flag.aac32.control" /> with a head trained on clean audio only, <Num id="p2d.flag.aac32.f2" /> with F2.
        </li>
        <li>
          After AAC 32 kb/s through a 48 kHz step: <Num id="p2d.flag.via48.control" /> against <Num id="p2d.flag.via48.f2" />.
        </li>
        <li>
          Still above the clean-audio figure of <Num id="p2d.flag.clean.f2" /> on every compressed condition: reduced, not removed.
        </li>
        <li>
          Still weak on synthetic room echo (error rate <Num id="p2d.echo.f2" />), AAC 16 kb/s (<Num id="p2d.aac16.f2" />) and Opus 12 kb/s (<Num id="p2d.opus12.f2" />).
        </li>
        <li>
          Ranking was better than the clean-only head on <Num id="p2d.heldout.better" /> of <Num id="p2d.heldout.n" /> unseen codec settings.
        </li>
        <li>
          The clean-only head used for comparison was weaker than the served model D: clean error rate <Num id="p2d.clean.control" /> against <Num id="voice.2c.D" /> for D.
        </li>
        <li>
          The service scores F2 as the mean of its three seed heads (42, 43 and 44), the object the test measured, and applies
          the one clean-validation threshold recorded for F2. There is no medium band, and the line was not recalibrated on test
          scores. The test was synthetic (codecs applied to research recordings), not real phone calls.
        </li>
      </ul>
    </ForEngineers>
  );
}
