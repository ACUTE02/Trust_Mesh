import { useState } from "react";
import { DemoState, ModelInUse, Score, VerdictTag, errorText, useHealth } from "../components/caseParts";
import { ClipInput, ClipView } from "../components/clip";
import { DotPlot, type PlotRow } from "../components/DotPlot";
import { CaveatNote, EvidenceRow, Live, Num, Section } from "../components/evidence";
import { TwoScoresInstrument } from "../components/TwoScoresInstrument";
import { api, type VoiceScore } from "../lib/api";
import type { Clip } from "../lib/audio";

const SILENCE_LADDER: PlotRow[] = [
  { id: "voice.2b.untrimmed", label: "CNN on untrimmed audio" },
  { id: "voice.2b.zerosOriginal", label: "Silence replaced by zeros, kept at its original length" },
  { id: "voice.2b.silenceOnly", label: "A classifier given only a short window of genuine silence" },
  { id: "voice.2b.A", label: "CNN on trimmed audio (the honest baseline)", reference: true },
  { id: "voice.p2.timingProbe", label: "Leading-silence duration alone, no model (Phase 2 sample, a different set of clips)" },
];

const PROGRESSION: PlotRow[] = [
  { id: "voice.p2.trimmed", label: "CNN, first run, few bonafide clips" },
  { id: "voice.2b.A", label: "CNN with the full bonafide set" },
  { id: "voice.2b.B", label: "CNN with augmentation" },
  { id: "voice.2b.C", label: "Model C: frozen XLS-R encoder, mean and std pooling" },
  { id: "voice.2c.D", label: "Model D: frozen XLS-R encoder, mean-only pooling (chosen)", reference: true },
];

const ATTACK_GAP: PlotRow[] = [
  { id: "voice.gap.A", label: "Plain CNN (interval excludes zero)" },
  { id: "voice.gap.B", label: "CNN with augmentation (interval excludes zero)" },
  { id: "voice.gap.C", label: "Model C (not significant)" },
  { id: "voice.gap.D", label: "Model D (not significant)" },
];

const IN_THE_WILD: PlotRow[] = [
  { id: "voice.itw.A", label: "Plain CNN" },
  { id: "voice.itw.C", label: "Model C" },
  { id: "voice.itw.D", label: "Model D" },
];

function VoiceDesk() {
  const health = useHealth();
  const [clip, setClip] = useState<Clip | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<VoiceScore | null>(null);

  async function score() {
    if (!clip) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      setResult(await api.voiceScore(clip.wav, clip.name));
    } catch (e) {
      setError(errorText(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="grid gap-6">
      <ClipInput
        clip={clip}
        health={health}
        onClip={(c) => {
          setClip(c);
          setResult(null);
          setError(null);
        }}
      />
      {clip && <ClipView clip={clip} leadSilence={result?.lead_silence_s} trailSilence={result?.trail_silence_s} />}
      <div>
        <button type="button" className="btn" disabled={!clip || loading} onClick={() => void score()}>
          Score this clip
        </button>
      </div>
      <DemoState loading={loading} error={error} waitText="Scoring the clip. The encoder runs on a CPU, so this takes a few seconds." />
      {!clip && !loading && <p className="m-0 text-sm text-ink-soft">No clip yet. Choose a file or record a few seconds of speech.</p>}
      {health === "down" && (
        <p className="m-0 text-sm text-ink-soft">
          Scoring needs the scoring service. Start the stack with docker compose up; you can still load a clip and look at it.
        </p>
      )}
      {result && (
        <div data-testid="voice-result" className="rule-t pt-5">
          <div className="text-xs text-ink-soft">Spoof score (0 = sounds bonafide, 1 = sounds spoofed)</div>
          <div className="mt-1 flex items-baseline gap-3">
            <span className="text-xl font-semibold" data-testid="voice-desk-score">
              <Score value={result.score} />
            </span>
            <VerdictTag verdict={result.verdict} />
          </div>
          <p className="measure mt-3 mb-0 text-sm">
            <ModelInUse model={result.model} requested={result.model_requested} fallback={result.fallback_used} />. Medium risk
            starts at <Live>{result.thresholds.medium.toFixed(4)}</Live> and high risk at{" "}
            <Live>{result.thresholds.high.toFixed(4)}</Live>. These thresholds were set on the same test scores the model was
            evaluated on, so they are not an out-of-sample error rate.
          </p>
          <p className="measure mt-2 mb-0 text-sm text-ink-soft">
            Silence at both ends is trimmed before scoring, because of the shortcut finding above. The clip is then cropped or
            repeated to <Num id="voice.scoredSeconds" />: this clip had <Live>{result.speech_s.toFixed(1)}</Live> seconds of
            speech after trimming.
          </p>
        </div>
      )}
      <CaveatNote>A research model on a research dataset. Not for real identity decisions.</CaveatNote>
    </div>
  );
}

export function VoicePage() {
  return (
    <>
      <section className="grid items-start gap-x-12 gap-y-10 pt-14 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]" aria-labelledby="voice-h">
        <div>
          <h1 id="voice-h" style={{ fontSize: "var(--text-hero)" }}>
            The first voice model scored <Num id="voice.p2.untrimmed" />. It was reading the silence.
          </h1>
          <p className="measure mt-6 mb-0">
            ASVspoof 2019 LA, evaluation partition. Phase 2B uses a <Num id="voice.clips" />-clip subset with a
            speaker-disjoint split and <Num id="voice.trainBonafide" /> bonafide training clips.
          </p>
          <div className="mt-4">
            <CaveatNote title="Not the official protocol.">
              The split is this project's own, so these numbers cannot be compared with published ASVspoof results.
            </CaveatNote>
          </div>
        </div>
        <TwoScoresInstrument
          title="One CNN on the same clips, with and without the silence at the ends."
          hue="spectrum"
          axisLabel="Equal error rate (lower is better)"
          axisMax={{ at: 0.3, label: "0.3" }}
          ticks={[
            { at: 0, label: "0" },
            { at: 0.1, label: "0.1" },
            { at: 0.2, label: "0.2" },
            { at: 0.3, label: "0.3" },
          ]}
          lines={[
            "Same model, same clips. Only the silence changed.",
            "In this dataset, bonafide clips start with more silence than spoofed ones. A model can separate them without listening to the voice at all.",
          ]}
          first={{ label: "Silence kept", id: "voice.p2.untrimmed" }}
          second={{ label: "Silence removed", id: "voice.p2.trimmed" }}
        />
      </section>

      <Section title="Where the shortcut lives" id="silence">
        <DotPlot
          caption="Equal error rate under each silence condition, lowest first. Hatched rows are shortcuts, not results."
          rows={SILENCE_LADDER}
          axisMax={0.3}
          axisLabel="equal error rate, lower is better"
          hue="spectrum"
        />
        <p className="measure mt-4 mb-0">
          Duration and position carry most of the shortcut: keeping only the length of the silence, with its content zeroed,
          still scores <Num id="voice.2b.zerosOriginal" />. The content of the silence carries a smaller, independent signal (
          <Num id="voice.2b.silenceOnly" /> from <Num id="voice.2b.silenceOnlySeconds" /> of it). This experiment does not say
          which acoustic property.
        </p>
      </Section>

      <Section title="From the first model to the chosen one" id="progression">
        <DotPlot
          caption="Equal error rate on trimmed audio, in the order the models were built. A bracket is a three-seed spread."
          rows={PROGRESSION}
          axisMax={0.3}
          axisLabel="equal error rate, lower is better"
          hue="spectrum"
        />
        <p className="measure mt-4 mb-0">
          More data helped: the first run had <Num id="voice.p2.trainBonafide" /> bonafide training clips, the second{" "}
          <Num id="voice.trainBonafide" />. The frozen self-supervised front-end helped far more. Model D has half of Model
          C's head parameters (<Num id="voice.params.D" /> against <Num id="voice.params.C" />) and the two are statistically
          tied, so D was chosen.
        </p>
        <p className="measure mt-3 mb-0 text-sm text-ink-soft">
          The scoring service loads one of Model D's three seeds. That seed's own equal error rate, recomputed from its saved
          test scores, is <Num id="voice.2c.D.seed42" />: a rough single-seed figure, not the headline.
        </p>
      </Section>

      <Section title="Does it generalise?" id="generalisation">
        <EvidenceRow
          ids={["voice.gap.A", "voice.gap.B", "voice.gap.C", "voice.gap.D"]}
          claim={
            <>
              With <Num id="voice.heldOutAttacks" /> attack systems held out of training, the plain CNN's error rate rose by{" "}
              <Num id="voice.gap.A" />. Models C and D moved by <Num id="voice.gap.C" /> and <Num id="voice.gap.D" />: not
              significant.
            </>
          }
          more="Held-out systems: A07, A12, A14 and A15. The value is the equal error rate on those attacks when unseen, minus the rate when seen. Brackets are 95% intervals."
        >
          <DotPlot
            caption="Change in equal error rate from seen to unseen attacks. Closer to zero is better."
            rows={ATTACK_GAP}
            axisMax={0.3}
            axisLabel="change in equal error rate"
            hue="spectrum"
            signedAxis
          />
        </EvidenceRow>
        <EvidenceRow
          ids={["voice.itw.A", "voice.itw.C", "voice.itw.D", "voice.itw.total"]}
          claim={
            <>
              On a different dataset, the plain CNN is worse than a coin flip at <Num id="voice.itw.A" />. Models C and D
              hold at <Num id="voice.itw.C" /> and <Num id="voice.itw.D" />.
            </>
          }
          more={
            <>
              In-the-Wild, <Num id="voice.itw.n" /> bonafide and <Num id="voice.itw.n" /> spoof clips. The sample is small: the
              intervals are wide, and C and D do not differ significantly.
            </>
          }
        >
          <DotPlot
            caption="Equal error rate on In-the-Wild. Brackets are 95% intervals."
            rows={IN_THE_WILD}
            axisMax={1}
            axisLabel="equal error rate, lower is better"
            hue="spectrum"
          />
        </EvidenceRow>
      </Section>

      <Section title="Case desk: score a clip" id="desk">
        <p className="measure mt-0 mb-6">
          Live against the local stack. Bring your own recording: no dataset clip is bundled with this site.
        </p>
        <VoiceDesk />
      </Section>

      <Section title="Limits" id="limits">
        <ul className="measure m-0 grid list-none gap-3 p-0">
          <li className="rule-t pt-3">Only a subset of the LA evaluation partition, on this project's own split.</li>
          <li className="rule-t pt-3">
            The out-of-domain check is <Num id="voice.itw.total" /> clips from one other dataset.
          </li>
          <li className="rule-t pt-3">
            Models E and F are implemented but never completed a run on real data, because of disk limits and bugs on
            the training service. They are untested, not disproven.
          </li>
          <li className="rule-t pt-3">
            The live verdict thresholds were set on the test scores themselves. They show how the bands were drawn, not how
            often a real caller would be wrongly flagged.
          </li>
        </ul>
      </Section>
    </>
  );
}
