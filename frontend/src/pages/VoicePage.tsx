import { useState } from "react";
import { DemoState, ModelInUse, Score, VerdictTag, errorText, useHealth } from "../components/caseParts";
import { ClipInput, ClipView } from "../components/clip";
import { DotPlot, type PlotRow } from "../components/DotPlot";
import { EngineerLayer, EngineerToggle, ForEngineers, Glossary, Meaning, VOICE_SCORE_MEANING } from "../components/engineers";
import { CaveatNote, EvidenceRow, Live, Num, Section } from "../components/evidence";
import { TechView } from "../components/TechView";
import { TwoScoresInstrument } from "../components/TwoScoresInstrument";
import { api, readCallLog, startCallLog, type ApiCall, type VoiceScore } from "../lib/api";
import type { Clip } from "../lib/audio";

// Layer 1 rows: plain names.
const SILENCE_PLAIN: PlotRow[] = [
  { id: "voice.2b.untrimmed", label: "Silence left in" },
  { id: "voice.2b.zerosOriginal", label: "Silence blanked out, but left at its full length" },
  { id: "voice.2b.A", label: "Silence removed (the honest test)", reference: true },
];

const PROGRESSION_PLAIN: PlotRow[] = [
  { id: "voice.p2.trimmed", label: "First try, with little training data" },
  { id: "voice.2b.A", label: "The same model with much more training data" },
  { id: "voice.2b.B", label: "More data, plus noisier training examples" },
  { id: "voice.2c.D", label: "A large ready-made speech model with a small add-on (chosen)", reference: true },
];

// Layer 2 rows: exact names.
const SILENCE_LADDER: PlotRow[] = [
  { id: "voice.2b.untrimmed", label: "LCNN, untrimmed LFCC" },
  { id: "voice.2b.zerosOriginal", label: "Silence ablation: zeros at original duration" },
  { id: "voice.2b.silenceOnly", label: "Silence-only classifier (short window of genuine silence)" },
  { id: "voice.2b.A", label: "A: LCNN, trimmed LFCC", reference: true },
  { id: "voice.p2.timingProbe", label: "Timing probe: leading-silence duration alone, no model (Phase 2 sample, different clips)" },
];

const PROGRESSION: PlotRow[] = [
  { id: "voice.p2.trimmed", label: "Phase 2 LCNN, trimmed" },
  { id: "voice.2b.A", label: "A: LCNN, trimmed, full bonafide set" },
  { id: "voice.2b.B", label: "B: A + augmentation" },
  { id: "voice.2b.C", label: "C: frozen XLS-R, mean + std pooling, layer-weighted head" },
  { id: "voice.2c.D", label: "D: frozen XLS-R, mean-only pooling (served)", reference: true },
];

const ATTACK_GAP: PlotRow[] = [
  { id: "voice.gap.A", label: "A: LCNN (interval excludes zero)" },
  { id: "voice.gap.B", label: "B: LCNN + augmentation (interval excludes zero)" },
  { id: "voice.gap.C", label: "C (not significant)" },
  { id: "voice.gap.D", label: "D (not significant)" },
];

const IN_THE_WILD: PlotRow[] = [
  { id: "voice.itw.A", label: "A: LCNN" },
  { id: "voice.itw.C", label: "C" },
  { id: "voice.itw.D", label: "D" },
];

const GLOSSARY: [string, string][] = [
  ["EER", "Equal error rate: the error rate at the setting where missed fakes and false alarms are equally common. Lower is better."],
  ["Bonafide / spoof", "A genuine human recording / a synthetic or converted one."],
  ["wav2vec2 (XLS-R)", "A large speech model trained by others on unlabelled speech. Used here as-is, with its weights frozen."],
  ["Head", "The small trainable part added on top of the frozen speech model."],
  ["Pooling", "How a recording's many short frames are summarised into one set of numbers."],
  ["Speaker-disjoint split", "No speaker appears in both the training and the test recordings."],
  ["Attack-disjoint", "Some kinds of fake are kept out of training entirely and used only for testing."],
  ["Temporal split", "Not used on this page; see the transaction page."],
  ["Calibration", "Setting warning levels so they mean a known error rate on new data. Here the levels were set on the test recordings themselves."],
  ["Seed", "The random starting point of a training run. Three seeds show how much a score moves by chance."],
  ["PR-AUC, FPR, XGBoost, GNN", "Terms from the transaction page; its glossary explains them."],
];

function VoiceDesk() {
  const health = useHealth();
  const [clip, setClip] = useState<Clip | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<VoiceScore | null>(null);
  const [calls, setCalls] = useState<ApiCall[]>([]);

  async function score() {
    if (!clip) return;
    setLoading(true);
    setError(null);
    setResult(null);
    startCallLog();
    try {
      setResult(await api.voiceScore(clip.wav, clip.name));
    } catch (e) {
      setError(errorText(e));
    } finally {
      setCalls(readCallLog());
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
        {!clip && !loading && <span className="ml-3 text-sm text-ink-soft">Add a recording first.</span>}
      </div>
      <DemoState loading={loading} error={error} waitText="Scoring the recording. This takes a few seconds." />
      {health === "down" && (
        <p className="m-0 text-sm text-ink-soft">
          Scoring needs the scoring service. Start the stack with docker compose up. You can still load a recording and look at it.
        </p>
      )}
      {result && (
        <div data-testid="voice-result" className="rule-t pt-5">
          <div className="text-xs text-ink-soft">Fake-voice score. 0 means it sounds genuine. 1 means it sounds fake.</div>
          <div className="mt-1 flex items-baseline gap-3">
            <span className="text-xl font-semibold" data-testid="voice-desk-score">
              <Score value={result.score} />
            </span>
            <VerdictTag verdict={result.verdict} />
          </div>
          <p className="measure mt-3 mb-0 text-sm">
            Scored by: <ModelInUse model={result.model} requested={result.model_requested} fallback={result.fallback_used} />.
          </p>
          <p className="measure mt-2 mb-0 text-sm">
            Medium risk starts at <Live>{result.thresholds.medium.toFixed(4)}</Live>. High risk starts at{" "}
            <Live>{result.thresholds.high.toFixed(4)}</Live>. These levels were set using the same test recordings. So they
            do not tell you how often a real caller would be wrongly flagged.
          </p>
          <p className="measure mt-2 mb-0 text-sm text-ink-soft">
            Silence at both ends was removed before scoring. Your recording had <Live>{result.speech_s.toFixed(1)}</Live> seconds
            of sound left. The model always listens to <Num id="voice.scoredSeconds" />.
          </p>
        </div>
      )}
      <CaveatNote>A research model on a research dataset. Not for real identity decisions. One recording proves nothing about accuracy.</CaveatNote>
      <TechView calls={calls} health={health} />
    </div>
  );
}

export function VoicePage() {
  return (
    <EngineerLayer page="voice">
      <section className="grid items-start gap-x-12 gap-y-10 pt-14 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]" aria-labelledby="voice-h">
        <div>
          <h1 id="voice-h" style={{ fontSize: "var(--text-hero)" }}>
            The model cheated by listening to silence. We removed the silence and measured again.
          </h1>
          <div className="measure mt-6 grid gap-2">
            <p className="m-0 text-lg">This model listens for signs that a voice is computer-made.</p>
            <p className="m-0">
              Its first score looked almost perfect: about <Num id="voice.p2.untrimmed.per100" /> in 100 recordings wrong. It was
              not real.
            </p>
            <p className="m-0">
              With the silence removed, the same model got about <Num id="voice.p2.trimmed.per100" /> in 100 wrong.
            </p>
          </div>
          <div className="mt-6">
            <EngineerToggle />
          </div>
        </div>
        <TwoScoresInstrument
          title="One voice model, the same recordings, with and without the silence at the ends."
          hue="spectrum"
          axisLabel="Error rate. Lower is better."
          axisMax={{ at: 0.3, label: "0.3" }}
          ticks={[
            { at: 0, label: "0" },
            { at: 0.1, label: "0.1" },
            { at: 0.2, label: "0.2" },
            { at: 0.3, label: "0.3" },
          ]}
          meaning={VOICE_SCORE_MEANING}
          shortcutNote="This score is a cheat, not a real result."
          lines={[
            "Same model, same recordings. Only the silence changed.",
            "In this dataset, genuine recordings start with more silence than fake ones.",
            "So a model can tell them apart without listening to the voice at all.",
          ]}
          first={{ label: "Silence kept", id: "voice.p2.untrimmed" }}
          second={{ label: "Silence removed", id: "voice.p2.trimmed" }}
        />
      </section>

      <ForEngineers title="protocol">
        <ul className="m-0 grid list-none gap-1.5 p-0">
          <li>
            Data: ASVspoof 2019 LA, evaluation partition only; a <Num id="voice.clips" />-clip subset (every bonafide clip, spoof
            sampled). Not the official train/dev/eval protocol, so not comparable with published ASVspoof numbers.
          </li>
          <li>
            Split: speaker-disjoint; <Num id="voice.trainBonafide" /> bonafide training clips in Phase 2B (
            <Num id="voice.p2.trainBonafide" /> in Phase 2).
          </li>
          <li>Metric: EER, bonafide as the target class. Seeds: three for A, C and D; single runs elsewhere.</li>
          <li>
            Front-end: wav2vec2-XLS-R-300M, frozen; per-layer pooling over frames, a learned softmax weight per layer, then a
            small MLP head. Model C pools mean and std (<Num id="voice.params.C" /> head parameters); Model D pools mean only (
            <Num id="voice.params.D" />).
          </li>
          <li>
            Trimming rule: energy-based trim of leading and trailing silence, then crop or repeat-pad to{" "}
            <Num id="voice.scoredSeconds" />, in training and in serving.
          </li>
          <li>
            Served: seed 42 of the three Model D seeds. Its own EER from the saved test scores is <Num id="voice.2c.D.seed42" />, a
            rough single-seed figure; the headline stays <Num id="voice.2c.D" />.
          </li>
        </ul>
      </ForEngineers>

      <Section title="Where the cheat was hiding" id="silence">
        <DotPlot
          caption="Error rate in three versions of the same test. Lowest first. Hatched rows are cheats, not results."
          rows={SILENCE_PLAIN}
          axisMax={0.3}
          axisLabel="error rate, lower is better"
          hue="spectrum"
        />
        <div className="measure mt-4 grid gap-2">
          <p className="m-0">Mostly, the model was measuring how long the silence lasted.</p>
          <p className="m-0">
            We blanked out the silence but kept its length. The error rate stayed very low, at <Num id="voice.2b.zerosOriginal" />.
          </p>
          <p className="m-0">So we cut the silence off every recording, in training and in the live demo.</p>
        </div>
        <ForEngineers title="silence ablations">
          <DotPlot
            caption="EER per silence condition, sorted. Hatched: shortcut, not a result."
            rows={SILENCE_LADDER}
            axisMax={0.3}
            axisLabel="EER, lower is better"
            hue="spectrum"
          />
          <p className="mt-3 mb-0">
            Duration and position carry most of the shortcut (zeros at original duration: <Num id="voice.2b.zerosOriginal" />). The
            content of the silence carries a smaller, independent signal (<Num id="voice.2b.silenceOnly" /> from{" "}
            <Num id="voice.2b.silenceOnlySeconds" /> of it). This experiment does not say which acoustic property.
          </p>
        </ForEngineers>
      </Section>

      <Section title="After the fix" id="progression">
        <p className="measure mt-0 mb-2 text-lg">
          After the fix, the chosen model gets about <Num id="voice.2c.D.per100" /> in 100 recordings wrong.
        </p>
        <div className="mb-5">
          <Meaning>
            That is an error rate of <Num id="voice.2c.D" />. {VOICE_SCORE_MEANING}
          </Meaning>
        </div>
        <DotPlot
          caption="Error rate with the silence removed, in the order we built the models. A bracket shows the spread across three training runs."
          rows={PROGRESSION_PLAIN}
          axisMax={0.3}
          axisLabel="error rate, lower is better"
          hue="spectrum"
        />
        <div className="measure mt-4 grid gap-2">
          <p className="m-0">More training data helped. Starting from a large ready-made speech model helped far more.</p>
          <p className="m-0">We split the data our own way. So these numbers cannot be compared with published results.</p>
        </div>
        <ForEngineers title="model progression">
          <DotPlot
            caption="EER on trimmed audio, in build order. Bracket: three-seed standard deviation."
            rows={PROGRESSION}
            axisMax={0.3}
            axisLabel="EER, lower is better"
            hue="spectrum"
          />
          <p className="mt-3 mb-0">
            C and D are statistically tied on every metric measured; D has half of C's head parameters (
            <Num id="voice.params.D" /> against <Num id="voice.params.C" />), so D is served.
          </p>
        </ForEngineers>
      </Section>

      <Section title="Does it still work on fakes it has never heard?" id="generalisation">
        <EvidenceRow
          ids={["voice.gap.A", "voice.gap.A.per100", "voice.gap.D"]}
          claim={<>We hid four kinds of fake during training. The simple model fell apart on them. The chosen model did not.</>}
          more={
            <>
              On the hidden kinds, the simple model got about <Num id="voice.gap.A.per100" /> more recordings in 100 wrong. The
              chosen model's error rate moved by <Num id="voice.gap.D" />. That is too small to tell apart from chance.
            </>
          }
        >
          <ForEngineers title="attack-disjoint test">
            <DotPlot
              caption="Unseen-minus-seen EER on four held-out attack systems (A07, A12, A14, A15). Brackets: 95% bootstrap intervals."
              rows={ATTACK_GAP}
              axisMax={0.3}
              axisLabel="change in EER"
              hue="spectrum"
              signedAxis
            />
          </ForEngineers>
        </EvidenceRow>
        <EvidenceRow
          ids={["voice.itw.A.per100", "voice.itw.D.per100", "voice.itw.A", "voice.itw.D", "voice.itw.total"]}
          claim={<>On a completely different set of recordings, the simple model was worse than guessing. The chosen model held up.</>}
          more={
            <>
              The simple model got about <Num id="voice.itw.A.per100" /> in 100 wrong. The chosen model got about{" "}
              <Num id="voice.itw.D.per100" /> in 100 wrong. This test used only <Num id="voice.itw.total" /> recordings, so treat
              it as a rough check.
            </>
          }
        >
          <ForEngineers title="cross-dataset test">
            <DotPlot
              caption="EER on In-the-Wild (balanced sample). Brackets: 95% intervals. C and D do not differ significantly."
              rows={IN_THE_WILD}
              axisMax={1}
              axisLabel="EER, lower is better"
              hue="spectrum"
            />
            <p className="mt-3 mb-0">
              <Num id="voice.itw.n" /> bonafide and <Num id="voice.itw.n" /> spoof clips.
            </p>
          </ForEngineers>
        </EvidenceRow>
      </Section>

      <Section title="Try it: score a recording" id="desk">
        <p className="measure mt-0 mb-6">Use your own recording. No dataset recording comes with this site.</p>
        <VoiceDesk />
      </Section>

      <Section title="Limits" id="limits">
        <ul className="measure m-0 grid list-none gap-3 p-0">
          <li className="rule-t pt-3">We tested on one dataset, plus a small check on a second one.</li>
          <li className="rule-t pt-3">The fake recordings in the data date from 2019. A newer voice generator may not be caught.</li>
          <li className="rule-t pt-3">
            The warning levels were set using the same test recordings. So they do not tell you how often a real caller would be
            wrongly flagged.
          </li>
          <li className="rule-t pt-3">Two further models were written but never finished a real run. They are untested, not disproven.</li>
        </ul>
        <ForEngineers title="limits">
          <ul className="m-0 grid list-none gap-1.5 p-0">
            <li>LA evaluation subset only, project split; no official-protocol number.</li>
            <li>
              Cross-dataset evidence is <Num id="voice.itw.total" /> In-the-Wild clips.
            </li>
            <li>Verdict thresholds are in-sample: set on the same test scores the EER is reported on.</li>
            <li>Single-seed serving: the API loads seed 42 of Model D; the reported figure is a three-seed mean.</li>
            <li>Models E (BiLSTM on frozen tokens) and F (on-the-fly augmentation) never completed a real run: untested, not disproven.</li>
          </ul>
        </ForEngineers>
        <ForEngineers title="glossary">
          <Glossary terms={GLOSSARY} />
        </ForEngineers>
      </Section>
    </EngineerLayer>
  );
}
