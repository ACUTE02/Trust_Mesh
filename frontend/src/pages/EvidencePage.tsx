import { useEffect, useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { DatasetCitations } from "../components/Footer";
import { EngineerLayer, EngineerToggle, ForEngineers } from "../components/engineers";
import { Marker, Num } from "../components/evidence";
import { KineticText } from "../components/KineticText";
import { evidence, type EvidenceId } from "../data/evidence";
import { Chapter, useResetStage } from "../world/chapters";
import { EXHIBIT_DRAWERS, clearEvidence, publishEvidence } from "../world/evidenceData";

/** A small row of squares: `filled` of `total` are solid. The counts come from the data module. */
function Squares({ total, filled, label }: { total: EvidenceId; filled: EvidenceId; label: string }) {
  const n = evidence(total).value ?? 0;
  const f = evidence(filled).value ?? 0;
  return (
    <span className="inline-flex items-center gap-1" role="img" aria-label={label}>
      {Array.from({ length: n }, (_, i) => (
        <span key={i} className={i < f ? "fill-ledger inline-block h-4 w-4" : "planned inline-block h-4 w-4"} aria-hidden />
      ))}
    </span>
  );
}

/** A bar whose width is value / of, both from the data module. */
function Gauge({ value, of, label }: { value: EvidenceId; of: EvidenceId; label: string }) {
  const share = (evidence(value).value ?? 0) / Math.max(1, evidence(of).value ?? 1);
  return (
    <span className="block h-3 w-full bg-track" role="img" aria-label={label}>
      <span className="fill-ledger block h-3" style={{ width: `${share * 100}%` }} />
    </span>
  );
}

interface Exhibit {
  key: string;
  object: string; // the kind of object it is
  title: string;
  front: ReactNode;
  back: ReactNode;
  ids: EvidenceId[];
}

const EXHIBITS: Exhibit[] = [
  {
    key: "reproduction",
    object: "a letter",
    title: "Do the site's numbers match their sources?",
    ids: ["infra.verifyChecks"],
    front: <p className="m-0 text-sm text-ink-soft">Every number on this site is looked up from one file that names its source. A check re-reads each source and compares.</p>,
    back: (
      <p className="m-0 text-sm">
        The transaction and voice numbers were re-computed from the saved model outputs and match. A separate script ran{" "}
        <Num id="infra.verifyChecks" /> end-to-end checks against the running stack and all passed. The build fails if a page shows
        a number that is not in that file.
      </p>
    ),
  },
  {
    key: "live",
    object: "a punched card",
    title: "Does the live service give the same answers as the offline model?",
    ids: ["evidence.live.n", "evidence.live.diff"],
    front: (
      <p className="m-0 flex flex-wrap gap-1" aria-hidden>
        {Array.from({ length: 24 }, (_, i) => (
          <span key={i} className="planned inline-block h-3 w-3 rounded-full" />
        ))}
      </p>
    ),
    back: (
      <p className="m-0 text-sm">
        <Num id="evidence.live.n" /> random test transactions were sent through the live service. The biggest difference from the
        offline model was <Num id="evidence.live.diff" />, and the risk level matched every time.
      </p>
    ),
  },
  {
    key: "audit",
    object: "a stamped certificate",
    title: "What did the final audit find?",
    ids: ["evidence.audit.bugs"],
    front: <p className="m-0 text-sm text-ink-soft">Each problem was reproduced first, fixed with the smallest change, and given a test that fails on the old code.</p>,
    back: (
      <p className="m-0 text-sm">
        <Num id="evidence.audit.bugs" /> bugs were found and fixed. Examples: the whole service waited while one voice clip was
        scored; an infinite weight wrote a broken case row; the website scored the previous recording after a rejected file.
      </p>
    ),
  },
  {
    key: "phone",
    object: "a specimen tray",
    title: "What does compression do to genuine speech?",
    ids: ["voice.phone.research.n", "voice.phone.research.aac", "voice.phone.libri.n", "voice.phone.libri.aac"],
    front: (
      <div className="grid gap-2 text-sm">
        <div className="flex items-center gap-3">
          <Squares total="voice.phone.research.n" filled="voice.phone.research.aac" label="Research clips still scored as low risk after compression" />
          <span className="text-ink-soft">research clips</span>
        </div>
        <div className="flex items-center gap-3">
          <Squares total="voice.phone.libri.n" filled="voice.phone.libri.aac" label="LibriSpeech clips still scored as low risk after compression" />
          <span className="text-ink-soft">one speaker</span>
        </div>
        <p className="m-0 text-xs text-ink-soft">Solid: still low risk after AAC at 32 kb/s. Dashed: flagged.</p>
      </div>
    ),
    back: (
      <p className="m-0 text-sm">
        Genuine clips, compressed like a voice memo: <Num id="voice.phone.research.aac" /> of <Num id="voice.phone.research.n" />{" "}
        research clips and <Num id="voice.phone.libri.aac" /> of <Num id="voice.phone.libri.n" /> from one speaker still passed. With
        an extra 48 kHz step, <Num id="voice.phone.research.via48" /> of <Num id="voice.phone.research.n" /> and{" "}
        <Num id="voice.phone.libri.via48" /> of <Num id="voice.phone.libri.n" /> passed. Clips from at least two speakers; very small samples.
      </p>
    ),
  },
  {
    key: "guard",
    object: "a gauge strip",
    title: "Which compressed audio does the warning catch?",
    ids: ["guard.mp3.warned", "guard.mulaw.warned", "guard.aac.warned", "guard.aac.n", "guard.clips.n"],
    front: (
      <div className="grid gap-2 text-xs text-ink-soft">
        <span>MP3 at 128 kb/s</span>
        <Gauge value="guard.mp3.warned" of="guard.clips.n" label="Share of MP3 clips the warning flagged" />
        <span>AAC at 32 kb/s and Opus at 24 kb/s</span>
        <Gauge value="guard.aac.warned" of="guard.aac.n" label="Share of AAC and Opus clips the warning flagged" />
      </div>
    ),
    back: (
      <p className="m-0 text-sm">
        The warning flagged <Num id="guard.mp3.warned" /> of <Num id="guard.clips.n" /> MP3 clips and <Num id="guard.mulaw.warned" />{" "}
        of <Num id="guard.clips.n" /> telephone-band clips, but <Num id="guard.aac.warned" /> of <Num id="guard.aac.n" /> AAC and
        Opus clips. It detects a missing high band, not compression itself. A rough check on a handful of clips.
      </p>
    ),
  },
];

const COULD_NOT_TEST = [
  "Music and singing",
  "Whispered speech",
  "Speech from many different speakers",
  "Phones other than the owner's",
  "A real deployment with real callers",
  "How accurate the combined score is",
  "Voices from newer generators than the 2019 data",
];

function ExhibitCard({ x, flipped, onFlip }: { x: Exhibit; flipped: boolean; onFlip: () => void }) {
  return (
    <article className="panel flex flex-col gap-3" data-cursor="Open the drawer" data-object={x.object}>
      <p className="m-0 flex items-center justify-between gap-3 text-xs text-ink-soft">
        <span>Exhibit: {x.object}</span>
        <Marker ids={x.ids} />
      </p>
      <h3 className="m-0 text-lg">{x.title}</h3>
      <div key={flipped ? "back" : "front"} className="flip-in min-h-16">
        {flipped ? x.back : x.front}
      </div>
      <button type="button" className="btn btn-quiet self-start text-sm" aria-pressed={flipped} onClick={onFlip}>
        {flipped ? "Turn the card back over" : "Pull the drawer and turn the card"}
      </button>
    </article>
  );
}

function Envelope({ open, onOpen }: { open: boolean; onOpen: () => void }) {
  return (
    <div className="panel" id="phase2d" data-object="a sealed envelope">
      <p className="m-0 mb-2 flex items-center justify-between gap-3 text-xs text-ink-soft">
        <span>Exhibit: a sealed envelope, opened</span>
        <Marker ids={["p2d.pooled.control", "p2d.pooled.f2", "p2d.clean.control", "p2d.clean.f2"]} />
      </p>
      <h3 className="m-0 text-lg">Phase 2D: can the voice check be trained to cope with compressed audio?</h3>
      <p className="measure mt-3 mb-3">
        Before the run, the success criteria were written down and sealed in the project notes. After the run, the result: it
        reduced the problem, not removed it.
      </p>
      {!open ? (
        <button type="button" className="btn" aria-expanded={false} onClick={onOpen}>
          Break the seal and read the result
        </button>
      ) : (
        <div className="grid gap-3 text-sm" data-testid="envelope-open">
          <p className="measure m-0">
            <strong className="font-semibold">What was tried.</strong> The same small voice head was trained twice: once on clean
            clips only (the control), once on clean clips plus squeezed copies of them (AAC, MP3, Opus, room echo and noise). Both
            were then tested on clean audio, on those six kinds of damage, and on four kinds of compression never used in training.
          </p>
          <p className="measure m-0">
            <strong className="font-semibold">What it did.</strong> Averaged over the six kinds of damage, the error rate fell from{" "}
            <Num id="p2d.pooled.control" interval /> to <Num id="p2d.pooled.f2" interval />. On clean audio it was not worse:{" "}
            <Num id="p2d.clean.control" /> for the control, <Num id="p2d.clean.f2" /> for the new head. After AAC at 32 kb/s the
            share of genuine clips wrongly flagged went from <Num id="p2d.flag.aac32.control" /> to <Num id="p2d.flag.aac32.f2" />, and
            with the extra 48 kHz step from <Num id="p2d.flag.via48.control" /> to <Num id="p2d.flag.via48.f2" />.
          </p>
          <p className="measure m-0">
            <strong className="font-semibold">Reduced, not removed.</strong> On clean audio the same threshold flags{" "}
            <Num id="p2d.flag.clean.f2" /> of genuine clips, so compressed audio is still flagged more. The new head is still weak
            on room echo (error rate <Num id="p2d.echo.f2" />) and on two compression settings it never saw (AAC at 16 kb/s,{" "}
            <Num id="p2d.aac16.f2" />; Opus at 12 kb/s, <Num id="p2d.opus12.f2" />). It ranked better than the control on only{" "}
            <Num id="p2d.heldout.better" /> of <Num id="p2d.heldout.n" /> unseen settings.
          </p>
          <p className="measure m-0 text-ink-soft">
            <strong className="font-semibold text-ink">Read this with care.</strong> The control was weaker than the voice model
            this site actually serves (its clean error rate was higher than <Num id="voice.2c.D" />, because it stopped training
            early), so part of the gain is a weak control. The test is dataset clips run through software codecs, plus a simple
            made-up echo and noise: not real phone calls. The site still serves the original voice model. Nothing here changed
            a threshold.
          </p>
          <ForEngineers title="Phase 2D">
            <p className="m-0">
              Same frozen XLS-R 300M encoder and Model D head recipe, three seeds; features precomputed per condition; paired
              bootstrap, with the pooled interval resampled by clip. Full table, caveats and criteria: docs/design_decisions.md,
              section &quot;Phase 2D FULL results&quot;.
            </p>
          </ForEngineers>
        </div>
      )}
    </div>
  );
}

export default function EvidencePage() {
  useResetStage();
  const [flipped, setFlipped] = useState<Set<string>>(new Set());
  const [envelopeOpen, setEnvelopeOpen] = useState(false);

  // The wall behind the page: a pulled exhibit pulls its drawer; the envelope's drawer follows the seal.
  useEffect(() => {
    publishEvidence({ open: EXHIBITS.flatMap((x, i) => (flipped.has(x.key) ? [EXHIBIT_DRAWERS[i]] : [])), envelopeOpen });
  }, [flipped, envelopeOpen]);
  useEffect(() => clearEvidence, []);

  const toggle = (key: string) =>
    setFlipped((prev) => {
      const next = new Set(prev);
      if (!next.delete(key)) next.add(key);
      return next;
    });

  return (
    <EngineerLayer page="evidence">
      <Chapter stage={0} className="hero" label="evidence-h">
        <div className="scrim chapter-card" style={{ maxWidth: "42rem" }}>
          <h1 id="evidence-h" style={{ fontSize: "var(--text-hero)" }}>
            <KineticText text="What we checked, and what we could not." mode="wipe" />
          </h1>
          <p className="measure mt-5 mb-0 text-lg">
            An archive of the checks behind this site: what was verified, what was found, and the drawers that stay empty because
            nobody tested them.
          </p>
          <p className="measure mt-2 mb-0">Each exhibit has a drawer in the wall behind the page. Pull one with its button.</p>
          <div className="mt-6">
            <EngineerToggle />
          </div>
        </div>
      </Chapter>

      <Chapter stage={1} id="exhibits" className="chapter" label="exhibits-h">
        <div>
          <h2 id="exhibits-h" className="mb-6 text-xl">
            <span className="scrim inline-block px-3 py-2">
              <KineticText text="Exhibits" mode="wipe" />
            </span>
          </h2>
          <div className="grid gap-4 md:grid-cols-2">
            {EXHIBITS.map((x) => (
              <ExhibitCard key={x.key} x={x} flipped={flipped.has(x.key)} onFlip={() => toggle(x.key)} />
            ))}
          </div>
        </div>
      </Chapter>

      <Chapter stage={2} id="untested" className="chapter" label="untested-h">
        <div className="panel" style={{ maxWidth: "44rem" }}>
          <h2 id="untested-h" className="mb-3 text-xl">
            <KineticText text="What we could not test" mode="wipe" />
          </h2>
          <p className="measure mt-0 mb-5 text-sm text-ink-soft">Dashed, empty drawers. Untested is not the same as safe.</p>
          <ul className="m-0 grid list-none gap-2 p-0">
            {COULD_NOT_TEST.map((t) => (
              <li key={t} className="planned px-3 py-2 text-sm">
                {t}
              </li>
            ))}
          </ul>
        </div>
      </Chapter>

      <Chapter stage={3} id="sealed" className="chapter" label="sealed-h">
        <div style={{ maxWidth: "46rem" }}>
          <h2 id="sealed-h" className="mb-5 text-xl">
            <span className="scrim inline-block px-3 py-2">
              <KineticText text="The sealed envelope" mode="wipe" />
            </span>
          </h2>
          <Envelope open={envelopeOpen} onOpen={() => setEnvelopeOpen(true)} />
        </div>
      </Chapter>

      <Chapter stage={4} id="index" className="below-fold mt-20" label="index-h">
        <div className="panel">
          <h2 id="index-h" className="mb-5 text-xl">
            Datasets and citations: the archive index
          </h2>
          <DatasetCitations />
          <p className="mt-6 mb-0 text-sm">
            <Link to="/voice">The voice page</Link> and <Link to="/graph">the transaction page</Link> hold the results these exhibits
            check.
          </p>
        </div>
      </Chapter>
    </EngineerLayer>
  );
}
