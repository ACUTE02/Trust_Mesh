# Voice Data Exploration -- ASVspoof 2019 LA

Source: `SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA (split 'test' = LA evaluation partition)`,
streamed with the HuggingFace `datasets` library.

**Subset actually used for every number in this phase: 6000 clips** (of the 71,237 in the LA
evaluation partition), sampled with a shuffle buffer at seed 42. This is the figure to quote; it is a
subset, not the full partition.

## Class balance
- Bonafide: 619
- Spoof: 5381
- Bonafide share: 10.32%

Spoof outnumbers bonafide by about 8.7 to 1, so the training loss is class-weighted by that ratio and
the headline metric is EER rather than accuracy -- the same reasoning that made PR-AUC the metric for
the graph engine.

## Clip durations (as provided, before any trimming)
- Mean: 3.12 s
- Median: 2.82 s
- Min: 0.56 s
- Max: 13.01 s
- Total audio: 5.21 hours

## Sample rate
- As provided: 16000 Hz (the pipeline resamples to 16 kHz if needed; it was never needed here).

## Speakers
- Distinct speaker IDs: 67

Speaker identity matters for the split: a random clip-level split would place the same speaker on
both sides, letting the model key on voice/channel identity instead of spoofing artifacts. The
headline split in this phase is speaker-disjoint; see `docs/design_decisions.md`.

## What the silence reveals
Measured by the notebook's energy-based trim (30 dB below each clip's peak):

| | Bonafide | Spoof |
|---|---|---|
| Mean leading silence | 1.10 s | 0.30 s |
| Mean trailing silence | 0.55 s | 0.29 s |
| Mean speech duration | 1.80 s | 2.50 s |
| Mean silence share of clip | 48% | 19% |

Bonafide clips carry almost four times as much leading silence as spoofed ones, and silence makes up
nearly half of a typical bonafide clip. Leading silence *alone* separates the classes at EER 0.18.
This is the known ASVspoof silence shortcut, and it is why the pipeline trims silence before
featurizing; see the silence-leakage results in `docs/design_decisions.md`.

## Known caveat
These clips all come from the LA **evaluation** partition, so the same spoofing systems (A07-A19)
appear in both the train and test halves of our split. Performance against *unseen* attacks is not
measured here.
