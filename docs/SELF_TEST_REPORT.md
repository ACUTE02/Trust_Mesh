# TrustMesh self-test report

Run on 2026-10-08 following `PHASE_2_SELF_TEST_RUNBOOK.md` (steps ST1 to ST9) plus one extra
step (ST10) with the owner's own voice clips. Every number here was measured in this run.

**What this report can and cannot show.** It can show that the running system gives the same results that are
written in the project log, and that it fails safely. It cannot create new proof of accuracy. The only accuracy
evidence is still the test data each model was scored on before.

No training was done. Nothing was pushed or deployed. No model code was changed. This report holds totals only:
no dataset clip, no real transaction with its score, and none of the owner's clips or their scores.

## 1. What we claimed, and what we measured now

| What we claimed | What we measured now | Result | What this proves | What it does not prove |
|---|---|---|---|---|
| The transaction model scores 0.7474 on later transactions (one training run). | 0.7474 on 15,329 labelled test transactions. Difference 0.00003. | Pass (allowed difference 0.0005) | The model file we serve is the one that was tested. | That it would score the same on other data. |
| The voice model's error rate is 0.0166, give or take 0.0019, over three training runs. | The three runs give 0.0194, 0.0152 and 0.0152. Mean 0.0166, spread 0.0019. | Pass (allowed difference 0.0005) | The saved test scores give the numbers in the log. | Anything about recordings outside that test set. |
| The one voice run we serve (run "42") has an error rate of about 0.0194. | 0.0194. | Pass | The site's "rough single-run figure" is right. | See row 6 for how unsure that figure is. |
| The live service gives the same transaction scores as the offline model. | 200 random test transactions sent through the service. Biggest difference: 0. Risk level matched every time. | Pass (allowed difference 0.001) | Nothing is lost between the model and the web service. | The same for voice: see the next row. |
| The live service gives the same voice scores as the saved test scores. | **Not run.** The test recordings are not on this machine. They were not downloaded. | Not run | Nothing. | The voice path (convert, trim, encode, score) is not checked end to end against the saved scores. |
| Run 42's error rate, with its uncertainty. | 0.0194, with a 95% range of 0.0159 to 0.0214 (1,000 resamples). | Estimate | The true figure for this test set is probably inside that range. | It is still one test set. |
| The voice warning levels were set on the same test recordings. How do they do on recordings they were not set on? | See section 3. About 1 genuine recording in 100 is flagged "high"; about 99.5 fakes in 100 are flagged. Nearly the same as before. | Estimate | The levels were not fitted to luck. | A real-world false-alarm rate. See section 6. |
| Bad or strange inputs give a clear error, never a crash or a silent wrong answer. | 12 cases tried. Two problems found and fixed. All 12 pass now. | Pass after fixes | The service refuses or flags bad input. | That it knows speech from noise: it does not. See section 4. |
| The site can be used with the keyboard alone. | Every button, link and field on all three pages was reached with the Tab key. All three desks were run with the keyboard. | Pass | Keyboard users can do everything. | Screen-reader use: not tested. |
| The site has no basic accessibility errors. | The axe-core checker found 0 problems in 12 page checks (3 pages, light and dark, plain and engineer view). | Pass | No automatic check fails. | Automatic checks catch only part of real accessibility problems. |
| The site fits phone, tablet and desktop screens. | 36 checks at 390, 768 and 1440 wide, both themes, both views. No sideways scrolling. | Pass | Layout fits. | How it feels on a real phone. |
| A fresh copy of the project tells you what files to add. | A fresh copy was started with no data and no internet. Two real problems found and fixed. See section 5. | Pass after fixes | A newcomer now gets a plain list. | A full first start with real downloads: not run. |

## 2. How correct is the project?

**Shown with evidence**

- The transaction numbers in the log are reproduced exactly from the model we serve.
- The voice numbers in the log are reproduced from the saved test scores.
- The live service returns the same transaction scores as the offline model.
- The service fails safely on bad input, after two fixes made during this test.
- The site works by keyboard, passes the automatic accessibility check, and fits small screens.

**An estimate, not a proof**

- How the voice warning levels behave on recordings they were not set on (section 3).
- The uncertainty on the one voice run we serve: 0.0159 to 0.0214.

**Not proven**

- **The voice model on ordinary phone recordings.** In a small check with the owner's own clips it failed. See section 6.
  This is the most important finding of this report.
- The combined score. It has never been tested for accuracy, and cannot be with this data.
- The Visual Engine. It has no results.
- Any real-world deployment.
- Modern fake-voice tools beyond the 2019 test data.
- That the live voice path matches the saved scores (the check could not be run).

## 3. The voice warning levels, tested on recordings they were not set on

The levels on the site (high from 0.6578, medium from 0.0239) were set using all the test recordings. That is why
the site says they do not tell you a real false-alarm rate.

Here we split the saved test scores in half, 200 different ways. Each time we set the levels on one half, using the
same rule, and measured on the other half. The test set has 2,165 genuine and 8,992 fake recordings.

| | On the recordings the levels were set on | On the other half (average of 200 splits) | Smallest to largest of the 200 |
|---|---|---|---|
| Genuine recordings flagged "high" | 1.0 in 100 | 1.1 in 100 | 0.3 to 2.5 in 100 |
| Genuine recordings flagged "medium" or "high" | 10.0 in 100 | 9.9 in 100 | 7.1 to 13.5 in 100 |
| Fake recordings flagged "high" | 97.6 in 100 | 97.6 in 100 | 96.8 to 98.4 in 100 |
| Fake recordings flagged "medium" or "high" | 99.5 in 100 | 99.5 in 100 | 99.3 to 99.7 in 100 |

**What it means.** On this dataset the levels behave almost the same on recordings they were not set on. So they
were not an accident of the particular recordings.

**What it does not mean.** This is one dataset of studio-style research recordings. It is not a real-world rate.
Section 6 shows why. The served levels were not changed, and these figures are not on the site.

## 4. Does it fail safely?

| What we sent | What came back | Result |
|---|---|---|
| A transaction number that does not exist | "Not found", with the allowed range | Pass |
| Letters instead of a transaction number | Refused: "should be a valid integer" (a technical message, but clear) | Pass |
| An empty file | Refused: "the file is empty" | Pass after fix (the old message was a line of program text) |
| A text file | Refused: "That is not an audio file" | Pass |
| A 61-second recording | Refused: "the limit is 60 s" | Pass |
| A file over 10 MB | Refused: "larger than 10 MB" | Pass |
| A 0.1-second recording | Scored, but marked low confidence with a warning | **Pass after fix.** Before, it got a confident score with no warning |
| 3 seconds of pure silence | Refused: "the recording is silent" | **Pass after fix.** Before, silence came back as "low risk" with no warning |
| 3 seconds of loud noise | An ordinary score, no warning | Pass (no crash), with a limit: see below |
| A 3-second computer tone | An ordinary score ("high risk"), no warning | Pass (no crash), with the same limit |
| 40 scoring requests sent at once | 30 scored, 10 refused with "too many requests" | Pass |
| Is the service still running after all that? | Yes | Pass |

**The limit you should know.** The service has no check that a sound is speech. Noise, music or a tone gets a normal
score, as if it were a voice. Only silence and very short clips are caught.

One more thing we learned: one request after another cannot reach the limit of 30 a minute, because scoring one
recording takes over a second. The limit only bites when many requests arrive together.

**Changes made to the service because of this step** (the model itself was not touched):
1. A silent recording is refused instead of scored.
2. A recording with under 1 second of sound left after trimming is scored but marked low confidence, with a warning the site shows.
3. An empty or unreadable file gets a plain message.

## 5. Starting from a fresh copy

We cloned the project into an empty folder, copied `.env.example` to `.env`, and started it with no data, no
weights and no internet.

**Problem 1 (now fixed).** The site would not build at all. Its data files had never been saved in git, because a
rule meant to hide the big `data/` folder also hid the site's own small `data` folder. Anyone cloning the project
would have been stuck. The rule now hides only the top-level folder, and the four files are saved.

**Problem 2 (now fixed).** With no dataset, the loader stopped with a long program error, the service never
started, and the site never opened. Nothing said what to add. Now the loader prints a plain list of the missing
files, with where each goes and where it comes from. The site opens anyway, and its desks say the scoring service
did not answer.

The list of files a fresh copy needs is now in the README under "Files you must supply": two graph model files, a
voice model file and its test scores, the transaction dataset and the speech model (the last two download
themselves if there is internet).

**What we could not do.** Rebuilding the big service image inside the fresh copy needs about 1 GB of downloads, and
that timed out on this connection. We used the image already built from the same code, and built only the site
image from the fresh copy. A complete first start with real downloads was not run.

## 6. A small check with the owner's own voice (ST10)

The owner supplied 3 pairs of clips: a genuine recording of himself, and a cloned-voice copy of the same sentence.
We converted copies to the format the service needs (the originals were not touched) and scored each one.
All six clips were between 5 and 11 seconds long. None was under 4 seconds, so none is marked "too short".

| | Count |
|---|---|
| Genuine recordings passed as genuine ("low risk") | **0 of 3** |
| Genuine recordings wrongly flagged as fake | **3 of 3**, all "high risk" |
| Cloned recordings caught ("high risk") | 3 of 3 |

**Read this plainly.** The model flagged everything, real and fake alike. It did not tell the owner's real voice
from the clone. Catching 3 of 3 fakes means nothing when it also flags 3 of 3 real recordings.

**This is a small demo check, not an accuracy measurement.** Three pairs from one person on one phone cannot give a
rate. But it is a clear warning sign, and it agrees with the next test.

**Why this happens: a follow-up test is in `docs/PHASE_2_STATUS_REPORT.md` ("Why do genuine phone recordings score as fake?").** It found that compression alone does not explain the whole result. The first diagnostic was:

We took 4 genuine recordings from the research dataset, which the model passes
correctly. We squeezed them the way a phone voice memo is squeezed (AAC, low bitrate) and scored them again. Only 1
of 4 still passed; 1 became "medium" and 2 became "high". So this kind of compression alone can make a genuine
voice look fake to this model. Four recordings is a small check too, but both checks point the same way: the model
learned from clean studio recordings and does not cope with compressed phone audio.

**What it means for the project.** The headline error rate (about 1.7 in 100) is true for the research test set.
It does not carry over to an ordinary phone recording. The site already says the model is not for real identity
decisions; this check shows why. For a demo, expect your own voice to be flagged.

The table with each clip's length, score and level is **not in this report and not in the repository**. The owner
asked that the clips and their scores never be copied into the repository, and this report is part of it. The
table is in a private file next to the clips: `my_voice_tests\MY_VOICE_RESULTS.md`.

Tool note: the conversion needs a program called ffmpeg, which was not installed on this machine. A copy was
installed inside the project's own Python folder only (the `imageio-ffmpeg` package in `.venv`). Nothing was
installed system-wide.

## 7. Clean-ups done

1. **The check script no longer leaves test rows behind.** The scoring service accepts `record=false`.
   `scripts/verify_serving.py` uses it, and now checks that the number of saved cases is the same before and after.
   The cases table was backed up first. Test rows left by earlier runs were removed; the 25 seeded rows remain.
2. **Two graph numbers that disagree (issue #45).** Report only; nothing on the site was changed.
   - The log and the leaderboard say 0.291 and 0.265. The result file on disk says 0.3279 and 0.2638.
   - They are two runs of the same notebook. The log's run is also the one the next notebook used for its
     "does this reproduce?" checks. The file's run is the one whose saved model files are on disk.
   - The result file carries no run date, and the notebooks hold no saved output, so the second run cannot be dated
     from the files. Its download time is 2026-09-23.
   - Which looks canonical: the log's run is the published one; the file's run is the only one that can be checked
     against files on disk. Both give the same ranking and the same conclusion. This is the owner's call, or one
     seeded re-run on Kaggle would settle it.
3. **A cleaned copy of the graph result file.** `artifacts/graph/results_sanitised.json` has the two blocks about one
   single transaction removed (its number, its predicted probability, its neighbour, and its feature details).
   Totals are kept. The number check passes on it for 79 of 81 entries. The other 2 are the two facts the site states
   about that one transaction (its number, and how little the prediction moved), which were in the removed blocks.
   The file is left uncommitted for the owner to decide.
4. **Recording mode.** Setting `TRUSTMESH_RECORDING_MODE=1` makes the site hide real transaction numbers, labels,
   scores and risk levels on the two desks that show transactions, and removes the technical view there. The voice
   desk is not affected. Checked in the browser: all 24 live values on the transaction desk read "hidden". It is
   off by default. Not checked: the "shortest path" box in this mode.

## 8. What was not run, and why

- **Live voice scores against the saved test scores (ST2, voice half).** The test recordings are not on this machine
  and the runbook says not to download them.
- **A full first start from a fresh copy with real downloads (ST7).** The 1 GB image rebuild timed out; the dataset
  and speech model were deliberately not downloaded.
- **Real speech through the microphone button.** The microphone flow was run with a fake device playing a tone. It
  works, but that is not a person speaking.
- **Screen readers.** Only the automatic checker and keyboard tests were run.
- **A real phone.** Small screens were emulated.
- **The "shortest path" box by keyboard and in recording mode.**
- **Any re-run of training** to settle the two disagreeing graph numbers. Training is not allowed here.
- **A larger check of the phone-audio problem.** Section 6 used 3 pairs and 4 dataset recordings. A proper test
  needs many genuine phone recordings from many people, which this project does not have.

## 9. Where the detail is

- Totals from each step: `data/self_test/*.json` on this machine (not in the repository).
- The scripts: `scripts/self_test_offline.py`, `self_test_live.py`, `self_test_browser.py`.
- The owner's clips and their results: `E:\TrustMesh\my_voice_tests\` (private).
