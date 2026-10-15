# Kaggle runbook: Phase 2D (does training on compressed audio fix the voice model?)

For the owner. Plain steps, no coding. Nothing here has been run yet: **nothing has been trained so far.**

## What this run does, in plain words

The voice model scores genuine speech as "more fake" after it has been compressed (phone calls, voice messages).
This run trains a second small copy of the model on clean clips **plus compressed copies of the same clips**, and a
control copy on clean clips only, with the same settings. Then it compares the two on clean audio and on ten kinds of
compressed or noisy audio, six that were used in training and four that were not.

It does not change the website or the served model. It writes numbers and small model files, nothing else.

## What you need

- A Kaggle account with phone verification (so that GPU and Internet can be switched on).
- One file from this repository: `notebooks/06_voice_codec_robust.ipynb`. It is self-contained; nothing else is uploaded.
- No dataset to attach. Like the earlier voice notebooks, it downloads the audio and the speech encoder itself, so
  **Internet must be On**.

## Step by step

1. On kaggle.com choose **Create → New Notebook**, then **File → Import Notebook** and pick
   `06_voice_codec_robust.ipynb`. Keep the notebook **Private**.
2. In the right-hand settings: **Accelerator = GPU T4 x1** (or better), **Internet = On**. Persistence: choose
   "Files only" (or leave default).
3. **Do the pilot first.** The setup cell near the top of the notebook contains the line

   ```
   RUN_MODE = os.environ.get("TRUSTMESH_RUN_MODE", "PILOT")   # "PILOT" (measure and stop) or "FULL"
   ```

   Leave it as it is. Click **Save Version → Save & Run All (Commit)**. (Do not use the interactive "Run All" first;
   the commit run starts from scratch anyway, so running it twice just wastes GPU time.)
4. When the version has finished, open its log and read the last lines. The pilot prints, for about 500 clips:
   - seconds per clip,
   - **projected hours** for the full extraction,
   - **projected cache size in GB** and the free disk space,
   - whether the disk is large enough with a safety margin.

   It processes about 500 clips, then stops. It trains nothing and downloads the encoder plus the dataset, which
   takes a few minutes of its own.
5. **Expected total time: unknown until the pilot.** The pilot's projection is a straight-line estimate and the
   real run can be slower or faster. Head training adds minutes, not hours.
6. If the projection fits (the disk check says OK and the hours are below about 9), change the word `"PILOT"` to
   `"FULL"` in that same line:

   ```
   RUN_MODE = os.environ.get("TRUSTMESH_RUN_MODE", "FULL")
   ```

   then again **Save Version → Save & Run All (Commit)**.
7. If the full run stops with the message "STOPPED EARLY on the time limit": nothing is lost. Every finished piece of
   work is saved in the version's output (in the `cache_phase2d` folder). Start a new notebook session with that
   version's output attached as an input, set the environment line in the setup cell to
   `os.environ.get("TRUSTMESH_RESUME_DIRS", "<path of the attached cache_phase2d folder>")`, keep `"FULL"`, and run again.
   It continues from the first missing piece.
8. If you see "Not enough free disk space": it stopped before starting, on purpose (the earlier runs failed on disk).
   Tell me the numbers it printed. Two safe levers: lower `EVAL_PER_CONDITION` in the setup cell (for example 2000), or
   run on a smaller part. Do not change anything else; the settings that make the comparison fair are fixed.

## What to download when it finishes

From the version's **Output** panel, into `artifacts/voice/` in this repository (these paths are git-ignored, nothing
here may be committed or pushed):

- `results_phase2d.json` (small; aggregates only, no clip names, no per-clip scores)
- `phase2d_control_s42.pt`, `phase2d_control_s43.pt`, `phase2d_control_s44.pt`
- `phase2d_F2_s42.pt`, `phase2d_F2_s43.pt`, `phase2d_F2_s44.pt` (small files; keep them private until you have checked
  the dataset terms)

Then tell me it is done. I will compare the numbers with the success criteria written in
`docs/design_decisions.md` (section "Phase 2D"), which were fixed before any run. A criterion that fails is reported as
a negative result; I will not retune and re-run to make it pass.

## If something goes wrong

- *The page says the dataset or encoder cannot be downloaded:* Internet is Off, or Kaggle's phone verification is not
  done. Fix the setting and run again.
- *"No ffmpeg found":* the notebook tries the system ffmpeg and then installs `imageio-ffmpeg` by itself. If both fail,
  send me the message; the run cannot make compressed copies without it.
- *A codec name is listed as SUBSTITUTED or UNAVAILABLE in the log:* this machine's ffmpeg lacks that codec. The notebook
  says what it used instead and writes it into the results file. This is allowed and visible; send me the line.
- *The split check stops the run ("P1 split ... differ"):* the notebook refuses to compare against a different split than
  Model D's. Send me the message; do not edit the check.
