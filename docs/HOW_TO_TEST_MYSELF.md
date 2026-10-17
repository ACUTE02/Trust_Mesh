# How to test TrustMesh yourself

A short guide for the owner. No help needed. Each step says what you should see, and what that does and
does not prove. Allow about 20 minutes.

Nothing in this file comes from a dataset. Use your own voice and your own files.

## Before you start

- Docker Desktop is installed.
- You are in the folder `E:\TrustMesh\TrustMesh_CODE`.

## Step 1. Start it and check it is healthy

1. Double-click `start.bat`. Wait for the word "Ready". The site opens at http://localhost:8080.
2. Open http://localhost:8000/health in the browser.

**You should see** a block of text that includes:
- `"graph":{"ready":true`
- `"voice":{"ready":true`
- `"serving_default":"D"` and `"fallback_active":false`
- `"neo4j":{"connected":true,"nodes":203769,"edges":234355`

**What it proves:** all three parts started, the chosen voice model (D) is the one loaded, and the graph database
holds the whole graph.
**What it does not prove:** that any score is right. It only says the parts are running.

If `fallback_active` is `true`, the voice model D file is missing and model C is answering instead. The site will
say so too.

## Step 2. The transaction desk

1. Open the **Graph Engine** page. Scroll to "Try it: score a transaction".
2. Click each of the three example numbers in turn.

**You should see, each time:** a score between 0 and 1, a risk level in words (low, medium or high), a small table
of how the transaction is connected, and a picture and table of the transactions around it.

3. Now type `999999` and press "Score this transaction".

**You should see:** a clear message: "Enter a whole number from 0 to 203768." No score appears.

4. Click "Show the technical view". You should see the requests the page made, how long each took, and the raw
   answer. Click it again to hide it.

**What it proves:** the page, the scoring service and the graph database talk to each other, and a wrong input is
refused politely.
**What it does not prove:** that the model is accurate. Three transactions are far too few. The accuracy evidence
is the test-set result on the page (the score on later transactions the model never saw).

## Step 3. The voice desk

1. Open the **Voice Engine** page. Scroll to "Try it: score a recording".
2. Record yourself: click "Record from the microphone", allow the microphone, speak for 5 seconds, click "Stop
   recording". Then click "Score this clip".
   (Or record on your phone, save the file, and use "Choose an audio file".)

**You should see:** a picture of your recording, a score between 0 and 1, and a risk level. For a real voice you
would hope for the genuine side: a low score and "low risk". Write down what you got.

**Known result (self-test, 2026-10-08):** three genuine phone recordings of the owner were all flagged "high risk".
Compressed phone audio seems to confuse this model. So do not be surprised if your own voice is flagged. See
`docs/SELF_TEST_REPORT.md`, section 6.

**Voice model choice.** Under the recording controls there is a small "Voice model" choice. D is the default and the
main model. F2 appears only when the service has its private weights; it was trained on copies of the training audio
that had been compressed like phone audio. Try your own phone memo with D and then with F2 and write both down. The
page names the model that scored (F2 says it is the mean of seeds 42, 43 and 44). F2 reduced the compressed-audio
problem in a synthetic test and did not remove it; it was not tested on real phone calls. The combined case desk always
uses D. Through the API: `POST /voice/score?model=F2`.

3. Make a fake clip with any free text-to-speech tool. Save it as MP3 or WAV. Upload it and score it. Write down
   what you got.

**Read this before you judge the result:**
- The model learned from fake voices made in 2019. A modern voice generator may **not** be caught. If your
  text-to-speech clip scores "low risk", that is a known limit, not a bug.
- Your own voice may score higher than you expect. A phone microphone, background noise or a very short clip can
  all move the score. The model has only been measured on studio-style research recordings and one small outside set.
- **One clip proves nothing about accuracy**, in either direction. The accuracy evidence is on the page: the
  test-set error rate, the test on kinds of fake it never trained on, and the small check on a different dataset.
- The warning levels were set using the same test recordings. They do not tell you how often a real caller would
  be wrongly flagged.

**What it proves:** recording, upload, conversion, scoring and display all work on your machine.
**What it does not prove:** that the model can tell your voice from a fake in general.

## Step 4. The combined desk

1. Go to the first page (Overview). Scroll to "Try it: one case, two checks".
2. Pick a transaction. Press "Score this case" without a recording.

**You should see:** a transaction score. The voice row and the combined row both say "empty". The Visual row is
dashed and says "not connected".

3. Add a recording (your own). Press "Score this case" again.

**You should see:** a transaction score, a voice score, and a combined score. Under them, always, the note: "The
combined score is a demonstration of the idea. It has no measured accuracy..."

**What it proves:** the two scores are fetched and averaged, and the page never hides the two parts or the warning.
**What it does not prove:** anything about how well a combined warning works. No dataset has the same case in both
checks, so it has never been tested.

## Step 5. The automatic checks

Open PowerShell in `E:\TrustMesh\TrustMesh_CODE` and run these one at a time.

```
.venv\Scripts\python.exe scripts\verify_serving.py
```
**Pass looks like:** every line starts with `[ok]`.
**A pass means:** every endpoint answered, the graph model reproduces its reported test score, the graph database
has the full graph, and the combined score equals the average of its two parts.

```
.venv\Scripts\python.exe -m pytest tests -q
```
**Pass looks like:** a last line such as `116 passed`.
**A pass means:** the code does what its own small tests expect, on made-up data. It says nothing about accuracy
on real data.

```
.venv\Scripts\python.exe scripts\check_frontend_numbers.py
```
**Pass looks like:** `checked 81 entries against their sources` and no line starting `MISMATCH`.
**A pass means:** every number on the site still matches the file it came from.

```
cd frontend
npm run build
```
**Pass looks like:** `28 passed` and then `built in ...`.
**A pass means:** the site builds, and no page shows a number that is not in the data file.

Note: `verify_serving.py` uses a few local test recordings in `data\demo_clips`. They are dataset files kept on
your machine only. Do not show them or their scores publicly.

## Step 6. Things nobody has tested yet (please try them)

These were **not** tested when the site was built. If one fails, note what you did and what you saw.

1. **The microphone button** (Step 3). Does the browser ask for permission? Does "Stop recording" produce a clip?
   Try Chrome and Edge.
2. **A real speech recording** through the site. So far only a computer-made test tone has been scored through
   the new pages.
3. **Keyboard only.** Put the mouse away. Press Tab to move and Enter or Space to press.
   - Can you reach the "Random split / Time split" control and change it with the arrow keys?
   - Can you choose a file, score a case, open the technical view, and open "Show the detail for engineers"?
   - Can you always see which item is selected (a dark outline)?
4. **Phone.** Open the site on your phone while it is on the same Wi-Fi (this needs the address of your computer,
   and may need a firewall change; skip it if that is a bother). Or narrow the browser window to phone width. Is
   everything readable with no sideways scrolling?
5. **Dark theme.** Click "Switch to dark theme" and read each page once.
6. **Read it as a stranger.** Read the first screen of each page without clicking "Show the detail for engineers".
   Is there any word you would have to explain? Write it down.

## When you are done

Double-click `stop.bat`. Your data is kept for next time.
