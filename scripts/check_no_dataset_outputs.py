"""Fail if anything derived from the image datasets is about to be committed.

The datasets behind the Visual Engine carry non-commercial and no-redistribution terms
(docs/PHASE_3_VISUAL_ENGINE.md, section 12a). So these never go into a public repository: dataset
images or frames, embedding files, any table with one row per dataset image (metadata, predictions,
error tables), model weights trained on the datasets (until their licenses are checked), and any
Celeb-DF or FaceForensics++ material. Code, specs, aggregate metrics, reliability diagrams and
aggregate per-generator tables may be committed.

Usage:
  python scripts/check_no_dataset_outputs.py                 # check what is staged (git diff --cached)
  python scripts/check_no_dataset_outputs.py --all           # check every tracked file
  python scripts/check_no_dataset_outputs.py --install-hook  # run it automatically before each commit
  python scripts/check_no_dataset_outputs.py path [path ...]  # check the given paths

Exit code 1 if a forbidden path is found or .gitignore is missing a required entry.
"""
import fnmatch
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (pattern, why). Patterns are matched against the lower-cased path with forward slashes.
FORBIDDEN = [
    ("artifacts/visual/private/*", "per-image outputs, embeddings and weights of the Visual Engine"),
    ("artifacts/visual/*.npz", "embedding or weight arrays"),
    ("artifacts/visual/*.parquet", "per-image table"),
    ("*/embeddings/*", "cached embeddings"),
    ("*metadata*.parquet", "per-image metadata table"),
    ("*predictions*.parquet", "per-image predictions"),
    ("*per_image*", "per-image table"),
    ("*celeb*df*", "Celeb-DF material"),
    ("*faceforensics*", "FaceForensics++ material"),
    ("artifacts/visual/*.jpg", "dataset image"), ("artifacts/visual/*.jpeg", "dataset image"),
    ("artifacts/visual/*.webp", "dataset image"), ("artifacts/visual/*.bmp", "dataset image"),
    ("data/*", "raw data folder"),
    # Graph and Voice engines (added 2026-10-08, after the dataset terms were read: docs/design_decisions.md).
    ("artifacts/*.pt", "model weights trained on a dataset"),
    ("artifacts/*.npz", "per-sample scores or arrays"),
    ("artifacts/graph/*xgboost*.json", "XGBoost model weights"),
    ("artifacts/graph/results.json", "holds one transaction's prediction and explanation example"),
    ("artifacts/graph/results_phase1c_partial.json", "holds one transaction's prediction and explanation example"),
    ("artifacts/voice/phase2c_*/*", "extracted weights or per-sample scores"),
    ("artifacts/voice/_to_delete/*", "extracted weights or per-sample scores"),
    ("*.wav", "audio clip"), ("*.flac", "audio clip"), ("*.mp3", "audio clip"), ("*.ogg", "audio clip"),
    ("*.sqlite", "cases table: per-sample predictions"),
    (".env", "local settings and secrets"), ("*/.env", "local settings and secrets"),
]
# Aggregate outputs that look like the above but are fine to commit.
ALLOWED = ["docs/*", "scripts/*", "tests/*", "engines/*", "notebooks/*"]
REQUIRED_GITIGNORE = ["artifacts/visual/private/", "artifacts/visual/**/*.npz", "artifacts/visual/**/*.parquet",
                      "**/embeddings/", "*celeb*df*/", "*faceforensics*/"]


def forbidden_paths(paths):
    """[(path, reason)] for every path that must not be committed."""
    found = []
    for path in paths:
        norm = str(path).replace("\\", "/").lower().lstrip("./")
        if any(fnmatch.fnmatch(norm, a) for a in ALLOWED):
            continue
        for pattern, why in FORBIDDEN:
            if fnmatch.fnmatch(norm, pattern):
                found.append((str(path), why))
                break
    return found


def missing_gitignore_entries(gitignore=ROOT / ".gitignore"):
    lines = {line.strip().lower() for line in gitignore.read_text(encoding="utf-8").splitlines()} if gitignore.exists() else set()
    return [entry for entry in REQUIRED_GITIGNORE if entry.lower() not in lines]


def git_paths(*args):
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    return [line for line in result.stdout.splitlines() if line.strip()]


def install_hook():
    hooks = ROOT / ".git" / "hooks"
    if not hooks.is_dir():
        sys.exit("not a git repository yet: run `git init` first, then install the hook")
    hook = hooks / "pre-commit"
    hook.write_text("#!/bin/sh\n# Blocks commits that contain dataset-derived files (see scripts/check_no_dataset_outputs.py).\n"
                    'exec "$(git rev-parse --show-toplevel)/.venv/Scripts/python.exe" '
                    '"$(git rev-parse --show-toplevel)/scripts/check_no_dataset_outputs.py"\n', encoding="utf-8")
    print(f"installed {hook}")


def main(argv):
    if "--install-hook" in argv:
        return install_hook()
    explicit = [a for a in argv if not a.startswith("--")]
    if explicit:
        paths, what = explicit, "given paths"
    elif "--all" in argv:
        paths, what = git_paths("ls-files"), "tracked files"
    else:
        paths, what = git_paths("diff", "--cached", "--name-only"), "staged files"
    problems = []
    if paths is None:
        print("not a git repository (yet): nothing is staged, checking .gitignore only")
        paths = []
    for path, why in forbidden_paths(paths):
        problems.append(f"FORBIDDEN in {what}: {path}  ({why})")
    for entry in missing_gitignore_entries():
        problems.append(f".gitignore is missing the entry: {entry}")
    if problems:
        print("\n".join(problems))
        sys.exit(1)
    print(f"ok: {len(paths)} {what} checked, no dataset-derived file, .gitignore has the required entries")


if __name__ == "__main__":
    main(sys.argv[1:])
