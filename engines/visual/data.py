"""Data access for the Visual Engine: secrets, sparse Parquet reads over HTTP ranges, folder
datasets, the streaming-feasibility measurements of Task 0, and synthetic data for local tests.

**Why a sparse Parquet reader.** OpenFake is 3.4 TB of Parquet shards of about 5 GB each, and each
row group holds about 70 images in about 128 MB (as read from two shard footers on 2026-10-07; Task 0
re-measures this on Kaggle). The label/model/prompt columns of a row group are a few kilobytes that
sit right after its image column. `SparseParquet` fetches the footer, works out the byte range of
exactly the column chunks asked for, and downloads only those -- so the whole dataset's metadata can
be counted without touching an image, and one row group's images can be fetched without the rest of
its shard. Nothing is ever written to disk.
"""
import csv
import hashlib
import io
import json
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from .preprocess import inspect_image

OPENFAKE_REPO = "ComplexDataLab/OpenFake"
DEEPFAKE_EVAL_REPO = "nuriachandra/Deepfake-Eval-2024"
HF_SECRET_NAME = "HF_TOKEN"
REAL, FAKE = 0, 1
META_COLUMNS = ("prompt", "label", "model", "type", "release_date", "row_id")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


# ---------------------------------------------------------------------------------------------------
# Secrets: read from Kaggle Secrets, never printed, never saved
# ---------------------------------------------------------------------------------------------------
def read_hf_token():
    """The Hugging Face token from the Kaggle secret `HF_TOKEN` (or the environment variable of the
    same name), or None. Callers must not print it or store it in results."""
    token = None
    try:
        from kaggle_secrets import UserSecretsClient

        token = UserSecretsClient().get_secret(HF_SECRET_NAME)
    except Exception:  # noqa: BLE001 -- not on Kaggle, or the secret is not attached to this notebook
        token = None
    return token or os.environ.get(HF_SECRET_NAME) or None


def scrub_secrets(value, secrets):
    """A copy of `value` (nested dicts/lists/strings) with every secret replaced by `<redacted>`.
    Everything written to a results file or printed from an exception goes through this."""
    secrets = [s for s in (secrets or []) if s]
    if isinstance(value, str):
        for secret in secrets:
            value = value.replace(secret, "<redacted>")
        return value
    if isinstance(value, dict):
        return {scrub_secrets(k, secrets): scrub_secrets(v, secrets) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [scrub_secrets(v, secrets) for v in value]
    return value


# ---------------------------------------------------------------------------------------------------
# Byte-range readers
# ---------------------------------------------------------------------------------------------------
def _with_retries(fn, attempts=4, wait=2.0):
    for attempt in range(attempts):
        try:
            return fn()
        except Exception:  # noqa: BLE001 -- network reads are retried, then re-raised
            if attempt == attempts - 1:
                raise
            time.sleep(wait * (attempt + 1))


class FsRangeReader:
    """Byte ranges through any fsspec filesystem (local files in tests, `HfFileSystem` on Kaggle)."""

    def __init__(self, fs):
        self.fs = fs
        self.requests = 0
        self.bytes_read = 0

    def read(self, path, start, end):
        data = _with_retries(lambda: self.fs.cat_file(path, start=start, end=end))
        self.requests += 1
        self.bytes_read += len(data)
        return data


class HttpRangeReader:
    """Byte ranges with plain HTTP `Range` requests against a Hugging Face dataset repository.
    Takes the same `datasets/<repo>/<file>` paths as `HfFileSystem`.

    The Hub answers a file request with a redirect to a signed storage URL. That URL is remembered
    per file, so reading the ~60 row groups of a shard costs the Hub one request, not sixty (the Hub
    limits how many requests an account may make; the storage host is not part of that limit and
    never sees the token). The token is sent to the Hub only, in the request header.
    """

    REDIRECTS = (301, 302, 303, 307, 308)

    def __init__(self, repo=OPENFAKE_REPO, token=None, revision="main", session=None, timeout=600):
        self.repo, self.revision, self.timeout = repo, revision, timeout
        self._headers = {"Authorization": f"Bearer {token}"} if token else {}
        if session is None:
            from huggingface_hub.utils import get_session

            session = get_session()
        self._session = session
        self._storage_url = {}
        self.requests = 0
        self.hub_requests = 0
        self.bytes_read = 0

    def _hub_url(self, path):
        from urllib.parse import quote

        prefix = f"datasets/{self.repo}/"
        filename = path[len(prefix):] if path.startswith(prefix) else path
        return f"https://huggingface.co/datasets/{self.repo}/resolve/{quote(self.revision, safe='')}/{quote(filename)}"

    def _get(self, url, headers):
        try:
            return self._session.get(url, headers=headers, follow_redirects=False, timeout=self.timeout)
        except TypeError:  # an older huggingface_hub whose session is `requests`
            return self._session.get(url, headers=headers, allow_redirects=False, timeout=self.timeout)

    def _resolve(self, path):
        """Follow the Hub's redirects (asking for one byte) to the storage URL of a file."""
        from urllib.parse import urljoin, urlparse

        url = self._hub_url(path)
        for _ in range(6):
            if not urlparse(url).netloc.endswith("huggingface.co"):
                return url
            response = self._get(url, {**self._headers, "Range": "bytes=0-0"})
            self.hub_requests += 1
            if response.status_code == 429:
                time.sleep(float(response.headers.get("Retry-After", 30)))
                continue
            if response.status_code not in self.REDIRECTS:
                if response.status_code in (200, 206):
                    return url   # the Hub serves this file itself
                raise OSError(f"HTTP {response.status_code} resolving {path}")
            url = urljoin(url, response.headers["Location"])
        raise OSError(f"too many redirects resolving {path}")

    def read(self, path, start, end):
        from urllib.parse import urlparse

        def fetch():
            url = self._storage_url.get(path)
            if url is None:
                url = self._storage_url[path] = self._resolve(path)
            headers = {"Range": f"bytes={start}-{end - 1}"}
            if urlparse(url).netloc.endswith("huggingface.co"):
                headers.update(self._headers)
            response = self._get(url, headers)
            if response.status_code == 429:
                time.sleep(float(response.headers.get("Retry-After", 30)))
                raise OSError(f"HTTP 429 for {path}")
            if response.status_code not in (200, 206):
                self._storage_url.pop(path, None)   # a signed URL that expired: resolve again on the retry
                raise OSError(f"HTTP {response.status_code} for a range read of {path}")
            data = response.content
            if len(data) != end - start:
                raise OSError(f"range read of {path} returned {len(data)} bytes, expected {end - start}")
            return data

        data = _with_retries(fetch)
        self.requests += 1
        self.bytes_read += len(data)
        return data


class _SparseFile(io.IOBase):
    """A read-only file made of the byte ranges fetched so far. A read outside them is fetched on
    demand and counted in `misses`, so tests can assert the range planning was exact."""

    def __init__(self, reader, path, size):
        self._reader, self._path, self._size = reader, path, int(size)
        self._parts = []  # (start, bytes)
        self._pos = 0
        self.misses = 0

    def add(self, start, data):
        self._parts.append((int(start), data))

    def clear(self, keep_from=None):
        self._parts = [p for p in self._parts if keep_from is not None and p[0] >= keep_from]

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self._pos

    def seek(self, offset, whence=0):
        self._pos = offset if whence == 0 else self._pos + offset if whence == 1 else self._size + offset
        return self._pos

    def read(self, n=-1):
        start = self._pos
        end = self._size if n is None or n < 0 else min(self._size, start + n)
        if end <= start:
            return b""
        for part_start, data in self._parts:
            if part_start <= start and end <= part_start + len(data):
                self._pos = end
                return bytes(data[start - part_start:end - part_start])
        self.misses += 1
        data = self._reader.read(self._path, start, end)
        self._pos = end
        return data


class SparseParquet:
    """One Parquet file read by byte range. `read(columns, row_groups)` downloads only the column
    chunks it needs."""

    def __init__(self, reader, path, size, footer_guess=1 << 20):
        import pyarrow.parquet as pq

        self.reader, self.path, self.size = reader, path, int(size)
        tail_start = max(0, self.size - footer_guess)
        tail = reader.read(path, tail_start, self.size)
        footer_len = int.from_bytes(tail[-8:-4], "little")
        # The Parquet reader starts by reading the last 64 KB, so the cached tail must cover that and the footer.
        needed = max(footer_len + 8, 1 << 16)
        if needed > len(tail) and tail_start > 0:
            tail_start = max(0, self.size - needed)
            tail = reader.read(path, tail_start, self.size)
        self._tail_start = tail_start
        self.file = _SparseFile(reader, path, self.size)
        self.file.add(tail_start, tail)
        self.parquet = pq.ParquetFile(self.file)
        self.meta = self.parquet.metadata
        self.columns = list(self.parquet.schema_arrow.names)

    @property
    def num_row_groups(self):
        return self.meta.num_row_groups

    def row_group_rows(self):
        return [self.meta.row_group(i).num_rows for i in range(self.meta.num_row_groups)]

    def _chunks(self, row_group, columns):
        group = self.meta.row_group(row_group)
        for i in range(group.num_columns):
            chunk = group.column(i)
            name = chunk.path_in_schema
            if any(name == c or name.startswith(c + ".") for c in columns):
                offset = chunk.data_page_offset
                if chunk.has_dictionary_page and chunk.dictionary_page_offset is not None:
                    offset = min(offset, chunk.dictionary_page_offset)
                yield offset, offset + chunk.total_compressed_size

    def column_bytes(self, columns, row_groups=None):
        """Compressed bytes of `columns` per row group (from the footer alone, nothing downloaded)."""
        groups = range(self.num_row_groups) if row_groups is None else row_groups
        return [sum(end - start for start, end in self._chunks(g, columns)) for g in groups]

    def read(self, columns, row_groups=None, max_gap=1 << 16):
        """An Arrow table of `columns` for `row_groups` (all by default)."""
        groups = list(range(self.num_row_groups)) if row_groups is None else list(row_groups)
        ranges = sorted(r for g in groups for r in self._chunks(g, columns))
        merged = []
        for start, end in ranges:
            if merged and start - merged[-1][1] <= max_gap:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        self.file.clear(keep_from=self._tail_start)
        for start, end in merged:
            self.file.add(start, self.reader.read(self.path, start, end))
        table = self.parquet.read_row_groups(groups, columns=list(columns))
        self.file.clear(keep_from=self._tail_start)
        return table


def image_bytes_column(table, column="image"):
    """The raw image bytes of an Arrow table's `image` column (a struct with a `bytes` field in
    Hugging Face image datasets, or a plain binary column)."""
    import pyarrow as pa

    array = table.column(column).combine_chunks()
    if pa.types.is_struct(array.type):
        array = array.field("bytes")
    return array.to_pylist()


def list_parquet_shards(fs, root):
    """Every Parquet file under `root` with its size, config (parent folder) and split (file-name
    prefix), sorted by path."""
    shards = []
    for path, info in fs.find(root, detail=True).items():
        if path.endswith(".parquet"):
            name = path.rsplit("/", 1)[-1]
            split = re.match(r"([A-Za-z_]+)", name).group(1) if re.match(r"([A-Za-z_]+)", name) else ""
            shards.append({"path": path, "size": int(info["size"]), "config": path.rsplit("/", 2)[-2],
                           "split": split})
    return sorted(shards, key=lambda s: s["path"])


# ---------------------------------------------------------------------------------------------------
# Identifiers
# ---------------------------------------------------------------------------------------------------
def prompt_id_of(prompt):
    """A stable id for a prompt (lower-cased, whitespace collapsed), or None for an empty one.
    Images made from the same prompt -- and the real photo whose caption the prompt came from --
    share it, so it is one of the keys splits are grouped by."""
    text = " ".join(str(prompt or "").lower().split())
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16] if text else None


def stable_fraction(key, seed=0):
    """A deterministic number in [0, 1) from a string key: the same key always lands in the same
    place, on any machine, in any session."""
    digest = hashlib.sha256(f"{seed}|{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2.0 ** 64


# ---------------------------------------------------------------------------------------------------
# Task 0: metadata-only scan and the timed streamed sample
# ---------------------------------------------------------------------------------------------------
def scan_shard_metadata(reader, shard, columns=META_COLUMNS):
    """Counts for one shard from its small columns only -- no image is downloaded. Returns
    aggregate counts plus per-row prompt ids (used in memory for group statistics, never saved)."""
    started = time.time()
    sparse = SparseParquet(reader, shard["path"], shard["size"])
    wanted = [c for c in columns if c in sparse.columns]
    table = sparse.read(wanted).to_pydict()
    n = sparse.meta.num_rows
    column = lambda name: table.get(name, [None] * n)  # noqa: E731
    labels, models = column("label"), column("model")
    return {
        "path": shard["path"], "config": shard["config"], "split": shard["split"], "size": shard["size"],
        "rows": n, "row_groups": sparse.num_row_groups, "row_group_rows": sparse.row_group_rows(),
        "image_bytes": int(sum(sparse.column_bytes(["image"]))),
        "columns": sparse.columns,
        "counts": Counter(zip(labels, models, column("type"), column("release_date"))),
        "label": labels, "model": models,
        "prompt_id": [prompt_id_of(p) for p in column("prompt")],
        "seconds": time.time() - started, "sparse_misses": sparse.file.misses,
    }


def scan_all_shards(reader, shards, threads=16, max_seconds=None, progress_every=25, log=print):
    """`scan_shard_metadata` over many shards in parallel, with a progress line every few shards
    and an optional time budget (the shards not reached are reported, not hidden)."""
    started, done, failed, skipped = time.time(), [], [], []
    with ThreadPoolExecutor(max_workers=threads) as pool:
        futures = [(shard, pool.submit(scan_shard_metadata, reader, shard)) for shard in shards]
        for i, (shard, future) in enumerate(futures):
            if max_seconds is not None and time.time() - started > max_seconds and not future.done():
                future.cancel()
            try:
                if future.cancelled():
                    skipped.append(shard["path"])
                else:
                    done.append(future.result())
            except Exception as exc:  # noqa: BLE001 -- one bad shard is recorded, the scan goes on
                failed.append({"path": shard["path"], "error": f"{type(exc).__name__}: {exc}"[:300]})
            if progress_every and (i + 1) % progress_every == 0:
                log(f"  metadata scan {i + 1}/{len(shards)} shards, {sum(d['rows'] for d in done)} rows, "
                    f"{time.time() - started:.0f}s")
    return {"shards": done, "failed": failed, "skipped_for_time": skipped, "seconds": time.time() - started}


def prompt_group_stats(scan):
    """How prompts tie rows together, from a metadata scan: how many prompts have both a real and a
    fake row, how many span several generators, and how big the groups are."""
    by_prompt = {}
    empty = 0
    for shard in scan["shards"]:
        for label, model, prompt in zip(shard["label"], shard["model"], shard["prompt_id"]):
            if prompt is None:
                empty += 1
                continue
            entry = by_prompt.setdefault(prompt, [0, 0, set()])
            entry[1 if label == "fake" else 0] += 1
            if label == "fake":
                entry[2].add(model)
    sizes = np.array([e[0] + e[1] for e in by_prompt.values()] or [0])
    return {
        "rows_with_empty_prompt": empty,
        "distinct_prompts": len(by_prompt),
        "prompts_with_real_and_fake": sum(1 for e in by_prompt.values() if e[0] and e[1]),
        "prompts_with_fakes_from_2plus_models": sum(1 for e in by_prompt.values() if len(e[2]) >= 2),
        "prompts_with_one_row": int((sizes == 1).sum()),
        "group_size_quantiles": {q: float(np.quantile(sizes, float(q))) for q in ("0.5", "0.9", "0.99", "1.0")},
    }


def time_streamed_sample(reader, shards, max_images=3000, max_seconds=600, decode_threads=4, log=print):
    """Download and decode real row groups from at least two shards and time it. The second shard
    is entered in the middle (a seek), to show a row group can be fetched without the ones before
    it. Images are decoded, hashed and pushed through the shared pipeline, then dropped."""
    started = time.time()
    out = {"images": 0, "image_bytes": 0, "download_seconds": 0.0, "decode_seconds": 0.0, "row_groups": [],
           "invalid": Counter(), "format": Counter(), "mode": Counter(), "sizes": Counter(), "labels": Counter()}
    widths, heights, file_sizes = [], [], []
    with ThreadPoolExecutor(max_workers=decode_threads) as pool:
        for k, shard in enumerate(shards):
            sparse = SparseParquet(reader, shard["path"], shard["size"])
            first = 0 if k == 0 else sparse.num_row_groups // 2
            for group in range(first, sparse.num_row_groups):
                if out["images"] >= max_images * (k + 1) / len(shards) or time.time() - started > max_seconds:
                    break
                t0 = time.time()
                table = sparse.read(["image", "label"], [group])
                blobs = image_bytes_column(table)
                t1 = time.time()
                records = list(pool.map(inspect_image, blobs))
                t2 = time.time()
                nbytes = sum(len(b) for b in blobs)
                out["images"] += len(blobs)
                out["image_bytes"] += nbytes
                out["download_seconds"] += t1 - t0
                out["decode_seconds"] += t2 - t1
                out["row_groups"].append({"shard": k, "row_group": group, "rows": len(blobs), "mb": round(nbytes / 1e6, 1),
                                          "download_s": round(t1 - t0, 2), "decode_s": round(t2 - t1, 2),
                                          "entered_by_seek": bool(k > 0 and group == first)})
                out["labels"].update(table.column("label").to_pylist())
                for r in records:
                    out["format"][r["format"]] += 1
                    out["mode"][r["mode"]] += 1
                    if r["ok"]:
                        widths.append(r["width"])
                        heights.append(r["height"])
                        file_sizes.append(r["file_size"])
                        out["sizes"][f"{r['width']}x{r['height']}"] += 1
                    else:
                        out["invalid"][r["reason"]] += 1
                log(f"  sample: shard {k} row group {group}: {len(blobs)} images, {nbytes / 1e6:.0f} MB, "
                    f"download {t1 - t0:.1f}s, decode {t2 - t1:.1f}s")
            out["sparse_misses"] = out.get("sparse_misses", 0) + sparse.file.misses
    seconds = out["download_seconds"] + out["decode_seconds"]
    out.update(
        wall_seconds=time.time() - started,
        mb_per_second=out["image_bytes"] / 1e6 / out["download_seconds"] if out["download_seconds"] else None,
        decode_images_per_second=out["images"] / out["decode_seconds"] if out["decode_seconds"] else None,
        images_per_second_sequential=out["images"] / seconds if seconds else None,
        mean_mb_per_image=out["image_bytes"] / 1e6 / out["images"] if out["images"] else None,
        width_quantiles=_quantiles(widths), height_quantiles=_quantiles(heights),
        file_size_quantiles=_quantiles(file_sizes), decode_threads=decode_threads,
        sizes=dict(out["sizes"].most_common(12)),
    )
    for key in ("invalid", "format", "mode", "labels"):
        out[key] = {str(k): v for k, v in out[key].items()}
    return out


def _quantiles(values, qs=(0.05, 0.5, 0.95)):
    return {str(q): float(np.quantile(values, q)) for q in qs} if len(values) else {}


def shards_to_fill_quota(per_shard_counts, quota):
    """Taking whole shards in the given order, how many are needed before each generator has
    `quota` rows. `per_shard_counts` is a list (in read order) of {generator: rows}. Returns
    {generator: shards needed, or None if the listed shards never reach the quota}."""
    running, needed = Counter(), {}
    generators = set().union(*[set(c) for c in per_shard_counts]) if per_shard_counts else set()
    for i, counts in enumerate(per_shard_counts, 1):
        running.update(counts)
        for g in generators:
            if g not in needed and running[g] >= quota:
                needed[g] = i
    return {g: needed.get(g) for g in sorted(generators)}


# ---------------------------------------------------------------------------------------------------
# Folder datasets (ArtiFact and the FFHQ thumbnails are attached to the Kaggle notebook as files)
# ---------------------------------------------------------------------------------------------------
def find_kaggle_inputs(base="/kaggle/input", depth=3, max_entries=500):
    """Attached dataset folders, nearest first, down to `depth` levels (Kaggle has mounted datasets
    at different depths over time). A folder with more than `max_entries` entries is listed but not
    opened further, so a folder of a hundred thousand images costs nothing."""
    found, level = [], [str(base)] if os.path.isdir(base) else []
    for _ in range(depth):
        below = []
        for folder in level:
            try:
                with os.scandir(folder) as entries:
                    children = []
                    for n, entry in enumerate(entries):
                        if n >= max_entries:
                            children = []
                            break
                        if entry.is_dir(follow_symlinks=False):
                            children.append(entry.path)
            except OSError:
                continue
            below.extend(sorted(children))
        found.extend(below)
        level = below
    return [f.replace("\\", "/") for f in found]


def match_input(folders, keywords, not_under=None):
    """The first attached folder whose name contains every one of `keywords` (case-insensitive),
    skipping `not_under` and anything inside it."""
    for folder in folders:
        name = Path(folder).name.lower()
        inside = bool(not_under) and (Path(folder).as_posix() + "/").startswith(Path(not_under).as_posix().rstrip("/") + "/")
        if all(k in name for k in keywords) and not inside:
            return folder
    return None


def summarize_folder(root, max_seconds=900, progress_every=200000, log=print):
    """File counts and bytes under `root`, by top-level folder and extension, plus the CSV files
    found. Stops at the time budget and says so."""
    started, root = time.time(), str(root)
    summary = {"root": root, "files": 0, "bytes": 0, "by_top_dir": {}, "csv_files": [], "complete": True}
    stack = [root]
    while stack:
        if time.time() - started > max_seconds:
            summary["complete"] = False
            break
        folder = stack.pop()
        try:
            entries = list(os.scandir(folder))
        except OSError:
            continue
        for entry in entries:
            if entry.is_dir(follow_symlinks=False):
                stack.append(entry.path)
                continue
            rel = os.path.relpath(entry.path, root).replace("\\", "/")
            top = rel.split("/", 1)[0] if "/" in rel else "."
            ext = os.path.splitext(entry.name)[1].lower()
            bucket = summary["by_top_dir"].setdefault(top, {"files": 0, "ext": {}})
            bucket["files"] += 1
            bucket["ext"][ext] = bucket["ext"].get(ext, 0) + 1
            summary["files"] += 1
            summary["bytes"] += entry.stat(follow_symlinks=False).st_size
            if ext == ".csv":
                summary["csv_files"].append(rel)
            if progress_every and summary["files"] % progress_every == 0:
                log(f"  walked {summary['files']} files under {root}, {time.time() - started:.0f}s")
    summary["seconds"] = time.time() - started
    return summary


def describe_csv(path, max_rows=None, max_distinct=40):
    """Column names, row count and the value counts of every low-cardinality column of a CSV."""
    counts, rows, columns = {}, 0, []
    with open(path, newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        for row in reader:
            rows += 1
            for name in columns:
                seen = counts.setdefault(name, Counter())
                if seen is not None and len(seen) <= max_distinct:
                    seen[row.get(name)] += 1
            if max_rows and rows >= max_rows:
                break
    return {"columns": columns, "rows": rows,
            "value_counts": {k: dict(v) for k, v in counts.items() if len(v) <= max_distinct}}


def list_images(root):
    """Every image file under `root`, as sorted paths."""
    found = []
    for folder, _, names in os.walk(root):
        found.extend(os.path.join(folder, n) for n in names if n.lower().endswith(IMAGE_EXTENSIONS))
    return sorted(found)


def image_header(path):
    """(width, height, mode, format) from the file header alone -- no pixel decode."""
    try:
        with Image.open(path) as img:
            return img.size[0], img.size[1], img.mode, img.format
    except Exception:  # noqa: BLE001
        return None


def header_summary(paths, threads=8):
    """Size, mode and format counts over image files (header reads only)."""
    with ThreadPoolExecutor(max_workers=threads) as pool:
        headers = list(pool.map(image_header, paths))
    good = [h for h in headers if h is not None]
    return {"files": len(paths), "unreadable": len(headers) - len(good),
            "sizes": dict(Counter(f"{h[0]}x{h[1]}" for h in good).most_common(12)),
            "modes": dict(Counter(h[2] for h in good)), "formats": dict(Counter(h[3] for h in good))}


def index_artifact(root):
    """One row per ArtiFact image from the `metadata.csv` in each generator folder. The layout
    (columns `image_path` and `target`, 0 = real) is the one the dataset's README describes; Task 0
    prints the real one, and this function raises if it differs rather than guessing."""
    rows = []
    for folder in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        meta = folder / "metadata.csv"
        if not meta.exists():
            continue
        with open(meta, newline="", encoding="utf-8", errors="replace") as handle:
            reader = csv.DictReader(handle)
            if not {"image_path", "target"} <= set(reader.fieldnames or []):
                raise ValueError(f"{meta} has columns {reader.fieldnames}, expected image_path and target")
            for row in reader:
                rel = row["image_path"].replace("\\", "/")
                rows.append({"image_id": f"artifact:{folder.name}/{rel}", "source_dataset": "artifact",
                             "generator": folder.name, "is_fake": int(str(row["target"]).strip() != "0"),
                             "category": row.get("category"), "path": str(folder / rel)})
    return rows


def index_ffhq(root):
    """One row per FFHQ thumbnail. Each FFHQ image is a different person's photo, so the file stem
    is both the source image id and the identity id."""
    return [{"image_id": f"ffhq:{Path(p).stem}", "source_dataset": "ffhq", "generator": "ffhq", "is_fake": REAL,
             "source_image_id": f"ffhq:{Path(p).stem}", "identity_id": f"ffhq:{Path(p).stem}", "path": p}
            for p in list_images(root)]


def read_file_bytes(path):
    with open(path, "rb") as handle:
        return handle.read()


# ---------------------------------------------------------------------------------------------------
# Synthetic data (unit tests and the notebook smoke run -- never a reported number)
# ---------------------------------------------------------------------------------------------------
SYNTHETIC_GENERATORS = {  # name -> (type, release date)
    "sd-1.5": ("base", "2022-10"), "sd-2.1": ("base", "2022-12"), "sdxl": ("base", "2023-07"),
    "sdxl-juggernaut": ("finetune", "2023-11"), "flux.1-dev": ("base", "2024-08"),
    "flux.1-schnell": ("base", "2024-08"), "midjourney-6": ("base", "2023-12"),
    "midjourney-7": ("base", "2025-04"), "gpt-image-1": ("base", "2025-04"),
}


def synthetic_picture(rng, fake, size=(160, 128), strength=40.0):
    """A smooth random picture. Fakes get a faint periodic grid on top -- a planted, learnable
    difference, so the heads have something to find."""
    w, h = size
    coarse = rng.uniform(0, 255, size=(6, 6, 3))
    img = Image.fromarray(coarse.astype(np.uint8)).resize((w, h), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float64) + rng.normal(0, 4.0, size=(h, w, 3))
    if fake:
        yy, xx = np.mgrid[0:h, 0:w]
        arr += strength * (((xx // 4) + (yy // 4)) % 2)[..., None]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _encode(img, fmt, **kwargs):
    buffer = io.BytesIO()
    img.save(buffer, format=fmt, **kwargs)
    return buffer.getvalue()


def make_synthetic_openfake(n_prompts=60, fakes_per_prompt=4, seed=0, with_problems=True):
    """OpenFake-shaped rows: one real photo per prompt and fakes of the same prompt from several
    generators. Reals are JPEGs of varying size with EXIF, fakes are fixed-size PNGs -- the planted
    file-property shortcut the baseline must catch. With `with_problems`, a few rows are an exact
    duplicate, a re-encoded near duplicate, a corrupt file, a tiny image, a grayscale image and an
    RGBA image."""
    rng = np.random.default_rng(seed)
    names = list(SYNTHETIC_GENERATORS)
    rows = []

    def add(image_bytes, label, model, kind, date, prompt):
        rows.append({"image": {"bytes": image_bytes, "path": None}, "prompt": prompt, "label": label,
                     "model": model, "type": kind, "release_date": date, "row_id": f"row{len(rows):06d}",
                     "image_sha256": hashlib.sha256(image_bytes).hexdigest()})

    for p in range(n_prompts):
        prompt = f"A photograph of scene number {p} with people in a public place"
        size = (int(rng.integers(140, 260)), int(rng.integers(120, 220)))
        exif = Image.Exif()
        exif[0x010F] = "SyntheticCam"
        real = _encode(synthetic_picture(rng, False, size), "JPEG", quality=int(rng.integers(70, 96)), exif=exif)
        add(real, "real", "laion" if p % 2 else "pexels", "real", "2021-08", prompt)
        for g in rng.choice(len(names), size=min(fakes_per_prompt, len(names)), replace=False):
            kind, date = SYNTHETIC_GENERATORS[names[g]]
            add(_encode(synthetic_picture(rng, True, (192, 192)), "PNG"), "fake", names[g], kind, date, prompt)
    if with_problems:
        first_real = rows[0]["image"]["bytes"]
        add(first_real, "real", "laion", "real", "2021-08", "an exact duplicate of the first real photo")
        again = _encode(Image.open(io.BytesIO(first_real)).resize((120, 100)), "JPEG", quality=60)
        add(again, "real", "laion", "real", "2021-08", "a re-encoded copy of the first real photo")
        add(b"not an image at all", "fake", names[0], "base", "2022-10", "a corrupt file")
        add(_encode(synthetic_picture(rng, True, (32, 32)), "PNG"), "fake", names[0], "base", "2022-10", "a tiny image")
        add(_encode(synthetic_picture(rng, False, (160, 128)).convert("L"), "PNG"), "real", "laion", "real",
            "2021-08", "a grayscale image")
        add(_encode(synthetic_picture(rng, True, (192, 192)).convert("RGBA"), "PNG"), "fake", names[2], "base",
            "2023-07", "an image with an alpha channel")
    return rows


def write_synthetic_openfake(root, rows, n_shards=3, rows_per_group=16, seed=0):
    """Write rows as Hugging-Face-style Parquet shards (`core/train-0000i-of-0000n.parquet`, several
    row groups each, rows shuffled across shards like the real dataset). Returns the shard paths."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    folder = Path(root) / "core"
    folder.mkdir(parents=True, exist_ok=True)
    order = np.random.default_rng(seed).permutation(len(rows))
    paths = []
    for i in range(n_shards):
        part = [rows[j] for j in order[i::n_shards]]
        path = folder / f"train-{i:05d}-of-{n_shards:05d}.parquet"
        pq.write_table(pa.Table.from_pylist(part), path, row_group_size=rows_per_group)
        paths.append(str(path))
    return paths


def write_synthetic_artifact(root, per_folder=12, seed=1):
    """An ArtiFact-shaped folder: one sub-folder per generator or real source, each with a
    `metadata.csv` (filename, image_path, target, category) and small JPEGs."""
    rng = np.random.default_rng(seed)
    root = Path(root)
    for name, target in (("stylegan2", 1), ("ddpm", 2), ("coco", 0), ("ffhq", 0)):
        (root / name / "images").mkdir(parents=True, exist_ok=True)
        with open(root / name / "metadata.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["filename", "image_path", "target", "category"])
            for i in range(per_folder):
                filename = f"img{i:05d}.jpg"
                synthetic_picture(rng, target != 0, (100, 100)).save(root / name / "images" / filename, quality=90)
                writer.writerow([filename, f"images/{filename}", target, "scene"])
    return str(root)


def write_synthetic_ffhq(root, n=16, seed=2):
    """An FFHQ-thumbnails-shaped folder: `n` 128x128 RGB PNGs of real 'faces'."""
    rng = np.random.default_rng(seed)
    folder = Path(root) / "thumbnails128x128"
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        synthetic_picture(rng, False, (128, 128)).save(folder / f"{i:05d}.png")
    return str(root)


def save_json(path, value, secrets=None):
    """Write JSON with secrets scrubbed and numpy scalars converted."""
    Path(path).write_text(json.dumps(scrub_secrets(value, secrets), indent=2, default=_jsonable), encoding="utf-8")


def _jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)
