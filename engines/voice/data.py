"""ASVspoof 2019 LA loading and train/test splits.

Dataset: `SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA` on HuggingFace -- parquet, no loading script,
71,237 clips, 16 kHz FLAC. IMPORTANT: it contains only the **evaluation** partition of LA
(7,355 bonafide / 63,882 spoof). The HF copy carrying the official train/dev/eval partitions
(`LanceaKing/asvspoof2019`) needs `trust_remote_code`, which current `datasets` versions no longer
support, so the official protocol (train on attacks A01-A06, test on unseen A07-A19) is not
reachable here. We therefore split the eval partition ourselves; see `speaker_disjoint_split` for
why the split is speaker-disjoint and what that does and does not fix.

Clips are streamed (not downloaded in full) so the subset size is a knob, and audio is fetched
undecoded and decoded here with soundfile -- decoding through the `datasets` Audio feature changed
backends across versions, and doing it ourselves keeps this working either way.
"""
import io
import json

import numpy as np

ASVSPOOF_DATASET = "SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA"
ASVSPOOF_SPLIT = "test"  # the dataset's only split: the LA evaluation partition
IN_THE_WILD_DATASET = "mueller91/In-The-Wild"
TARGET_SR = 16000
BONAFIDE, SPOOF = 1, 0  # label 1 = bonafide = the positive/target class for EER


def _decode_audio(blob):
    """Decode an undecoded `datasets` audio value to (float32 mono waveform, sample rate)."""
    import soundfile as sf

    if isinstance(blob, dict) and blob.get("bytes"):
        wave, sr = sf.read(io.BytesIO(blob["bytes"]), dtype="float32", always_2d=False)
    elif isinstance(blob, dict) and blob.get("array") is not None:
        wave, sr = np.asarray(blob["array"], dtype=np.float32), int(blob["sampling_rate"])
    else:
        path = blob["path"] if isinstance(blob, dict) else blob
        if isinstance(path, str) and "://" in path:
            # Archive-backed datasets (e.g. In-the-Wild, shipped as one zip) come back as chained fsspec
            # URLs like "zip://a.wav::hf://datasets/.../x.zip", which libsndfile can't open. fsspec can,
            # and it only range-reads the needed zip member, not the whole archive.
            import fsspec

            with fsspec.open(path, "rb") as handle:
                wave, sr = sf.read(io.BytesIO(handle.read()), dtype="float32", always_2d=False)
        else:
            wave, sr = sf.read(path, dtype="float32", always_2d=False)
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    return np.ascontiguousarray(wave, dtype=np.float32), int(sr)


def load_asvspoof_subset(n_clips, seed=42, shuffle_buffer=20000, dataset=ASVSPOOF_DATASET,
                         split=ASVSPOOF_SPLIT, progress_every=500):
    """Stream a shuffled subset of ASVspoof 2019 LA. Returns a list of clip dicts."""
    from datasets import Audio, load_dataset

    stream = load_dataset(dataset, split=split, streaming=True)
    stream = stream.cast_column("audio", Audio(decode=False))
    stream = stream.shuffle(seed=seed, buffer_size=shuffle_buffer).take(n_clips)

    clips = []
    for row in stream:
        wave, sr = _decode_audio(row["audio"])
        try:
            notes = json.loads(row.get("notes") or "{}")
        except (json.JSONDecodeError, TypeError):
            notes = {}
        # The dataset's ClassLabel is 0=bonafide, 1=spoof; we flip so 1 = bonafide = target class.
        clips.append({
            "waveform": wave,
            "sample_rate": sr,
            "label": BONAFIDE if int(row["label"]) == 0 else SPOOF,
            "speaker_id": notes.get("speaker_id") or "unknown",
            "utterance_id": notes.get("utterance_id") or row.get("path") or f"clip_{len(clips)}",
            "path": row.get("path"),
        })
        if progress_every and len(clips) % progress_every == 0:
            print(f"  loaded {len(clips)}/{n_clips} clips")
    return clips


def load_in_the_wild_clips(n_clips, seed=42, shuffle_buffer=2000, dataset=IN_THE_WILD_DATASET):
    """A handful of out-of-domain clips for the generalization spot-check. Column names on this
    dataset aren't documented, so the audio/label columns are detected rather than assumed."""
    from datasets import Audio, load_dataset

    stream = load_dataset(dataset, split="train", streaming=True)
    columns = list(next(iter(stream.take(1))).keys())
    audio_col = next((c for c in columns if "audio" in c.lower() or c.lower() in ("file", "path")), None)
    label_col = next((c for c in columns if c.lower() in ("label", "key", "class", "type", "target")), None)
    if audio_col is None:
        raise RuntimeError(f"no audio-looking column in {columns}")

    try:
        stream = stream.cast_column(audio_col, Audio(decode=False))
    except (ValueError, TypeError):
        pass
    stream = stream.shuffle(seed=seed, buffer_size=shuffle_buffer).take(n_clips)

    clips = []
    for row in stream:
        wave, sr = _decode_audio(row[audio_col])
        raw_label = row.get(label_col)
        label = None
        if isinstance(raw_label, str):
            label = BONAFIDE if raw_label.strip().lower() in ("bona-fide", "bonafide", "real", "genuine") else SPOOF
        elif isinstance(raw_label, (int, np.integer)):
            label = int(raw_label)  # unknown polarity; recorded but flagged in the results
        clips.append({"waveform": wave, "sample_rate": sr, "label": label,
                      "raw_label": raw_label, "label_column": label_col, "speaker_id": "unknown",
                      "utterance_id": f"itw_{len(clips)}", "path": row.get("path")})
    return clips


ASVSPOOF_PROTOCOL = "protocols/ASVspoof2019.LA.cm.eval.trl.txt"
IN_THE_WILD_ARCHIVE = "datasets/mueller91/In-The-Wild/release_in_the_wild.zip"
_BONAFIDE_WORDS = {"bona-fide", "bonafide", "bona fide", "real", "genuine"}
_SPOOF_WORDS = {"spoof", "fake", "deepfake", "synthetic"}


def load_asvspoof_table(dataset=ASVSPOOF_DATASET, split=ASVSPOOF_SPLIT, cache_dir=None):
    """Download the whole parquet once and return it with audio left undecoded.

    Phase 2 streamed a shuffled sample. Phase 2B keeps *every* bonafide clip, which means touching
    every row anyway, and random access by index into a local memory-mapped table beats a streamed
    shuffle buffer for that (streaming, not training, was the slow part of Phase 2).
    """
    from datasets import Audio, load_dataset

    table = load_dataset(dataset, split=split, cache_dir=cache_dir)
    return table.cast_column("audio", Audio(decode=False))


def table_metadata(table):
    """Labels, utterance and speaker IDs for every row, without decoding any audio."""
    labels = np.array([BONAFIDE if int(v) == 0 else SPOOF for v in table["label"]])
    notes = []
    for raw in table["notes"]:
        try:
            notes.append(json.loads(raw or "{}"))
        except (json.JSONDecodeError, TypeError):
            notes.append({})
    paths = list(table["path"])
    utterances = [n.get("utterance_id") or (p.rsplit(".", 1)[0] if p else f"row_{i}")
                  for i, (n, p) in enumerate(zip(notes, paths))]
    return {
        "label": labels,
        "utterance_id": utterances,
        "speaker_id": [n.get("speaker_id") or "unknown" for n in notes],
        "columns": list(table.features),
        "note_keys": sorted({k for n in notes for k in n}),
    }


def decode_row(table, index):
    return _decode_audio(table[int(index)]["audio"])


def load_eval_protocol(path=None, dataset=ASVSPOOF_DATASET, filename=ASVSPOOF_PROTOCOL):
    """Official ASVspoof CM protocol: one line per clip, `SPEAKER UTTERANCE - ATTACK KEY`.

    The parquet has no attack column, but the same HF repo ships this file, which is the source of
    attack IDs (A07-A19). It is only trusted after `verify_protocol` passes.
    """
    if path is None:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=dataset, filename=filename, repo_type="dataset")
    protocol = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) != 5:
                continue
            speaker, utterance, _, attack, key = parts
            protocol[utterance] = {
                "speaker_id": speaker,
                "attack": "bonafide" if attack == "-" else attack,
                "label": BONAFIDE if key == "bonafide" else SPOOF,
            }
    return protocol, str(path)


def verify_protocol(protocol, utterance_ids, labels, speaker_ids):
    """The join is only usable if it reproduces the dataset's own labels and speakers exactly --
    a protocol file for a different partition, or misaligned IDs, fails this loudly."""
    n = len(utterance_ids)
    found = label_ok = speaker_ok = 0
    bonafide_clean = True
    for utterance, label, speaker in zip(utterance_ids, labels, speaker_ids):
        entry = protocol.get(utterance)
        if entry is None:
            continue
        found += 1
        label_ok += entry["label"] == label
        speaker_ok += entry["speaker_id"] == speaker
        if label == BONAFIDE and entry["attack"] != "bonafide":
            bonafide_clean = False
    return {
        "clips": n,
        "found_in_protocol": found,
        "label_agreement": label_ok,
        "speaker_agreement": speaker_ok,
        "bonafide_rows_have_no_attack": bonafide_clean,
        "usable": found == n and label_ok == n and speaker_ok == n and bonafide_clean,
    }


def select_subset(labels, spoof_per_bonafide, seed=42):
    """Every bonafide clip plus a random sample of spoof clips capped at `spoof_per_bonafide` per
    bonafide clip. If a budget forces a smaller set, lower the ratio -- bonafide is never cut."""
    labels = np.asarray(labels)
    bonafide = np.flatnonzero(labels == BONAFIDE)
    spoof = np.flatnonzero(labels == SPOOF)
    n_spoof = min(len(spoof), int(round(spoof_per_bonafide * len(bonafide))))
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([bonafide, rng.choice(spoof, n_spoof, replace=False)]))


def choose_held_out_attacks(attacks, n_held_out, seed=42):
    spoof_attacks = sorted({a for a in attacks if a != "bonafide"})
    n = max(1, min(n_held_out, len(spoof_attacks) - 1))  # leave at least one attack for training
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(spoof_attacks, size=n, replace=False).tolist())


def drop_attacks(indices, attacks, held_out):
    """Train/val side of the attack-disjoint protocol: bonafide plus spoofs from non-held-out attacks."""
    held = set(held_out)
    return [i for i in indices if attacks[i] == "bonafide" or attacks[i] not in held]


def keep_only_attacks(indices, attacks, held_out):
    """Test side of the attack-disjoint protocol: bonafide plus spoofs from held-out attacks only."""
    held = set(held_out)
    return [i for i in indices if attacks[i] == "bonafide" or attacks[i] in held]


class ZipAudioArchive:
    """Members of a zip that may be remote (hf://, https://), read without downloading it whole.

    One zip filesystem is opened and reused, so the central directory of a multi-GB archive is read
    once and each clip is a single range read -- reopening a chained `zip://...::hf://...` URL per clip
    (what Phase 2's loader effectively did) re-reads the directory every time.
    """

    def __init__(self, archive, protocol="hf"):
        import fsspec

        self.archive, self.protocol = archive, protocol
        self.fs = fsspec.filesystem("zip", fo=archive, target_protocol=protocol)

    def members(self):
        return sorted(self.fs.find(""))

    def read_bytes(self, member):
        with self.fs.open(member, "rb") as handle:
            return handle.read()


def read_archive_metadata(archive):
    """Find the label metadata inside the archive (In-the-Wild ships `meta.csv` with file/label
    columns) rather than assuming a HuggingFace column schema that isn't documented."""
    import pandas as pd

    members = archive.members()
    csvs = [m for m in members if m.lower().endswith(".csv")]
    if not csvs:
        raise RuntimeError(f"no metadata .csv among {len(members)} archive members")
    meta_member = next((m for m in csvs if m.rsplit("/", 1)[-1].lower() == "meta.csv"), csvs[0])
    df = pd.read_csv(io.BytesIO(archive.read_bytes(meta_member)))

    def looks_like_files(col):
        return df[col].astype(str).str.lower().str.endswith((".wav", ".flac", ".mp3")).mean() > 0.9

    def looks_like_labels(col):
        values = set(df[col].astype(str).str.strip().str.lower().unique())
        return bool(values) and values <= (_BONAFIDE_WORDS | _SPOOF_WORDS)

    file_col = next((c for c in df.columns if looks_like_files(c)), None)
    label_col = next((c for c in df.columns if looks_like_labels(c)), None)
    if file_col is None or label_col is None:
        raise RuntimeError(f"could not identify file/label columns in {meta_member}: {list(df.columns)}")

    prefix = meta_member.rsplit("/", 1)[0] + "/" if "/" in meta_member else ""
    raw = df[label_col].astype(str).str.strip()
    table = pd.DataFrame({
        "member": prefix + df[file_col].astype(str),
        "raw_label": raw,
        "label": np.where(raw.str.lower().isin(_BONAFIDE_WORDS), BONAFIDE, SPOOF),
    })
    info = {"metadata_member": meta_member, "file_column": file_col, "label_column": label_col,
            "raw_label_values": sorted(raw.unique().tolist()), "rows": int(len(table)),
            "rows_bonafide": int((table.label == BONAFIDE).sum()), "rows_spoof": int((table.label == SPOOF).sum())}
    return table, info


def load_archive_clips_balanced(archive, n_per_class, seed=42, max_failures=None):
    """`n_per_class` clips of each class, read one by one; every failure is recorded, not hidden."""
    import soundfile as sf

    table, info = read_archive_metadata(archive)
    max_failures = max_failures if max_failures is not None else 2 * n_per_class
    rng = np.random.default_rng(seed)
    clips, failures = [], []
    for label in (BONAFIDE, SPOOF):
        pool = table[table.label == label].reset_index(drop=True)
        loaded = 0
        for pos in rng.permutation(len(pool)):
            if loaded >= n_per_class:
                break
            row = pool.iloc[int(pos)]
            try:
                wave, sr = sf.read(io.BytesIO(archive.read_bytes(row.member)), dtype="float32", always_2d=False)
            except Exception as exc:  # noqa: BLE001 -- recorded per clip, and capped below
                failures.append({"member": row.member, "error": f"{type(exc).__name__}: {exc}"})
                if len(failures) > max_failures:
                    raise RuntimeError(f"{len(failures)} clips failed to load; first: {failures[0]}") from exc
                continue
            if wave.ndim > 1:
                wave = wave.mean(axis=1)
            clips.append({"waveform": np.ascontiguousarray(wave, dtype=np.float32), "sample_rate": int(sr),
                          "label": label, "raw_label": row.raw_label, "speaker_id": "unknown",
                          "utterance_id": row.member, "attack": "bonafide" if label == BONAFIDE else "unknown"})
            loaded += 1
    report = {**info, "requested_per_class": n_per_class, "loaded": len(clips),
              "loaded_bonafide": sum(c["label"] == BONAFIDE for c in clips),
              "loaded_spoof": sum(c["label"] == SPOOF for c in clips),
              "failed": len(failures), "failures": failures[:20]}
    return clips, report


def write_synthetic_archive(path, n_real=4, n_fake=5, include_broken=True, sample_rate=TARGET_SR):
    """In-the-Wild-shaped zip (release_in_the_wild/*.wav + meta.csv with file/speaker/label) for the
    local smoke test and unit tests -- exercises the same fsspec zip path as the real hf:// archive.
    `include_broken` adds one undecodable member so failure recording is exercised too."""
    import soundfile as sf
    import zipfile

    rows = ["file,speaker,label"]
    rng = np.random.default_rng(0)
    with zipfile.ZipFile(path, "w") as zf:
        for i in range(n_real + n_fake):
            label = "bona-fide" if i < n_real else "spoof"
            t = np.arange(int(1.5 * sample_rate)) / sample_rate
            wave = (0.2 * np.sin(2 * np.pi * rng.uniform(100, 300) * t)).astype(np.float32)
            buf = io.BytesIO()
            sf.write(buf, wave, sample_rate, format="WAV")
            zf.writestr(f"release_in_the_wild/{i}.wav", buf.getvalue())
            rows.append(f"{i}.wav,Speaker {i % 3},{label}")
        if include_broken:
            zf.writestr("release_in_the_wild/broken.wav", b"not audio")
            rows.append("broken.wav,Speaker 0,spoof")
        zf.writestr("release_in_the_wild/meta.csv", "\n".join(rows))
    return str(path)


def write_synthetic_protocol(clips, path):
    """Protocol file in the official format, for the smoke test's protocol-verification path."""
    with open(path, "w", encoding="utf-8") as handle:
        for clip in clips:
            attack = "-" if clip["label"] == BONAFIDE else clip["attack"]
            key = "bonafide" if clip["label"] == BONAFIDE else "spoof"
            handle.write(f"{clip['speaker_id']} {clip['utterance_id']} - {attack} {key}\n")
    return str(path)


def make_synthetic_asvspoof(n_clips=120, seed=0, sample_rate=TARGET_SR, n_attacks=6, bonafide_pad_s=(0.02, 0.1)):
    """Tiny ASVspoof-shaped set for local smoke tests only -- never used for reported numbers.

    Deliberately includes the failure mode the real data is suspected of: spoof clips get more
    leading/trailing silence than bonafide ones, so the silence-leakage check has something to find.
    Spoof clips get one of `n_attacks` synthetic attack IDs, drawn independently of speaker (as in real
    ASVspoof, where every target speaker is spoofed by every system), from a separate RNG so adding
    attacks didn't change the synthetic audio other tests rely on.
    """
    rng = np.random.default_rng(seed)
    attack_rng = np.random.default_rng(seed + 1)
    clips = []
    for i in range(n_clips):
        label = BONAFIDE if i % 4 == 0 else SPOOF  # ~25% bonafide, imbalanced like LA
        attack = "bonafide" if label == BONAFIDE else f"A{7 + int(attack_rng.integers(n_attacks)):02d}"
        speech_s = rng.uniform(1.0, 2.5)
        t = np.arange(int(speech_s * sample_rate), dtype=np.float32) / sample_rate
        f0 = rng.uniform(100, 200)
        speech = 0.3 * np.sin(2 * np.pi * f0 * t) + 0.1 * np.sin(2 * np.pi * 2 * f0 * t)
        if label == SPOOF:  # a crude high-frequency "vocoder artifact"
            speech += 0.05 * np.sin(2 * np.pi * 6500 * t)
        speech += 0.01 * rng.standard_normal(speech.shape).astype(np.float32)
        pad_s = rng.uniform(0.4, 0.8) if label == SPOOF else rng.uniform(*bonafide_pad_s)
        pad = 1e-4 * rng.standard_normal(int(pad_s * sample_rate)).astype(np.float32)
        clips.append({
            "waveform": np.concatenate([pad, speech, pad]).astype(np.float32),
            "sample_rate": sample_rate,
            "label": label,
            "speaker_id": f"SPK_{i % 12:02d}",
            "utterance_id": f"SYN_{i:04d}",
            "path": f"synthetic/{i}.wav",
            "attack": attack,
        })
    return clips


def class_counts(clips):
    labels = np.array([c["label"] for c in clips])
    return {"bonafide": int((labels == BONAFIDE).sum()), "spoof": int((labels == SPOOF).sum()),
            "total": int(len(labels))}


def pos_weight(clips):
    """Weight for the positive (bonafide, minority) class in BCE, as in the graph engine."""
    counts = class_counts(clips)
    return float(counts["spoof"] / max(counts["bonafide"], 1))


def _both_classes_present(clips, indices):
    labels = {clips[i]["label"] for i in indices}
    return BONAFIDE in labels and SPOOF in labels


def speaker_disjoint_split(clips, test_fraction=0.3, seed=42, attempts=12):
    """Split so no speaker appears in both train and test.

    A random clip-level split lets the same speaker land on both sides, and a model can then key on
    speaker identity/recording channel instead of spoofing artifacts -- the audio analogue of the
    Phase 1 random-vs-temporal split leak. Speakers are shuffled and sent whole to the test side
    until it reaches `test_fraction`, so a speaker is never split across sides.

    Caveat this does NOT fix: every clip here comes from the LA *evaluation* partition, so the same
    spoofing systems (A07-A19) appear on both sides. Unseen-attack generalization is not measured.
    """
    by_speaker = {}
    for idx, clip in enumerate(clips):
        by_speaker.setdefault(clip["speaker_id"], []).append(idx)

    target_test = test_fraction * len(clips)
    for attempt in range(attempts):
        rng = np.random.default_rng(seed + attempt)
        speakers = sorted(by_speaker)
        rng.shuffle(speakers)

        train_idx, test_idx, n_test = [], [], 0
        for speaker in speakers:
            members = by_speaker[speaker]
            # Send speakers to test until adding another would overshoot the target by more than it undershoots.
            if n_test + len(members) / 2 <= target_test:
                test_idx.extend(members)
                n_test += len(members)
            else:
                train_idx.extend(members)
        # Both sides need both classes, or EER is undefined on that side. Speakers are assigned whole,
        # so this isn't guaranteed by construction -- reshuffle and retry.
        if train_idx and test_idx and _both_classes_present(clips, train_idx) and _both_classes_present(clips, test_idx):
            return sorted(train_idx), sorted(test_idx)

    # Too few speakers (or a speaker holds an entire class) for a disjoint split with both classes on
    # both sides. Fall back to a random split rather than returning an unusable one; the caller's
    # split report will show shared_speakers > 0, making the fallback visible instead of silent.
    return random_split(clips, test_fraction, seed)


def random_split(clips, test_fraction=0.3, seed=42):
    """Clip-level random split -- speakers appear on both sides. Kept as the leakage comparison."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(clips))
    n_test = int(round(test_fraction * len(clips)))
    return sorted(order[n_test:].tolist()), sorted(order[:n_test].tolist())


def split_report(clips, train_idx, test_idx):
    train_speakers = {clips[i]["speaker_id"] for i in train_idx}
    test_speakers = {clips[i]["speaker_id"] for i in test_idx}
    return {
        "train": class_counts([clips[i] for i in train_idx]),
        "test": class_counts([clips[i] for i in test_idx]),
        "train_speakers": len(train_speakers),
        "test_speakers": len(test_speakers),
        "shared_speakers": len(train_speakers & test_speakers),
    }
