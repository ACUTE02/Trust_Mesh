"""Frozen CLIP image encoder and the resumable embedding cache (Task 5).

The encoder is frozen (eval mode, no gradients). Each image is embedded once; only the embedding
and the image id are kept -- never the image. Embeddings are written in chunks, one file per chunk,
so a session that stops can resume, and each chunk prints a progress line (issue #27: a silent long
run looks the same as a hang).
"""
import hashlib
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from .preprocess import decode_image
from .preprocess import preprocess_image
from .preprocess import variant_image

CLIP_MODEL = "openai/clip-vit-large-patch14"


class ClipEncoder:
    """CLIP's image tower with its projection head. `embed` returns the projected image embedding
    (the vector CLIP compares with text; 768 numbers for ViT-L/14 according to its config -- Task 0
    confirms the real figure), not L2-normalized."""

    def __init__(self, name=None, device=torch.device("cpu"), fp16=True, model=None):
        if model is None:
            from transformers import CLIPVisionModelWithProjection

            name = name or CLIP_MODEL
            model = CLIPVisionModelWithProjection.from_pretrained(name)
        self.device = device
        self.half = bool(fp16) and device.type == "cuda"
        self.model = model.to(device).eval()
        if self.half:
            self.model.half()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.name = name or type(model).__name__
        self.dim = int(self.model.config.projection_dim)
        self.pre_projection_dim = int(self.model.config.hidden_size)

    @torch.no_grad()
    def embed(self, pixels):
        """[batch, 3, 224, 224] float pixels -> float32 array [batch, dim]."""
        x = pixels.to(self.device)
        out = self.model(pixel_values=x.half() if self.half else x.float())
        return out.image_embeds.float().cpu().numpy()


def tiny_clip_for_smoke():
    """A randomly initialised, tiny CLIP image tower: the real transformers code path with no
    download. Never used for a reported number."""
    from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection

    config = CLIPVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                              image_size=224, patch_size=32, projection_dim=24)
    torch.manual_seed(0)
    return CLIPVisionModelWithProjection(config)


def embedding_is_deterministic(encoder, pixels):
    """Same input twice in eval mode -> the same embedding (max absolute difference returned)."""
    a, b = encoder.embed(pixels), encoder.embed(pixels)
    return {"identical": bool(np.array_equal(a, b)), "max_abs_difference": float(np.abs(a - b).max())}


class ChunkStore:
    """Embedding chunks on disk: `<name>-<key>.npz` holding image ids, variant labels and vectors.

    `resume_dirs` are extra folders to look in for chunks finished by an earlier session (on Kaggle,
    the previous version's output attached as an input). A chunk found anywhere counts as done.
    """

    def __init__(self, out_dir, name="emb", resume_dirs=()):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.dirs = [self.out_dir] + [Path(d) for d in resume_dirs if d and Path(d).is_dir()]

    def _file(self, key):
        return f"{self.name}-{key}.npz"

    def find(self, key):
        for folder in self.dirs:
            if (folder / self._file(key)).exists():
                return folder / self._file(key)
        return None

    def has(self, key):
        return self.find(key) is not None

    def save(self, key, image_ids, variants, embeddings):
        tmp = self.out_dir / (self._file(key) + ".tmp")
        with open(tmp, "wb") as handle:
            np.savez(handle, image_id=np.asarray(image_ids, dtype=str), variant=np.asarray(variants, dtype=str),
                     embedding=np.asarray(embeddings, dtype=np.float32))
        os.replace(tmp, self.out_dir / self._file(key))

    def keys(self):
        prefix = self.name + "-"
        return sorted({p.name[len(prefix):-4] for d in self.dirs for p in d.glob(f"{prefix}*.npz")})

    def load(self):
        """Every chunk concatenated: (image ids, variants, embeddings)."""
        ids, variants, vectors = [], [], []
        for key in self.keys():
            with np.load(self.find(key)) as chunk:
                ids.append(chunk["image_id"])
                variants.append(chunk["variant"])
                vectors.append(chunk["embedding"])
        if not ids:
            return np.array([], dtype=str), np.array([], dtype=str), np.zeros((0, 0), dtype=np.float32)
        return np.concatenate(ids), np.concatenate(variants), np.concatenate(vectors)

    def size_gb(self):
        return sum(p.stat().st_size for p in self.out_dir.glob(f"{self.name}-*.npz")) / 1e9


def _prepare_item(item, seed):
    image_id, data, variants = item
    record = decode_image(data)
    if not record["ok"]:
        return []
    return [(image_id, v, preprocess_image(variant_image(record["rgb"], v, image_id, seed))) for v in variants]


def embed_items(encoder, items, batch_size=64, seed=0, decode_threads=4):
    """(image_id, image bytes, [variants]) items -> (ids, variants, embeddings). Images are decoded
    a batch at a time, so only one batch of pixels is ever in memory."""
    ids, variants, vectors = [], [], []
    with ThreadPoolExecutor(max_workers=decode_threads) as pool:
        for start in range(0, len(items), batch_size):
            prepared = [p for group in pool.map(lambda it: _prepare_item(it, seed), items[start:start + batch_size])
                        for p in group]
            for b in range(0, len(prepared), batch_size):
                part = prepared[b:b + batch_size]
                vectors.append(encoder.embed(torch.stack([p[2] for p in part])))
                ids.extend(p[0] for p in part)
                variants.extend(p[1] for p in part)
    dim = encoder.dim
    return ids, variants, (np.concatenate(vectors) if vectors else np.zeros((0, dim), dtype=np.float32))


def extract_embeddings(encoder, chunks, store, batch_size=64, seed=0, decode_threads=4, log=print):
    """Run `embed_items` over chunks and save each one. `chunks` yields (key, load) where `load()`
    returns that chunk's items; a chunk already in the store is skipped without loading it."""
    started, done, skipped, vectors = time.time(), 0, 0, 0
    for key, load in chunks:
        if store.has(key):
            skipped += 1
            log(f"  embeddings: chunk {key} already done, skipped")
            continue
        t0 = time.time()
        items = load()
        ids, variants, embeddings = embed_items(encoder, items, batch_size, seed, decode_threads)
        store.save(key, ids, variants, embeddings)
        done += 1
        vectors += len(ids)
        log(f"  embeddings: chunk {key}: {len(items)} images -> {len(ids)} vectors in {time.time() - t0:.0f}s "
            f"(total {vectors} vectors, {time.time() - started:.0f}s, cache {store.size_gb():.2f} GB)")
    return {"chunks_done": done, "chunks_skipped": skipped, "vectors_written": vectors,
            "seconds": time.time() - started, "cache_gb": store.size_gb()}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
