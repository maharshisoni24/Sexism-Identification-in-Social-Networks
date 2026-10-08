"""
generate_embeddings.py — Batch Gemini embeddings for EXIST 2021 via REST.

Strategy: batchEmbedContents with 50 texts per call.
  3 API keys × 3 models = 9 buckets, each with own rate-limit cooldown.
  No gRPC — pure requests, works in background processes.

Fallback: paraphrase-multilingual-mpnet-base-v2 (768-dim, en+es).

Checkpointing: every batch to data/checkpoint_<split>.json.
"""

import argparse
import json
import os
import time
import numpy as np
import pandas as pd
import requests

# Load .env if present (pip install python-dotenv, or set env vars manually)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # fall back to environment variables already set

EMBED_DIM = 768
BATCH_SIZE = 50

# API keys loaded from .env (GEMINI_KEY_1 / GEMINI_KEY_2 / GEMINI_KEY_3)
# Copy .env.example to .env and fill in your keys before running.
API_KEYS = [k for k in [
    os.getenv("GEMINI_KEY_1"),
    os.getenv("GEMINI_KEY_2"),
    os.getenv("GEMINI_KEY_3"),
] if k]


GEMINI_MODELS = [
    "models/gemini-embedding-2",
    "models/gemini-embedding-001",
    "models/gemini-embedding-2-preview",
]

LOCAL_MODELS = [
    "paraphrase-multilingual-mpnet-base-v2",
    "all-mpnet-base-v2",
]

_bucket_cooldown: dict = {}   # (ki, mi) -> unix time when available again
_local_model_obj = None


# ---------------------------------------------------------------------------
# Gemini batch REST
# ---------------------------------------------------------------------------

def _buckets_by_priority():
    now = time.time()
    available, cooling = [], []
    for ki, key in enumerate(API_KEYS):
        for mi, model in enumerate(GEMINI_MODELS):
            ready_at = _bucket_cooldown.get((ki, mi), 0)
            if now >= ready_at:
                available.append((ki, mi, key, model))
            else:
                cooling.append((ready_at, ki, mi, key, model))
    yield from available
    for ready_at, ki, mi, key, model in sorted(cooling):
        yield (ki, mi, key, model)


def embed_batch_gemini(texts: list) -> list:
    """Embed a batch of texts. Returns list of 768-dim lists. Rotates buckets on rate limit."""
    for ki, mi, api_key, model_name in _buckets_by_priority():
        now = time.time()
        ready_at = _bucket_cooldown.get((ki, mi), 0)
        if now < ready_at:
            wait = ready_at - now
            if wait > 90:
                continue
            print(f"  [wait {wait:.0f}s] key{ki+1}/{model_name.split('/')[-1]}", flush=True)
            time.sleep(wait)

        short = model_name.replace("models/", "")
        url = (f"https://generativelanguage.googleapis.com/v1beta/"
               f"{model_name}:batchEmbedContents?key={api_key}")
        payload = {
            "requests": [
                {"model": model_name,
                 "content": {"parts": [{"text": t}]},
                 "taskType": "SEMANTIC_SIMILARITY",
                 "outputDimensionality": EMBED_DIM}
                for t in texts
            ]
        }
        try:
            resp = requests.post(url, json=payload, timeout=30)
        except requests.exceptions.Timeout:
            print(f"  [timeout] key{ki+1}/{short} — skip bucket 30s", flush=True)
            _bucket_cooldown[(ki, mi)] = time.time() + 30
            continue
        except Exception as e:
            print(f"  [conn error] {e}", flush=True)
            continue

        if resp.status_code == 429 or "RESOURCE_EXHAUSTED" in resp.text:
            _bucket_cooldown[(ki, mi)] = time.time() + 60
            print(f"  [rate limit] key{ki+1}/{short} cooldown 60s", flush=True)
            continue
        if not resp.ok:
            print(f"  [HTTP {resp.status_code}] key{ki+1}/{short}: {resp.text[:80]}", flush=True)
            _bucket_cooldown[(ki, mi)] = time.time() + 10
            continue

        return [e["values"] for e in resp.json()["embeddings"]]

    raise RuntimeError("All Gemini buckets exhausted")


# ---------------------------------------------------------------------------
# Local sentence-transformers fallback
# ---------------------------------------------------------------------------

def embed_batch_local(texts: list) -> list:
    global _local_model_obj
    from sentence_transformers import SentenceTransformer
    for name in LOCAL_MODELS:
        try:
            if getattr(_local_model_obj, "_mname", None) != name:
                print(f"  Loading local model: {name}", flush=True)
                _local_model_obj = SentenceTransformer(name)
                _local_model_obj._mname = name
            if _local_model_obj.get_sentence_embedding_dimension() != EMBED_DIM:
                _local_model_obj = None
                continue
            vecs = _local_model_obj.encode(texts, batch_size=64, show_progress_bar=True)
            return vecs.tolist()
        except Exception as e:
            print(f"  Local {name} failed: {e}", flush=True)
    raise RuntimeError("No local model worked")


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def load_checkpoint(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        print(f"  Checkpoint: {len(d)} done", flush=True)
        return d
    return {}


def save_checkpoint(path: str, done: dict):
    with open(path, "w") as f:
        json.dump(done, f)


# ---------------------------------------------------------------------------
# TSV
# ---------------------------------------------------------------------------

def load_tsv(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path, sep="\t", header=0,
        names=["test_case", "id", "source", "language", "text", "task1", "task2"],
        dtype=str,
    ).dropna(subset=["text"])
    df["id"] = df["id"].str.strip()
    df["text"] = df["text"].str.strip()
    return df


# ---------------------------------------------------------------------------
# Embed a split
# ---------------------------------------------------------------------------

def embed_split(df: pd.DataFrame, split: str, args) -> tuple:
    checkpoint_path = os.path.join(args.output_dir, f"checkpoint_{split}.json")
    done = load_checkpoint(checkpoint_path)

    ids = df["id"].tolist()
    texts = df["text"].tolist()
    total = len(ids)
    remaining = [(ids[i], texts[i]) for i in range(total) if ids[i] not in done]
    print(f"  {split}: {total} total, {len(done)} done, {len(remaining)} remaining", flush=True)

    if not remaining:
        print("  All done.", flush=True)
    elif args.backend == "local":
        all_texts = [t for _, t in remaining]
        all_ids = [sid for sid, _ in remaining]
        vecs = embed_batch_local(all_texts)
        for sid, vec in zip(all_ids, vecs):
            done[sid] = vec
        save_checkpoint(checkpoint_path, done)
    else:
        gemini_failed = False
        for batch_start in range(0, len(remaining), BATCH_SIZE):
            if gemini_failed:
                break
            batch = remaining[batch_start: batch_start + BATCH_SIZE]
            batch_ids = [sid for sid, _ in batch]
            batch_texts = [t for _, t in batch]

            try:
                vecs = embed_batch_gemini(batch_texts)
                for sid, vec in zip(batch_ids, vecs):
                    done[sid] = vec
            except RuntimeError as e:
                print(f"\n  Gemini exhausted: {e}\n  Switching to local...", flush=True)
                gemini_failed = True
                break

            save_checkpoint(checkpoint_path, done)
            n_done = len(done)
            pct = 100 * n_done / total
            ready = sum(1 for v in _bucket_cooldown.values() if v < time.time())
            total_buckets = len(API_KEYS) * len(GEMINI_MODELS)
            print(f"  [{split}] {n_done}/{total} ({pct:.0f}%)  buckets ready: {ready}/{total_buckets}", flush=True)
            time.sleep(args.sleep_secs)

        if gemini_failed:
            still_missing = [(sid, t) for sid, t in remaining if sid not in done]
            if still_missing:
                vecs = embed_batch_local([t for _, t in still_missing])
                for (sid, _), vec in zip(still_missing, vecs):
                    done[sid] = vec
                save_checkpoint(checkpoint_path, done)

    # Build ordered arrays
    all_ids_out, all_embeds = [], []
    for sid in ids:
        all_ids_out.append(int(sid))
        emb = done.get(sid)
        if emb:
            all_embeds.append(emb)
        else:
            print(f"  WARNING: missing id={sid}, zero-fill", flush=True)
            all_embeds.append([0.0] * EMBED_DIM)

    return all_ids_out, np.array(all_embeds, dtype=np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    for split, tsv_path in [("train", args.train_tsv), ("test", args.test_tsv)]:
        print(f"\n{'='*60}\n{split}: {tsv_path}", flush=True)
        df = load_tsv(tsv_path)
        ids, embeds = embed_split(df, split, args)

        if embeds.shape[1] != EMBED_DIM:
            raise ValueError(f"Expected dim={EMBED_DIM}, got {embeds.shape[1]}")

        out = os.path.join(args.output_dir, f"embeddings_{split}.safetensors")
        import torch
        from safetensors.torch import save_file
        save_file(
            {"ids": torch.tensor(ids, dtype=torch.int64),
             "embeddings": torch.tensor(embeds, dtype=torch.float32)},
            out,
        )
        print(f"  Saved {out}  shape={embeds.shape}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_tsv",  default="data/EXIST2021_training.tsv")
    parser.add_argument("--test_tsv",   default="data/EXIST2021_test.tsv")
    parser.add_argument("--output_dir", default="data/")
    parser.add_argument("--backend",    choices=["gemini", "local"], default="gemini")
    parser.add_argument("--sleep_secs", type=float, default=1.0,
                        help="Sleep between batch calls (1s = ~60 batches/min)")
    args = parser.parse_args()
    main(args)
