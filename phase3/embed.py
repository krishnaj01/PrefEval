"""Embed every unique query / document text once per encoder (sentence-transformers, official prompts from encoders.yaml).

    python -m phase3.embed --encoder all            # or e.g. --encoder bge_base
    python -m phase3.embed --encoder bge_base --check   # determinism check only (re-encode a sample, compare)

Cache layout (git-ignored, reproducible): data/phase3/emb/<encoder>/
    shard_<kind>_<k>.npz   hashes (sha256 of the raw text) + L2-normalised float32 vectors, written chunk by chunk (resume unit)
    meta.json              hf_id, resolved commit, prompts, dim, latency, versions
Embedding each message independently is exactly what a live system would do turn by turn (bi-encoder), so
pre-computing changes nothing about the results; the per-message latency is measured separately.
"""

import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")   # must be set before huggingface_hub is imported (memory, see local_snapshot)

import argparse  # noqa: E402
import glob
import json
import logging
import time

import numpy as np

from phase3.common import (EMB_DIR, encoder_names, encoders_config, env_versions, now, retrieval_config, sha256_text,
                           write_run_manifest)
from phase3.conversations import all_texts

CHUNK = 1024
log = logging.getLogger("phase3.embed")


# ----------------------------------------------------------------------------- cache
def _shards(enc, kind):
    return sorted(glob.glob(os.path.join(EMB_DIR, enc, f"shard_{kind}_*.npz")))


def load_cache(enc, kind):
    """dict sha256(text) -> unit vector, for kind in {query, doc}."""
    out = {}
    for p in _shards(enc, kind):
        z = np.load(p)
        out.update(zip(z["hashes"].tolist(), z["vecs"]))
    return out


def load_meta(enc):
    p = os.path.join(EMB_DIR, enc, "meta.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


# ----------------------------------------------------------------------------- model
# Only model.safetensors is needed (all five ship it): skip the duplicate weight formats (pytorch_model.bin, ONNX, OpenVINO,
# GGUF, TF/Flax), which would add several GB per model.
SKIP_FILES = ["onnx/*", "openvino/*", "gguf/*", "coreml/*", "*.onnx", "*.bin", "*.gguf", "*.h5", "*.msgpack", "*.ot"]


def local_snapshot(hf_id, revision):
    """Folder of the pinned commit in the local HF cache; downloaded once if missing. Loading from this folder means
    no network access at all on later runs (the model runs locally in this Python process; Ollama is not involved).
    A marker records a *complete* download: a cache folder can exist with only a few files (e.g. one config fetched
    on its own), which must not count as downloaded."""
    from huggingface_hub import snapshot_download
    marker = os.path.join(EMB_DIR, "_snapshots", f"{hf_id.replace('/', '__')}@{revision}")
    if os.path.exists(marker):        # not snapshot_download(local_files_only=True): it rejects the skipped ONNX files
        with open(marker) as f:
            path = f.read().strip()
        if os.path.exists(os.path.join(path, "config.json")):
            return path
    log.info("downloading %s@%s (one time)", hf_id, revision[:10])
    # one file at a time over plain HTTP (xet disabled at import, see top): the xet downloader ramped up to ~47 parallel
    # connections buffering in RAM, and systemd-oomd killed the desktop session (2026-10-24)
    path = snapshot_download(hf_id, revision=revision, ignore_patterns=SKIP_FILES, max_workers=1)
    os.makedirs(os.path.dirname(marker), exist_ok=True)
    with open(marker, "w") as f:
        f.write(path + "\n")
    return path


def load_model(enc):
    import torch
    from sentence_transformers import SentenceTransformer
    c = encoders_config()[enc]
    path = local_snapshot(c["hf_id"], c["revision"])
    m = SentenceTransformer(path, device="cuda" if torch.cuda.is_available() else "cpu", local_files_only=True,
                            model_kwargs={"dtype": torch.float32})
    m.max_seq_length = c["max_seq_length"]
    m.eval()
    return m, c


def encode(model, texts, prompt, batch_size):
    return model.encode(texts, prompt=prompt or None, batch_size=batch_size, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=False).astype(np.float32)


def count_truncated(c, prompt, texts):
    """Documents longer than max_seq_length (the encoder only sees their beginning). Uses the raw tokenizer, because
    the sentence-transformers one already truncates."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(local_snapshot(c["hf_id"], c["revision"]))
    tok.model_max_length = 10 ** 6       # silence the 'longer than maximum' warning; we only count
    return int(sum(len(tok(prompt + t)["input_ids"]) > c["max_seq_length"] for t in texts))


def measure_latency(model, texts, prompt, n, warmup=10):
    """Median / p95 ms to embed ONE message (batch 1), i.e. the cost added per incoming turn."""
    import torch
    sample = texts[:: max(1, len(texts) // n)][:n]
    for t in sample[:warmup]:
        encode(model, [t], prompt, 1)
    ms = []
    for t in sample:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        encode(model, [t], prompt, 1)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ms.append(1000 * (time.perf_counter() - t0))
    return {"median_ms": float(np.median(ms)), "p95_ms": float(np.percentile(ms, 95)), "n": len(ms)}


# ----------------------------------------------------------------------------- run
def embed_encoder(enc, rcfg):
    queries, docs = all_texts(rcfg["reply_llms"], max(rcfg["inter_turns"]))
    todo = {}
    for kind, texts in (("query", queries), ("doc", docs)):
        have = load_cache(enc, kind)
        todo[kind] = [t for t in texts if sha256_text(t) not in have]
        log.info("%s %s: %d texts, %d cached, %d to embed", enc, kind, len(texts), len(texts) - len(todo[kind]),
                 len(todo[kind]))
    meta = load_meta(enc)
    if not todo["query"] and not todo["doc"] and meta and "latency" in meta:
        log.info("%s: complete, skipping", enc)
        return meta

    model, c = load_model(enc)
    prompts = {"query": c["query_prompt"], "doc": c["document_prompt"]}
    os.makedirs(os.path.join(EMB_DIR, enc), exist_ok=True)
    t0 = time.time()
    for kind in ("query", "doc"):
        start = len(_shards(enc, kind))
        for k, i in enumerate(range(0, len(todo[kind]), CHUNK)):
            chunk = todo[kind][i:i + CHUNK]
            vecs = encode(model, chunk, prompts[kind], rcfg["encode_batch_size"])
            path = os.path.join(EMB_DIR, enc, f"shard_{kind}_{start + k:04d}.npz")
            np.savez(path + ".tmp.npz", hashes=np.array([sha256_text(t) for t in chunk]), vecs=vecs)
            os.replace(path + ".tmp.npz", path)     # atomic: a crash never leaves a half-written shard
            log.info("%s %s: %d/%d", enc, kind, min(i + CHUNK, len(todo[kind])), len(todo[kind]))
    encode_s = time.time() - t0

    meta = {"encoder": enc, "hf_id": c["hf_id"], "revision": c["revision"], "query_prompt": c["query_prompt"],
            "document_prompt": c["document_prompt"], "max_seq_length": c["max_seq_length"],
            "dim": int(model.get_embedding_dimension()),
            "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
            "n_query": len(queries), "n_doc": len(docs),
            "truncated_docs": count_truncated(c, prompts["doc"], docs),
            "encode_seconds_this_run": round(encode_s, 1),
            "latency": measure_latency(model, docs, prompts["doc"], rcfg["latency_samples"]),
            "updated": now(), "env": env_versions()}
    with open(os.path.join(EMB_DIR, enc, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    write_run_manifest({"step": "embed", "encoder": enc, "meta": {k: v for k, v in meta.items() if k != "env"}})
    log.info("%s: done (%d dim, %.1f ms/message)", enc, meta["dim"], meta["latency"]["median_ms"])
    del model
    import torch
    torch.cuda.empty_cache()
    return meta


def check_determinism(enc, rcfg, n=128):
    """Re-encode a fixed sample and compare with the cache (batching can change floats slightly, never rankings)."""
    queries, docs = all_texts(rcfg["reply_llms"], max(rcfg["inter_turns"]))
    model, c = load_model(enc)
    out = {}
    for kind, texts, prompt in (("query", queries, c["query_prompt"]), ("doc", docs, c["document_prompt"])):
        cache = load_cache(enc, kind)
        sample = texts[:: max(1, len(texts) // n)][:n]
        new = encode(model, sample, prompt, rcfg["encode_batch_size"])
        old = np.stack([cache[sha256_text(t)] for t in sample])
        out[kind] = {"max_abs_diff": float(np.abs(new - old).max()), "min_cosine": float((new * old).sum(1).min())}
    log.info("%s determinism: %s", enc, out)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", default="all", help="'all' or comma-separated names from configs/phase3/encoders.yaml")
    ap.add_argument("--check", action="store_true", help="only run the determinism check")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for name in ("httpx", "httpx2"):
        logging.getLogger(name).setLevel(logging.WARNING)
    rcfg = retrieval_config()
    encs = encoder_names() if a.encoder == "all" else a.encoder.split(",")
    for enc in encs:
        assert enc in encoders_config(), f"unknown encoder {enc}"
        check_determinism(enc, rcfg) if a.check else embed_encoder(enc, rcfg)


if __name__ == "__main__":
    main()
