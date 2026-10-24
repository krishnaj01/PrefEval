"""Rank each question's candidate pool by cosine similarity to the question; store top-k ids and the gold rank.

    python -m phase3.retrieve --encoder all          # 5 encoders + simcse_paper (from stored scores); seconds each
    PHASE3_RESULTS_DIR=results/phase3_smoke python -m phase3.retrieve --encoder bge_base --topics travel_hotel --inter-turns 3,68

Output: results/phase3/retrieval/<encoder>/<pool>/inter<N>.jsonl, one row per question, where <pool> is
  explicit@<reply_llm>  (pool contains that LLM's own reply; simcse_paper uses explicit@paper = the authors' reply score)
  implicit-choice
Ties are broken by conversation order (stable sort), the same as upstream's `sorted(..., reverse=True)` for SimCSE.
"""

import argparse
import logging
import os

import numpy as np

from phase3.common import (RETRIEVAL_DIR, SIMCSE, encoder_names, retrieval_config, sha256_text, write_jsonl,
                           write_run_manifest)
from phase3.conversations import FORMS, pool, query, question_ids, simcse_available, simcse_scores

log = logging.getLogger("phase3.retrieve")


def pool_names(encoder, form, rcfg):
    if form == "implicit-choice":
        return [(form, None)]
    llms = ["paper"] if encoder == SIMCSE else rcfg["reply_llms"]
    return [(f"explicit@{llm}", llm) for llm in llms]


def out_path(encoder, pool_name, n):
    return os.path.join(RETRIEVAL_DIR, encoder, pool_name, f"inter{n}.jsonl")


def rank(scores):
    """Indices sorted by descending score, ties kept in conversation order."""
    return sorted(range(len(scores)), key=lambda i: -scores[i])


def row_for(topic, tid, form, n, cands, scores, top_k, simcse_ok):
    order = rank(scores)
    gold = next(i for i, m in enumerate(cands) if m["gold"])
    top = order[:top_k]
    return {"topic": topic, "task_id": tid, "n_candidates": len(cands), "gold_id": cands[gold]["id"],
            "gold_rank": order.index(gold) + 1, "top_ids": [cands[i]["id"] for i in top],
            "top_sources": [cands[i]["source"] for i in top], "top_scores": [round(float(scores[i]), 6) for i in top],
            "simcse_available": simcse_ok}


def run(encoder, rcfg, topics=None, inter_turns=None, force=False):
    from phase3.embed import load_cache
    qcache = dcache = None
    if encoder != SIMCSE:
        qcache, dcache = load_cache(encoder, "query"), load_cache(encoder, "doc")
        assert qcache and dcache, f"no embeddings for {encoder}: run `python -m phase3.embed --encoder {encoder}`"
    qids = [(t, i) for t, i in question_ids() if not topics or t in topics]
    written = 0
    for form in FORMS:
        for pool_name, llm in pool_names(encoder, form, rcfg):
            for n in inter_turns or rcfg["inter_turns"]:
                path = out_path(encoder, pool_name, n)
                if os.path.exists(path) and not force and not topics:
                    continue
                rows = []
                for topic, tid in qids:
                    ok = simcse_available(topic, form, tid)
                    if encoder == SIMCSE:
                        if not ok:
                            continue
                        cands = pool(topic, tid, form, n, "llama3-8b" if form == "explicit" else None)  # ids/sources only
                        scores = simcse_scores(topic, tid, form, n)
                    else:
                        cands = pool(topic, tid, form, n, llm)
                        q = qcache[sha256_text(query(topic, tid))]
                        scores = np.stack([dcache[sha256_text(m["text"])] for m in cands]) @ q
                    rows.append(row_for(topic, tid, form, n, cands, list(scores), rcfg["top_k_saved"], ok))
                write_jsonl(path, rows)
                written += 1
                r5 = np.mean([r["gold_rank"] <= 5 for r in rows])
                log.info("%s %s inter%d: %d questions, recall@5=%.3f", encoder, pool_name, n, len(rows), r5)
    if not topics:
        write_run_manifest({"step": "retrieve", "encoder": encoder, "files_written": written})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", default="all", help=f"'all' (= encoders.yaml + {SIMCSE}) or comma-separated names")
    ap.add_argument("--topics", help="comma-separated topics (smoke runs; files are always rewritten)")
    ap.add_argument("--inter-turns", help="comma-separated, default from retrieval.yaml")
    ap.add_argument("--force", action="store_true", help="rewrite existing files")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if a.topics or a.inter_turns:   # partial runs must never overwrite the committed results
        assert os.environ.get("PHASE3_RESULTS_DIR"), "partial runs: set PHASE3_RESULTS_DIR=results/phase3_smoke"
    rcfg = retrieval_config()
    encs = [SIMCSE] + encoder_names() if a.encoder == "all" else a.encoder.split(",")
    for enc in encs:
        run(enc, rcfg, a.topics.split(",") if a.topics else None,
            [int(x) for x in a.inter_turns.split(",")] if a.inter_turns else None, a.force)


if __name__ == "__main__":
    main()
