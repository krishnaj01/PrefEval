"""Build the preference memory for every question and context length with the Phase 2 classifier (no LLM calls).

    python -m phase3.memory            # ~1-2 min on GPU; writes results/phase3/memory/

How a memory is built (per question, per length N), in conversation order:
  1. Every USER message before the question is scored by DistilBERT (Phase 2). For a question of topic t the fold model
     whose TEST fold contains t is used, with that fold's validation-tuned threshold, so the classifier never saw t.
  2. A message with P(preference) >= threshold becomes a memory entry. If the message right before it is an assistant
     message, it is attached (implicit-choice: options + the user's selection; explicit: the preference is the first message).
  3. Memory at length N = entries among the first messages (head + first 2N LMSYS messages), so memories grow with N.

Outputs (committed, small):
  results/phase3/memory/probs.jsonl                  fold, sha256(text), P(preference)  (cache)
  results/phase3/memory/<form>/inter<N>.jsonl        per question: entries [{ids, turn, gold}], gold_in_memory
  results/phase3/memory/summary.csv / .md            coverage of the true preference, false entries, memory size
"""

import argparse
import json
import logging
import os

import numpy as np
import pandas as pd

from phase3.common import (MAIN_RESULTS_DIR, REPO, load_yaml, read_jsonl, retrieval_config, sha256_text, write_jsonl,
                           write_run_manifest)
from phase3.conversations import FORMS, pool, question_ids

MEMORY_DIR = os.path.join(MAIN_RESULTS_DIR, "memory")     # deterministic full build; never a smoke variant
PROBS = os.path.join(MEMORY_DIR, "probs.jsonl")
log = logging.getLogger("phase3.memory")


def memory_config():
    return load_yaml(os.path.join(REPO, "configs", "phase3", "memory.yaml"))


def topic_fold():
    with open(os.path.join(REPO, "data", "phase2", "splits.json")) as f:
        folds = json.load(f)["topic_folds"]
    return {t: int(k) for k, ts in folds.items() for t in ts}


def classifier_dir(name, fold):
    return os.path.join(REPO, "models", "phase2", name, f"fold{fold}", "best")


def threshold(name, fold):
    with open(os.path.join(classifier_dir(name, fold), "phase2_threshold.json")) as f:
        return json.load(f)["threshold"]


# ----------------------------------------------------------------------------- classification
def user_texts_by_fold(max_inter, reply_llm):
    """fold -> set of user-message texts that questions of that fold's topics contain."""
    tf = topic_fold()
    out = {}
    for topic, tid in question_ids():
        for form in FORMS:
            for m in pool(topic, tid, form, max_inter, reply_llm):
                if m["role"] == "user":
                    out.setdefault(tf[topic], set()).add(m["text"])
    return out


def classify(cfg, max_inter, reply_llm):
    """Fill the probability cache for every (fold, user text) not yet scored; returns {(fold, sha): prob}."""
    cache = {(r["fold"], r["sha"]): r["prob"] for r in read_jsonl(PROBS)}
    need = {k: sorted(t for t in texts if (k, sha256_text(t)) not in cache)
            for k, texts in user_texts_by_fold(max_inter, reply_llm).items()}
    if not any(need.values()):
        return cache
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for k, texts in sorted(need.items()):
        if not texts:
            continue
        d = classifier_dir(cfg["classifier"], k)
        tok = AutoTokenizer.from_pretrained(d)
        model = AutoModelForSequenceClassification.from_pretrained(d).to(device).eval()
        rows = []
        with torch.no_grad():
            for i in range(0, len(texts), 64):
                batch = texts[i:i + 64]
                enc = tok(batch, truncation=True, max_length=cfg["classifier_max_len"], padding=True,
                          return_tensors="pt").to(device)
                p = torch.softmax(model(**enc).logits.float(), dim=-1)[:, 1].cpu().numpy()
                rows += [{"fold": k, "sha": sha256_text(t), "prob": round(float(x), 6)} for t, x in zip(batch, p)]
        for r in rows:
            cache[(r["fold"], r["sha"])] = r["prob"]
        log.info("fold %d: scored %d user messages", k, len(rows))
        del model
    write_jsonl(PROBS, [{"fold": f, "sha": s, "prob": p} for (f, s), p in sorted(cache.items())])
    return cache


# ----------------------------------------------------------------------------- memory
def build_memory(cands, probs, fold, tau, attach):
    """Memory entries for one candidate pool (conversation order)."""
    entries, user_turn = [], 0
    for i, m in enumerate(cands):
        if m["role"] != "user":
            continue
        user_turn += 1
        if probs[(fold, sha256_text(m["text"]))] < tau:
            continue
        ids = [m["id"]]
        if attach and i > 0 and cands[i - 1]["role"] == "assistant":
            ids = [cands[i - 1]["id"], m["id"]]
        entries.append({"ids": ids, "turn": user_turn, "gold": m["gold"], "source": m["source"]})
    return entries


def run(cfg, rcfg):
    tf = topic_fold()
    reply_llm = rcfg["reply_llms"][0]   # the reply is an assistant message: never classified, never attached
    probs = classify(cfg, max(rcfg["inter_turns"]), reply_llm)
    taus = {k: threshold(cfg["classifier"], k) for k in sorted(set(tf.values()))}
    stats = []
    for form in FORMS:
        for n in [0] + rcfg["inter_turns"]:
            rows = []
            for topic, tid in question_ids():
                k = tf[topic]
                cands = pool(topic, tid, form, n, reply_llm)
                ent = build_memory(cands, probs, k, taus[k], cfg["attach_previous_assistant"])
                rows.append({"topic": topic, "task_id": tid, "fold": k, "entries": ent,
                             "gold_in_memory": any(e["gold"] for e in ent)})
            write_jsonl(os.path.join(MEMORY_DIR, form, f"inter{n}.jsonl"), rows)
            sizes = np.array([len(r["entries"]) for r in rows])
            false = np.array([sum(not e["gold"] for e in r["entries"]) for r in rows])
            false_lmsys = np.array([sum(e["source"] == "lmsys" for e in r["entries"]) for r in rows])
            stats.append({"form": form, "inter_turns": n, "n": len(rows),
                          "gold_in_memory": float(np.mean([r["gold_in_memory"] for r in rows])),
                          "mean_size": float(sizes.mean()), "max_size": int(sizes.max()),
                          "share_size_gt_topk": float((sizes > cfg["memory_topk"]).mean()),
                          "mean_false_entries": float(false.mean()), "mean_false_lmsys": float(false_lmsys.mean()),
                          "share_empty": float((sizes == 0).mean())})
            log.info("%s inter%d: true preference in memory %.3f, mean size %.2f", form, n,
                     stats[-1]["gold_in_memory"], stats[-1]["mean_size"])
    df = pd.DataFrame(stats)
    df.to_csv(os.path.join(MEMORY_DIR, "summary.csv"), index=False)
    with open(os.path.join(MEMORY_DIR, "summary.md"), "w") as f:
        f.write("# Preference memory built by the Phase 2 classifier (DistilBERT, fold-held-out)\n\n"
                + df.to_markdown(index=False, floatfmt=".3f") + "\n")
    write_run_manifest({"step": "memory", "classifier": cfg["classifier"], "thresholds": taus,
                        "summary": stats})
    print(df.to_markdown(index=False, floatfmt=".3f"))


def load_memory(form, n):
    """{(topic, task_id): row} for one form / length (written by `run`)."""
    return {(r["topic"], r["task_id"]): r for r in read_jsonl(os.path.join(MEMORY_DIR, form, f"inter{n}.jsonl"))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(memory_config(), retrieval_config())


if __name__ == "__main__":
    main()
