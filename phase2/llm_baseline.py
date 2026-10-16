"""Step 7: LLM-prompted preference detector (the baseline the trained classifiers are compared against).

    python -m phase2.llm_baseline

Zero-shot Llama-3-8B-Instruct (Ollama, temperature 0) answers YES/NO with the same definition used for pseudo-labelling,
on a fixed random sample of labelled PrefEval rows (configs/phase2/data.yaml: llm_baseline.n_samples, seed).
OASST rows are excluded on purpose: their labels were produced by this same prompt, so scoring it on them would be circular.
Persona-turn labels come from a *different* prompt (with the reference preference), so they are a fair target.

Output: results/phase2/llm_baseline/predictions.jsonl (resumable cache) + metrics.json (accuracy, P/R/F1, latency).
"""

import json
import os
import random

import numpy as np
from tqdm import tqdm

from phase2.common import DATASET, RESULTS_DIR, append_jsonl, data_config, read_jsonl, write_run_manifest
from phase2.metrics import binary_metrics
from phase2.pseudolabel import YESNO_TEMPLATE, ask, make_backend, parse_yesno

OUT = os.path.join(RESULTS_DIR, "llm_baseline")


def sample_rows():
    cfg = data_config()
    rows = [r for r in read_jsonl(DATASET) if r["topic"] is not None]       # PrefEval rows only
    rows.sort(key=lambda r: r["uid"])
    return random.Random(cfg["seed"]).sample(rows, min(cfg["llm_baseline"]["n_samples"], len(rows)))


def main():
    os.makedirs(OUT, exist_ok=True)
    cache = os.path.join(OUT, "predictions.jsonl")
    rows = sample_rows()
    done = {r["uid"] for r in read_jsonl(cache)}
    todo = [r for r in rows if r["uid"] not in done]
    backend, info = make_backend()
    print(f"LLM baseline: {len(todo)} to query ({len(done)} cached)")
    for r in tqdm(todo, desc="llm-baseline"):
        raw, sec = ask(backend, YESNO_TEMPLATE.format(text=r["text"]), 3)
        lab = parse_yesno(raw)
        append_jsonl(cache, {"uid": r["uid"], "fold": r["fold"], "source": r["source"], "form": r["form"],
                             "topic": r["topic"], "label": r["label"], "raw": raw, "pred": lab,
                             "prob": float(lab) if lab is not None else 0.0, "seconds": sec})
    got = read_jsonl(cache)
    y = [g["label"] for g in got]
    p = [g["prob"] for g in got]
    secs = [g["seconds"] for g in got]
    m = {"model": "llm_baseline", "llm": info, "n": len(got), "unparseable": sum(g["pred"] is None for g in got),
         "metrics": binary_metrics(y, p, 0.5),
         "latency": {"device": "cuda (Ollama)", "median_ms": float(np.median(secs) * 1000), "mean_ms": float(np.mean(secs) * 1000)}}
    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(m, f, indent=1)
    mm = m["metrics"]
    print(f"LLM baseline: F1={mm['f1']:.3f} P={mm['precision']:.3f} R={mm['recall']:.3f} acc={mm['accuracy']:.3f} "
          f"median latency {m['latency']['median_ms']:.0f} ms/turn")
    write_run_manifest({"event": "llm_baseline", "n": len(got), "f1": mm["f1"], "llm": info})


if __name__ == "__main__":
    main()
