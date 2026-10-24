"""Multi-preference stress test, level 1 (retrieval over the memory, no LLM calls).

    python -m phase3.stress            # seconds; writes results/phase3/stress/retrieval.{csv,md}

Memory = the classifier's entries at 3k tokens (inter_turns 8) + D preferences from questions of OTHER topics
(phase3.prompts.stress_memory; same sampling and order as the tier S MCQ runs). For each policy:
  all  : every entry goes into the prompt -> the true preference is always there, among D+1 entries (dilution)
  top5 : the 5 entries most similar to the question (EmbeddingGemma-2) -> is the true preference among them?
Reported over all questions and over those whose true preference reached memory (classifier hit).
Level 2 (end-to-end accuracy) = tier S MCQ runs (configs/phase3/tierS.yaml), summarised by phase3.aggregate.
"""

import os

import numpy as np
import pandas as pd

from phase3.common import MAIN_RESULTS_DIR, write_run_manifest
from phase3.conversations import FORMS, pool
from phase3.memory import load_memory, memory_config
from phase3.prompts import entries_from_memory, gold_position, select, stress_memory
from phase3.conversations import mcq

STRESS_DIR = os.path.join(MAIN_RESULTS_DIR, "stress")
N_INTER = 8


def main():
    cfg = memory_config()
    rows = []
    for form in FORMS:
        mem = load_memory(form, N_INTER)
        for d in cfg["stress"]["distractors"]:
            for pol in cfg["stress"]["policies"]:
                hit, pos, size, cond = [], [], [], []
                for (topic, tid), row in mem.items():
                    cands = pool(topic, tid, form, N_INTER, "llama3-8b")
                    entries = stress_memory(entries_from_memory(row, cands), topic, tid, form, d, cfg["seed"])
                    used = select(entries, mcq(topic)[tid]["question"], cfg["memory_topk"], pol, cfg["retriever"])
                    p = gold_position(used)
                    hit.append(p is not None)
                    cond.append(row["gold_in_memory"])
                    pos.append(p if p else np.nan)
                    size.append(len(used))
                hit, cond = np.array(hit), np.array(cond)
                rows.append({"form": form, "distractors": d, "policy": pol, "n": len(hit),
                             "entries_in_prompt": float(np.mean(size)),
                             "gold_in_prompt": float(hit.mean()),
                             "gold_in_prompt_given_in_memory": float(hit[cond].mean()),
                             "mean_gold_position": float(np.nanmean(pos))})
    df = pd.DataFrame(rows)
    os.makedirs(STRESS_DIR, exist_ok=True)
    df.to_csv(os.path.join(STRESS_DIR, "retrieval.csv"), index=False)
    md = ("# Stress test, level 1: is the true preference among the entries put in the prompt?\n\n"
          f"Memory at 3k tokens (inter_turns {N_INTER}) + D preferences from other topics; top5 = EmbeddingGemma-2 "
          "retrieval over the memory.\n\n" + df.to_markdown(index=False, floatfmt=".3f") + "\n")
    with open(os.path.join(STRESS_DIR, "retrieval.md"), "w") as f:
        f.write(md)
    write_run_manifest({"step": "stress_retrieval", "rows": rows})
    print(md)


if __name__ == "__main__":
    main()
