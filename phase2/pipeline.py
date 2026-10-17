"""Run Phase 2 stages in order. Every stage skips work that is already finished, so rerunning resumes.

    python -m phase2.pipeline --stage data       # build_data
    python -m phase2.pipeline --stage pseudo     # LLM pseudo-labels: persona, oasst, lmsys      (Ollama, ~1 h)
    python -m phase2.pipeline --stage splits     # assemble dataset + topic folds
    python -m phase2.pipeline --stage train      # all 4 models x 5 folds                       (GPU, several h)
    python -m phase2.pipeline --stage baseline   # LLM-prompted detector                        (Ollama, ~20 min)
    python -m phase2.pipeline --stage analyze    # evaluate + ensemble + report                 (minutes)
    python -m phase2.pipeline --stage all        # everything above (the manual audit is optional and done separately)

Each step runs as a separate Python process (clean GPU memory between models); the pipeline stops at the first failure.
"""

import argparse
import subprocess
import sys

from phase2.common import MODEL_NAMES

STAGES = {
    "data": [["phase2.build_data"]],
    "pseudo": [["phase2.pseudolabel", "--what", "persona"], ["phase2.pseudolabel", "--what", "oasst"],
               ["phase2.pseudolabel", "--what", "refine"], ["phase2.pseudolabel", "--what", "refine-persona-strict"],
               ["phase2.pseudolabel", "--what", "lmsys"]],
    "splits": [["phase2.splits"]],
    "train": [["phase2.train", "--config", f"configs/phase2/{m}.yaml", "--fold", "all"] for m in MODEL_NAMES],
    "baseline": [["phase2.llm_baseline"]],
    "analyze": [["phase2.evaluate"], ["phase2.ensemble"], ["phase2.report"]],
}
ORDER = ["data", "pseudo", "splits", "train", "baseline", "analyze"]


def run(step):
    cmd = [sys.executable, "-m", *step]
    print(f"\n=== {' '.join(step)}", flush=True)
    r = subprocess.run(cmd)
    if r.returncode:
        sys.exit(f"Stage step failed (exit {r.returncode}): {' '.join(step)} - fix the cause and rerun; finished work is kept.")


def free_gpu_from_ollama():
    """Unload any model Ollama keeps in GPU memory before training (no-op if Ollama is not installed)."""
    try:
        out = subprocess.run(["ollama", "ps"], capture_output=True, text=True).stdout.splitlines()[1:]
        for line in out:
            if line.strip():
                subprocess.run(["ollama", "stop", line.split()[0]])
    except FileNotFoundError:
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=ORDER + ["all"])
    stages = ORDER if (s := ap.parse_args().stage) == "all" else [s]
    for st in stages:
        if st == "train":
            free_gpu_from_ollama()
        for step in STAGES[st]:
            run(step)
    print("\nPipeline finished.")


if __name__ == "__main__":
    main()
