"""Phase 3 vs Phase 1 baselines (classification accuracy), stress-test level 2, figures. Works on partial results.

    python -m phase3.aggregate

Accuracy = macro average over topics of the strict upstream metric (`extract_choice`), as in Phase 1 / the paper.
Two coverages, both computed from the per-question records (never from Phase 1's summary file):
  all     : every question the method has (Phase 1 RAG: 949 explicit / 973 implicit; everything else: 1,000)
  rag_set : only questions that have stored SimCSE scores -> Phase 1 RAG vs rag_gemma vs memory_rag on identical questions
Writes results/phase3/summary/{summary.csv, comparison.md, stress.md} and figures/.
"""

import glob
import json
import os

import numpy as np
import pandas as pd

from phase1.backend import BACKEND_REV
from phase3.common import PHASE1_RESULTS, RESULTS_DIR
from phase3.conversations import simcse_available

TOKENS = {0: "0.2k", 3: "1k", 8: "3k", 28: "10k", 48: "16k", 68: "23k"}
P1_METHODS = ["zero-shot", "remind", "cot", "selfcritic", "rag"]
P3_METHODS = ["memory", "rag_gemma", "memory_rag"]
LABEL = {"zero-shot": "Zero-shot", "remind": "Reminder", "cot": "CoT", "selfcritic": "Self-Critic",
         "rag": "RAG (SimCSE, paper)", "memory": "Memory (ours)", "rag_gemma": "RAG (EmbeddingGemma-2, ours)",
         "memory_rag": "Memory + RAG (ours)"}
COLORS = {"memory": "#2a78d6", "rag_gemma": "#eb6834", "memory_rag": "#1baf7a"}     # dataviz slots 1-3 (validated)
BASE_STYLE = {"zero-shot": ("#9a998f", "o"), "remind": ("#6b6a63", "s"), "rag": ("#3d3c37", "^")}
OUT = os.path.join(RESULTS_DIR, "summary")


def load_records(pattern, phase):
    rows = []
    for path in glob.glob(pattern):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("backend_rev") != BACKEND_REV:
                    continue
                rows.append({"phase": phase, "model": r["model"], "form": r["form"], "method": r["method"],
                             "variant": r.get("variant", r["method"]), "inter_turns": r["inter_turns"],
                             "topic": r["topic"], "task_id": r["task_id"], "correct": bool(r["correct"]),
                             "parsed": r["choice"] is not None, "lenient": r["choice_lenient"] == r["correct_idx"],
                             "stress_d": r.get("stress_d", 0), "policy": r.get("policy", "-"),
                             "gold_in_memory": r.get("gold_in_memory"), "gold_position": r.get("gold_position"),
                             "gold_in_rag": r.get("gold_in_rag"), "seconds": r["seconds"],
                             "truncated": bool(r.get("truncated"))})
    return pd.DataFrame(rows)


def macro(g):
    t = g.groupby("topic").agg(acc=("correct", "mean"), len_=("lenient", "mean"))
    return pd.Series({"acc": 100 * t.acc.mean(), "acc_lenient": 100 * t.len_.mean(), "topics": len(t), "n": len(g),
                      "parse_fail_pct": 100 * (1 - g.parsed.mean()), "truncated": int(g.truncated.sum()),
                      "sec_per_q": g.seconds.mean()})


def summarise(df):
    out = []
    keys = ["model", "form", "method", "variant", "inter_turns"]
    for cov in ("all", "rag_set"):
        sub = df if cov == "all" else df[df.rag_ok]
        s = sub.groupby(keys).apply(macro, include_groups=False).reset_index()
        s["coverage"] = cov
        out.append(s)
    return pd.concat(out, ignore_index=True)


def comparison_md(s):
    lines = ["# Phase 3 vs Phase 1 baselines (classification accuracy %, macro over topics)", "",
             "`all` = each method's full coverage (Phase 1 RAG: questions with stored SimCSE scores); "
             "RAG rows at the bottom of each table are on identical questions (`rag_set`).", ""]
    main = s[(s.variant == s.method)]
    for (model, form), g in main.groupby(["model", "form"]):
        if not (g.method.isin(P3_METHODS)).any():
            continue
        ns = sorted(g.inter_turns.unique())
        rows = []
        for m in P1_METHODS + P3_METHODS:
            r = g[(g.method == m) & (g.coverage == "all")].set_index("inter_turns")
            if len(r):
                rows.append([LABEL[m]] + [f"{r.acc[n]:.1f}" if n in r.index else "–" for n in ns])
        rows.append(["**same questions (rag_set)**"] + [""] * len(ns))
        for m in ("rag", "rag_gemma", "memory_rag"):
            r = g[(g.method == m) & (g.coverage == "rag_set")].set_index("inter_turns")
            if len(r):
                rows.append([LABEL[m]] + [f"{r.acc[n]:.1f}" if n in r.index else "–" for n in ns])
        head = "| Method | " + " | ".join(TOKENS[n] for n in ns) + " |"
        lines += [f"## {model} | {form}", "", head, "|---" * (len(ns) + 1) + "|",
                  *["| " + " | ".join(r) + " |" for r in rows], ""]
        d = g[(g.coverage == "all") & g.method.isin(P3_METHODS)][
            ["method", "inter_turns", "n", "topics", "parse_fail_pct", "acc_lenient", "truncated", "sec_per_q"]]
        lines += ["Phase 3 diagnostics:", "", d.to_markdown(index=False, floatfmt=".2f"), ""]
    return "\n".join(lines)


def stress_md(s, df):
    st = s[(s.coverage == "all") & s.variant.str.startswith("memory_stress")].copy()
    if st.empty:
        return "# Stress test (level 2)\n\nNo tier S results yet.\n"
    st["distractors"] = st.variant.str.extract(r"_d(\d+)_").astype(int)
    st["policy"] = st.variant.str.extract(r"_(all|top5)$")
    base = s[(s.coverage == "all") & (s.variant == "memory") & (s.model.isin(st.model)) &
             (s.inter_turns.isin(st.inter_turns))][["model", "form", "inter_turns", "acc"]]
    lines = ["# Stress test, level 2: accuracy with D extra preferences in memory (memory-only prompt)", ""]
    for (model, form, n), g in st.groupby(["model", "form", "inter_turns"]):
        p = g.pivot(index="distractors", columns="policy", values="acc")
        b = base[(base.model == model) & (base.form == form) & (base.inter_turns == n)]
        if len(b):
            p.loc[0] = b.acc.iloc[0]
        p = p.sort_index()
        lines += [f"## {model} | {form} | {TOKENS[n]} tokens (D = 0: tier A memory run)", "",
                  p.to_markdown(floatfmt=".1f"), ""]
    return "\n".join(lines)


def figures(s):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#9a998f", "axes.labelcolor": "#3d3c37", "xtick.color": "#5c5b54",
                         "ytick.color": "#5c5b54", "axes.grid": True, "grid.color": "#e6e5df", "grid.linewidth": 0.6})
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    main = s[(s.variant == s.method) & (s.coverage == "all")]
    for (model, form), g in main.groupby(["model", "form"]):
        if not g.method.isin(P3_METHODS).any():
            continue
        ns = sorted(g.inter_turns.unique())
        x = {n: i for i, n in enumerate(ns)}
        fig, ax = plt.subplots(figsize=(6, 3.6))
        for m, (c, mk) in BASE_STYLE.items():
            r = g[g.method == m].sort_values("inter_turns")
            if len(r):
                ax.plot([x[n] for n in r.inter_turns], r.acc, color=c, ls="--", lw=1.5, marker=mk, ms=5, label=LABEL[m])
        for m, c in COLORS.items():
            r = g[g.method == m].sort_values("inter_turns")
            if len(r):
                ax.plot([x[n] for n in r.inter_turns], r.acc, color=c, lw=2.2, marker="o", ms=6, label=LABEL[m],
                        markeredgecolor="white", markeredgewidth=0.8)
        ax.set_xticks(range(len(ns)), [TOKENS[n] for n in ns])
        ax.set_ylim(0, 100)
        ax.set_xlabel("conversation length (tokens, paper Fig. 6 rows)")
        ax.set_ylabel("accuracy % (macro over topics)")
        ax.set_title(f"{model} | {form}", fontsize=10, color="#1f1e1b")
        ax.legend(frameon=False, fontsize=7.5, loc="lower left")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "figures", f"acc_vs_context_{model}_{form}.png"), dpi=200)
        plt.close(fig)


def main():
    p1 = load_records(os.path.join(PHASE1_RESULTS, "*", "*", "*", "inter*", "*.jsonl"), "phase1")
    p3 = load_records(os.path.join(RESULTS_DIR, "mcq", "*", "*", "*", "inter*", "*.jsonl"), "phase3")
    if p3.empty:
        print("no Phase 3 MCQ results yet")
        return
    # baselines only where Phase 3 has the same (model, form) so tables stay comparable
    p1 = p1[p1.form.isin(p3.form.unique()) & p1.model.isin(p3.model.unique())]
    df = pd.concat([p1, p3], ignore_index=True)
    df["rag_ok"] = [simcse_available(t, f, i) for t, f, i in zip(df.topic, df.form, df.task_id)]
    s = summarise(df)
    os.makedirs(OUT, exist_ok=True)
    s.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    md = comparison_md(s)
    with open(os.path.join(OUT, "comparison.md"), "w") as f:
        f.write(md)
    smd = stress_md(s, df)
    with open(os.path.join(OUT, "stress.md"), "w") as f:
        f.write(smd)
    figures(s)
    print(md)
    print(smd)


if __name__ == "__main__":
    main()
