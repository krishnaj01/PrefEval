"""Aggregate Phase 1 results and compare them with the paper (Figure 6).

    python -m phase1.aggregate                    # only topics whose cell is complete
    python -m phase1.aggregate --include-partial  # also partially-finished cells (e.g. smoke test)

Writes to results/phase1/summary/:
    per_topic.csv     one row per (model, form, method, inter_turns, topic)
    summary.csv       one row per (model, form, method, inter_turns), averaged over topics
    vs_paper.md       our numbers next to the paper's, with deltas and topic coverage
    acc_vs_context_<model>_<form>.png   small multiples: accuracy vs context length, ours vs paper
"""

import argparse
import glob
import json
import os

import pandas as pd

from phase1.backend import BACKEND_REV
from phase1.runner import ALL_TOPICS, METHODS, REPO, RESULTS, load_yaml, MODELS_YAML, n_questions, rag_unavailable_reason, rag_available_count, _n_dataset

SUMMARY_DIR = os.path.join(RESULTS, "summary")
PAPER = json.load(open(os.path.join(REPO, "phase1", "paper_baselines.json")))
TOKENS_LABEL = {0: "0.2k", 3: "1k", 8: "3k", 28: "10k", 48: "16k", 68: "23k"}  # paper's x-axis rows
METHOD_LABEL = {"zero-shot": "Zero-shot", "remind": "Reminder", "cot": "CoT", "selfcritic": "Self-Critic",
                "rag": "RAG (top-5)"}


def load_records():
    rows = []
    for path in glob.glob(os.path.join(RESULTS, "*", "*", "*", "inter*", "*.jsonl")):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("backend_rev") != BACKEND_REV:
                    continue
                rows.append({
                    "model": r["model"], "form": r["form"], "method": r["method"], "inter_turns": r["inter_turns"],
                    "topic": r["topic"], "task_id": r["task_id"], "correct": bool(r["correct"]),
                    "parse_fail": r["choice"] is None,
                    "correct_lenient": r["choice_lenient"] == r["correct_idx"],
                    "truncated": bool(r["truncated"]), "seconds": r["seconds"],
                    "prompt_tokens": max((c["prompt_tokens"] for c in r["calls"] if c["purpose"] != "pref_response"),
                                         default=0),
                })
    return pd.DataFrame(rows)


def per_topic_table(df):
    keys = ["model", "form", "method", "inter_turns", "topic"]
    g = df.groupby(keys).agg(n=("correct", "size"), acc=("correct", "mean"), parse_fail=("parse_fail", "mean"),
                             acc_lenient=("correct_lenient", "mean"), truncated=("truncated", "sum"),
                             sec_per_q=("seconds", "mean"), prompt_tokens=("prompt_tokens", "mean")).reset_index()
    g["n_expected"] = [n_questions(t, form=f, method=m) for t, f, m in zip(g["topic"], g["form"], g["method"])]
    g["complete"] = g["n"] >= g["n_expected"]
    return g


def summary_table(pt):
    keys = ["model", "form", "method", "inter_turns"]
    # Paper averages accuracy over topics -> macro average (mean of per-topic accuracies).
    s = pt.groupby(keys).apply(lambda x: pd.Series({
        "topics": len(x),
        "topics_list": ",".join(sorted(x["topic"])),
        "n_questions": int(x["n"].sum()),
        "acc_macro": 100 * x["acc"].mean(),
        "acc_micro": 100 * (x["acc"] * x["n"]).sum() / x["n"].sum(),
        "acc_lenient_macro": 100 * x["acc_lenient"].mean(),
        "parse_fail_pct": 100 * (x["parse_fail"] * x["n"]).sum() / x["n"].sum(),
        "truncated": int(x["truncated"].sum()),
        "avg_prompt_tokens": (x["prompt_tokens"] * x["n"]).sum() / x["n"].sum(),
        "sec_per_q": (x["sec_per_q"] * x["n"]).sum() / x["n"].sum(),
    }), include_groups=False).reset_index()
    s["context"] = s["inter_turns"].map(lambda i: TOKENS_LABEL.get(i, f"{i} turns"))
    return s


def paper_value(model, form, method, inter):
    paper_name = load_yaml(MODELS_YAML).get(model, {}).get("paper_name")
    form_key = "explicit" if form == "explicit" else None
    try:
        return PAPER[form_key][paper_name][method][str(inter)]
    except (KeyError, TypeError):
        return None


def write_vs_paper(s, path):
    lines = ["# Phase 1: our local reproduction vs. paper (Figure 6, classification task)", "",
             "Accuracy in %, macro-averaged over topics (as in the paper). Paper numbers are averaged over "
             "**all 20 topics**, so with fewer topics the comparison is only indicative.", "",
             f"Backend revision: {BACKEND_REV}. Paper: Zhao et al., ICLR 2025, Fig. 6.", ""]
    for (model, form), grp in s.groupby(["model", "form"]):
        lines += [f"## {model} | {form}", "",
                  "| Method | Context (inter_turns) | Ours | Paper | Δ (ours − paper) | Topics | Questions | Parse-fail % | Truncated |",
                  "|---|---|---|---|---|---|---|---|---|"]
        grp = grp.assign(m_order=grp["method"].map(METHODS.index)).sort_values(["m_order", "inter_turns"])
        for _, r in grp.iterrows():
            p = paper_value(model, form, r["method"], r["inter_turns"])
            ptxt, dtxt = ("–", "–") if p is None else (f"{p:.0f}", f"{r['acc_macro'] - p:+.1f}")
            missing = [t for t in ALL_TOPICS if r["method"] == "rag" and rag_unavailable_reason(t, form)]
            usable = 20 - len(missing)
            cov = f"{r['topics']}/{usable}" + ("" if r["topics"] >= usable else " ⚠ partial") + ("†" if missing else "")
            lines.append(f"| {METHOD_LABEL[r['method']]} | {r['context']} ({r['inter_turns']}) | {r['acc_macro']:.1f} | "
                         f"{ptxt} | {dtxt} | {cov} | {r['n_questions']} | {r['parse_fail_pct']:.1f} | {r['truncated']} |")
        short = [(t, rag_available_count(t, form), _n_dataset(t)) for t in ALL_TOPICS
                 if not rag_unavailable_reason(t, form) and rag_available_count(t, form) < _n_dataset(t)]
        if short and "rag" in set(grp["method"]):
            lost = sum(n - k for _, k, n in short)
            lines.append(f"‡ RAG covers {sum(_n_dataset(t) for t in ALL_TOPICS) - lost} of {sum(_n_dataset(t) for t in ALL_TOPICS)} questions: "
                         "the upstream retrieval files end early for " + ", ".join(f"{t} ({k}/{n})" for t, k, n in short) + ".")
        broken = [t for t in ALL_TOPICS if rag_unavailable_reason(t, form)]
        if broken and "rag" in set(grp["method"]):
            lines.append(f"† RAG excludes {', '.join(broken)}: the upstream precomputed SimCSE retrieval file is "
                         "truncated (only 4/51 questions parse), so RAG is averaged over the remaining topics.")
        lines.append("")
    if s.empty:
        lines.append("_No complete cells yet. Use --include-partial to see in-progress numbers._")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def plot(s, model, form, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OURS, PAPER_C, INK, MUTED, GRID = "#2a78d6", "#52514e", "#0b0b0b", "#52514e", "#e6e5e0"
    grp = s[(s["model"] == model) & (s["form"] == form)]
    methods = [m for m in METHODS if m in set(grp["method"])]
    if not methods:
        return
    xs = sorted(set(grp["inter_turns"]) | {int(k) for m in methods for k in
                                           (PAPER.get("explicit", {}).get(load_yaml(MODELS_YAML)[model]["paper_name"], {})
                                            .get(m, {}) if form == "explicit" else {})})
    pos = {x: i for i, x in enumerate(xs)}
    fig, axes = plt.subplots(1, len(methods), figsize=(2.9 * len(methods), 3.2), sharey=True, squeeze=False)
    for ax, m in zip(axes[0], methods):
        ours = grp[grp["method"] == m].sort_values("inter_turns")
        ax.plot([pos[x] for x in ours["inter_turns"]], ours["acc_macro"], color=OURS, lw=2, marker="o", ms=6,
                label="Ours (local Q8)")
        if form == "explicit":
            pv = PAPER["explicit"].get(load_yaml(MODELS_YAML)[model]["paper_name"], {}).get(m, {})
            px = sorted(int(k) for k in pv)
            ax.plot([pos[x] for x in px], [pv[str(x)] for x in px], color=PAPER_C, lw=2, ls="--", marker="o", ms=6,
                    mfc="white", label="Paper (Fig. 6)")
        ax.set_title(METHOD_LABEL[m], fontsize=10, color=INK)
        ax.set_xticks(range(len(xs)), [f"{TOKENS_LABEL.get(x, x)}\n({x})" for x in xs], fontsize=8, color=MUTED)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.tick_params(colors=MUTED, labelsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
    axes[0][0].set_ylabel("Accuracy (%)", color=MUTED, fontsize=9)
    fig.supxlabel("Context length: paper token row (inter_turns)", fontsize=9, color=MUTED)
    topics = int(grp["topics"].max())
    fig.suptitle(f"{model} | {form} | classification accuracy vs context ({topics} topic(s))", fontsize=11, color=INK)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--include-partial", action="store_true", help="include unfinished topic cells")
    opts = ap.parse_args()
    df = load_records()
    if df.empty:
        print("No results yet.")
        return
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    pt = per_topic_table(df)
    pt.to_csv(os.path.join(SUMMARY_DIR, "per_topic.csv"), index=False, float_format="%.4f")
    used = pt if opts.include_partial else pt[pt["complete"]]
    if used.empty:
        print("No complete topic cells yet; rerun with --include-partial.")
        return
    s = summary_table(used)
    s.to_csv(os.path.join(SUMMARY_DIR, "summary.csv"), index=False, float_format="%.2f")
    write_vs_paper(s, os.path.join(SUMMARY_DIR, "vs_paper.md"))
    for (model, form) in s[["model", "form"]].drop_duplicates().itertuples(index=False):
        plot(s, model, form, os.path.join(SUMMARY_DIR, f"acc_vs_context_{model}_{form}.png"))
    print(open(os.path.join(SUMMARY_DIR, "vs_paper.md")).read())
    print(f"Wrote {SUMMARY_DIR}/ (per_topic.csv, summary.csv, vs_paper.md, plots)"
          + ("  [includes partial cells]" if opts.include_partial else ""))


if __name__ == "__main__":
    main()
