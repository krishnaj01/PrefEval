"""Generate every table, figure and number used in the Phase 3 report (report/phase3_report.tex).

    python report/make_assets_phase3.py      (repo root, `prefeval-p2` env)

Reads only results/phase3/ (and Phase 1/2 results for baselines) and writes report/generated/p3_*.tex and
report/figures/p3_*.pdf. Run `python -m phase3.aggregate` first so results/phase3/summary/ is current.
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
RES = os.path.join(REPO, "results", "phase3")
OUT = os.path.join(REPO, "report", "generated")
FIG = os.path.join(REPO, "report", "figures")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

TOK = {0: "0.2k", 3: "1k", 8: "3k", 28: "10k", 48: "16k", 68: "23k"}
ENC = ["simcse_paper", "bge_base", "e5_base", "bge_large", "mxbai_large", "embeddinggemma2"]
ENC_LABEL = {"simcse_paper": "SimCSE (paper)", "bge_base": "bge-base-v1.5", "e5_base": "e5-base-v2",
             "bge_large": "bge-large-v1.5", "mxbai_large": "mxbai-embed-large", "embeddinggemma2": "EmbeddingGemma-2"}
M_LABEL = {"zero-shot": "Zero-shot", "remind": "Reminder", "cot": "CoT", "selfcritic": "Self-Critic",
           "rag": "RAG, SimCSE (paper)", "memory": "\\textbf{Memory (ours)}", "rag_gemma": "RAG, EmbeddingGemma-2 (ours)",
           "memory_rag": "Memory + RAG (ours)"}
MODEL_LABEL = {"llama3-8b": "Llama-3-8B", "mistral7b": "Mistral-7B-v0.2"}
COLORS = {"memory": "#2a78d6", "rag_gemma": "#eb6834", "memory_rag": "#1baf7a"}   # dataviz categorical slots 1-3
ENC_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]                # slots 1-5 (validated)
GRAY = "#6b6a63"


def write(name, lines):
    with open(os.path.join(OUT, name), "w") as f:
        f.write("\n".join(lines) + "\n")


def tab(cols, header_lines, rows, name):
    write(name, [r"\begin{tabular}{" + cols + "}", r"\toprule", *[h + r" \\" for h in header_lines], r"\midrule",
                 *[r if r.startswith(r"\midrule") or r.startswith(r"\addlinespace") else r + r" \\" for r in rows],
                 r"\bottomrule", r"\end{tabular}"])


def records(pattern):
    rows = []
    for p in glob.glob(pattern):
        with open(p) as f:
            rows += [json.loads(l) for l in f if l.strip()]
    return rows


def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#9a998f", "axes.labelcolor": "#3d3c37", "xtick.color": "#5c5b54",
                         "ytick.color": "#5c5b54", "axes.grid": True, "grid.color": "#e6e5df", "grid.linewidth": 0.6,
                         "pdf.fonttype": 42})
    return plt


plt = style()
macros = {}

# =============================================================================== 1. retrieval benchmark
ret = pd.read_csv(os.path.join(RES, "retrieval", "summary.csv"))
ret = ret[ret.subset == "common"]
head = ret[(ret.form == "implicit-choice") | (ret.reply == "llama3-8b") | (ret.reply == "paper")]
rows = []
for e in ENC:
    g = head[head.encoder == e]
    ex = g[g.form == "explicit"].set_index("inter_turns")
    im = g[g.form == "implicit-choice"].set_index("inter_turns")
    lat = "--" if e == "simcse_paper" else f"{g.latency_ms.iloc[0]:.1f}"
    vals = [f"{ex.loc[n, 'recall@5']:.2f}" for n in (3, 8, 28, 68)] + [f"{ex.loc[68, 'recall@1']:.2f}"] + \
           [f"{im.loc[n, 'recall@5']:.2f}" for n in (3, 8, 28, 68)]
    name = ENC_LABEL[e] if e != "embeddinggemma2" else r"\textbf{EmbeddingGemma-2}"
    rows.append(f"{name} & {g.params_m.iloc[0]:.0f}M & " + " & ".join(vals) + f" & {lat}")
    if e == "simcse_paper":
        rows.append(r"\midrule")
tab("lr|ccccc|cccc|r",
    [r" & & \multicolumn{5}{c|}{Explicit} & \multicolumn{4}{c|}{Implicit-choice} & ",
     r"Retriever & Params & R@5 1k & 3k & 10k & 23k & R@1 23k & R@5 1k & 3k & 10k & 23k & ms/msg"],
    rows, "p3_retrieval.tex")
g = head.set_index(["encoder", "form", "inter_turns"])
macros.update({
    "qSimExR": f"{100 * g.loc[('simcse_paper', 'explicit', 68), 'recall@5']:.1f}",
    "qGemExR": f"{100 * g.loc[('embeddinggemma2', 'explicit', 68), 'recall@5']:.1f}",
    "qSimImR": f"{100 * g.loc[('simcse_paper', 'implicit-choice', 68), 'recall@5']:.1f}",
    "qGemImR": f"{100 * g.loc[('embeddinggemma2', 'implicit-choice', 68), 'recall@5']:.1f}",
    "qSimExROne": f"{100 * g.loc[('simcse_paper', 'explicit', 3), 'recall@1']:.1f}",
    "qGemExROne": f"{100 * g.loc[('embeddinggemma2', 'explicit', 3), 'recall@1']:.1f}",
    "qGemLat": f"{g.loc[('embeddinggemma2', 'explicit', 3), 'latency_ms']:.1f}",
    "qMxbaiExR": f"{100 * g.loc[('mxbai_large', 'explicit', 68), 'recall@5']:.1f}",
    "qEFiveExR": f"{100 * g.loc[('e5_base', 'explicit', 68), 'recall@5']:.1f}",
})
reply_gap = ret[(ret.form == "explicit") & (ret.encoder != "simcse_paper")].pivot_table(
    index=["encoder", "inter_turns"], columns="reply", values="recall@5")
macros["qReplyGap"] = f"{100 * (reply_gap['llama3-8b'] - reply_gap['mistral7b']).abs().max():.1f}"

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
for ax, form in zip(axes, ("explicit", "implicit-choice")):
    for i, e in enumerate(ENC):
        s = head[(head.encoder == e) & (head.form == form)].sort_values("inter_turns")
        kw = dict(color=GRAY, ls="--", lw=1.8, marker="o", ms=4) if e == "simcse_paper" else \
            dict(color=ENC_COLORS[i - 1], lw=1.8, marker="os^Dv"[i - 1], ms=4, markeredgecolor="white", markeredgewidth=0.6)
        ax.plot(range(len(s)), s["recall@5"], label=ENC_LABEL[e], **kw)
    ax.set_xticks(range(5), ["1k", "3k", "10k", "16k", "23k"])
    ax.set_title(form, fontsize=9)
    ax.set_xlabel("conversation length (tokens)")
    ax.set_ylim(0, 1.02)
axes[0].set_ylabel("recall@5 of the preference")
axes[1].legend(frameon=False, fontsize=6.8, loc="lower left")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "p3_retrieval.pdf"))
plt.close(fig)

# =============================================================================== 2. memory quality
mem = pd.read_csv(os.path.join(RES, "memory", "summary.csv"))
rows = []
for form in ("explicit", "implicit-choice"):
    m = mem[mem.form == form].set_index("inter_turns")
    rows.append(f"{form} & " + " & ".join(f"{100 * m.loc[n, 'gold_in_memory']:.1f}" for n in TOK) + " & " +
                " & ".join(f"{m.loc[n, 'mean_false_entries']:.2f}" for n in (28, 48, 68)) + " & " + f"{m.loc[68, 'mean_size']:.2f}")
tab("l|cccccc|ccc|c", [r" & \multicolumn{6}{c|}{True preference in memory (\%)} & \multicolumn{3}{c|}{False entries / question} & Mean size",
                       r"Form & 0.2k & 1k & 3k & 10k & 16k & 23k & 10k & 16k & 23k & at 23k"], rows, "p3_memory.tex")
me = mem.set_index(["form", "inter_turns"])
macros.update({"qMemEx": f"{100 * me.loc[('explicit', 8), 'gold_in_memory']:.1f}",
               "qMemIm": f"{100 * me.loc[('implicit-choice', 8), 'gold_in_memory']:.1f}",
               "qFalseSixteen": f"{me.loc[('explicit', 48), 'mean_false_entries']:.2f}",
               "qFalseTwentyThree": f"{me.loc[('explicit', 68), 'mean_false_entries']:.2f}"})
macros["qMissedEx"] = str(int(round(1000 * (1 - me.loc[("explicit", 8), "gold_in_memory"]))))

# =============================================================================== 3. main comparison
s = pd.read_csv(os.path.join(RES, "summary", "summary.csv"))
main = s[(s.variant == s.method) & (s.coverage == "all")]
rset = s[(s.variant == s.method) & (s.coverage == "rag_set")]


def acc(model, form, method, n, frame=main):
    r = frame[(frame.model == model) & (frame.form == form) & (frame.method == method) & (frame.inter_turns == n)]
    return float(r.acc.iloc[0]) if len(r) else None


def fmt(v, best=False):
    if v is None:
        return "--"
    return f"\\textbf{{{v:.1f}}}" if best else f"{v:.1f}"


METHODS = ["zero-shot", "remind", "cot", "selfcritic", "rag", "memory", "rag_gemma", "memory_rag"]
for model, cells in (("llama3-8b", [("explicit", n) for n in (0, 3, 8)] + [("implicit-choice", n) for n in (0, 3, 8)]),
                     ("mistral7b", [("explicit", n) for n in (0, 3, 8, 28, 48, 68)])):
    best = {c: max(v for v in (acc(model, c[0], m, c[1]) for m in METHODS) if v is not None) for c in cells}
    rows = []
    for m in METHODS:
        if m == "memory":
            rows.append(r"\midrule")
        vals = [acc(model, f, m, n) for f, n in cells]
        if all(v is None for v in vals):
            continue
        rows.append(M_LABEL[m] + " & " + " & ".join(fmt(v, v is not None and abs(v - best[c]) < 1e-9)
                                                     for v, c in zip(vals, cells)))
    if model == "llama3-8b":
        hdr = [r" & \multicolumn{3}{c|}{Explicit} & \multicolumn{3}{c}{Implicit-choice}",
               r"Method & 0.2k & 1k & 3k & 0.2k & 1k & 3k"]
        tab("l|ccc|ccc", hdr, rows, "p3_main_llama.tex")
    else:
        hdr = [r" & \multicolumn{6}{c}{Explicit}", r"Method & 0.2k & 1k & 3k & 10k & 16k & 23k"]
        tab("l|cccccc", hdr, rows, "p3_main_mistral_ex.tex")
cells = [("implicit-choice", n) for n in (0, 3, 8)]
best = {c: max(v for v in (acc("mistral7b", c[0], m, c[1]) for m in METHODS) if v is not None) for c in cells}
rows = []
for m in METHODS:
    if m == "memory":
        rows.append(r"\midrule")
    vals = [acc("mistral7b", f, m, n) for f, n in cells]
    rows.append(M_LABEL[m] + " & " + " & ".join(fmt(v, v is not None and abs(v - best[c]) < 1e-9) for v, c in zip(vals, cells)))
tab("l|ccc", [r" & \multicolumn{3}{c}{Implicit-choice}", r"Method & 0.2k & 1k & 3k"], rows, "p3_main_mistral_im.tex")

# like-for-like RAG (identical questions)
rows = []
for model, form, ns in (("llama3-8b", "explicit", (3, 8)), ("llama3-8b", "implicit-choice", (3, 8)),
                        ("mistral7b", "explicit", (3, 8)), ("mistral7b", "implicit-choice", (3, 8))):
    for n in ns:
        a, b, c = (acc(model, form, m, n, rset) for m in ("rag", "rag_gemma", "memory_rag"))
        rows.append(f"{MODEL_LABEL[model]} & {form} & {TOK[n]} & {fmt(a)} & {fmt(b)} & {b - a:+.1f} & {fmt(c)}")
tab("lll|cc|c|c", ["Model & Form & Length & SimCSE RAG & EmbeddingGemma-2 RAG & $\\Delta$ & Memory + RAG"], rows,
    "p3_rag_same.tex")


def best_baseline(model, form, n):
    vals = {m: acc(model, form, m, n) for m in ("zero-shot", "remind", "cot", "selfcritic", "rag")}
    vals = {k: v for k, v in vals.items() if v is not None}
    k = max(vals, key=vals.get)
    return k, vals[k]


bl = best_baseline("mistral7b", "explicit", 68)
macros.update({
    "qLlExMemThree": fmt(acc("llama3-8b", "explicit", "memory", 8)),
    "qLlExRagThree": fmt(acc("llama3-8b", "explicit", "rag", 8)),
    "qLlImMemThree": fmt(acc("llama3-8b", "implicit-choice", "memory", 8)),
    "qLlImRagThree": fmt(acc("llama3-8b", "implicit-choice", "rag", 8)),
    "qLlExGemThree": fmt(acc("llama3-8b", "explicit", "rag_gemma", 8)),
    "qMiMemTwentyThree": fmt(acc("mistral7b", "explicit", "memory", 68)),
    "qMiRagTwentyThree": fmt(acc("mistral7b", "explicit", "rag", 68)),
    "qMiRemTwentyThree": fmt(acc("mistral7b", "explicit", "remind", 68)),
    "qMiZsTwentyThree": fmt(acc("mistral7b", "explicit", "zero-shot", 68)),
    "qMiMemTen": fmt(acc("mistral7b", "explicit", "memory", 28)),
    "qMiGainTwentyThree": f"{acc('mistral7b', 'explicit', 'memory', 68) - bl[1]:.1f}",
    "qMiImMemZero": fmt(acc("mistral7b", "implicit-choice", "memory", 0)),
    "qMiImRemZero": fmt(acc("mistral7b", "implicit-choice", "remind", 0)),
    "qMiImMemThree": fmt(acc("mistral7b", "implicit-choice", "memory", 8)),
    "qMiImRagThree": fmt(acc("mistral7b", "implicit-choice", "rag", 8)),
})

# figure: accuracy vs length, 3 panels
fig, axes = plt.subplots(1, 4, figsize=(9.6, 2.9), sharey=True,
                         gridspec_kw={"width_ratios": [3, 3, 6, 3]})
panels = [("llama3-8b", "explicit", (0, 3, 8)), ("llama3-8b", "implicit-choice", (0, 3, 8)),
          ("mistral7b", "explicit", (0, 3, 8, 28, 48, 68)), ("mistral7b", "implicit-choice", (0, 3, 8))]
base = {"zero-shot": ("#b5b4ab", "o"), "remind": ("#7d7c74", "s"), "rag": ("#3d3c37", "^")}
for ax, (model, form, ns) in zip(axes, panels):
    for m, (c, mk) in base.items():
        xs = [i for i, n in enumerate(ns) if acc(model, form, m, n) is not None]
        ax.plot(xs, [acc(model, form, m, ns[i]) for i in xs], color=c, ls="--", lw=1.3, marker=mk, ms=3.5,
                label={"zero-shot": "Zero-shot", "remind": "Reminder", "rag": "RAG (SimCSE)"}[m])
    for m, c in COLORS.items():
        xs = [i for i, n in enumerate(ns) if acc(model, form, m, n) is not None]
        ax.plot(xs, [acc(model, form, m, ns[i]) for i in xs], color=c, lw=2, marker="o", ms=4,
                markeredgecolor="white", markeredgewidth=0.6,
                label={"memory": "Memory (ours)", "rag_gemma": "RAG, EmbeddingGemma-2", "memory_rag": "Memory + RAG"}[m])
    ax.set_xticks(range(len(ns)), [TOK[n] for n in ns])
    ax.set_title(f"{MODEL_LABEL[model]} | {form}", fontsize=8.5)
    ax.set_ylim(30, 100)
axes[0].set_ylabel("accuracy % (macro over topics)")
axes[2].set_xlabel("conversation length (tokens)")
h, l = axes[2].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=6, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(os.path.join(FIG, "p3_main.pdf"))
plt.close(fig)

# =============================================================================== 4. why memory+RAG < memory (per-question)
def by_q(model, form, var, n):
    return {(r["topic"], r["task_id"]): r for r in
            records(os.path.join(RES, "mcq", model, form, var, f"inter{n}", "*.jsonl"))}


rows = []
for model, form, n in (("llama3-8b", "explicit", 8), ("llama3-8b", "implicit-choice", 8), ("mistral7b", "explicit", 8),
                       ("mistral7b", "explicit", 68), ("mistral7b", "implicit-choice", 8)):
    m, mr = by_q(model, form, "memory", n), by_q(model, form, "memory_rag", n)
    lost = [k for k in m if m[k]["correct"] and not mr[k]["correct"]]
    won = sum(not m[k]["correct"] and mr[k]["correct"] for k in m)
    pf = sum(mr[k]["choice"] is None for k in lost)
    gold_rag = sum(bool(mr[k]["gold_in_rag"]) for k in lost)
    same = sum(m[k]["choice"] == mr[k]["choice"] for k in m)
    rows.append(f"{MODEL_LABEL[model]} & {form} & {TOK[n]} & {len(lost)} & {pf} & {len(lost) - pf} & {gold_rag} & {won} & "
                f"{100 * same / len(m):.1f}")
    if (model, form, n) == ("llama3-8b", "implicit-choice", 8):
        macros.update({"qLostIm": str(len(lost)), "qLostImParse": str(pf), "qLostImGold": str(gold_rag)})
    if (model, form, n) == ("llama3-8b", "explicit", 8):
        macros.update({"qLostEx": str(len(lost)), "qLostExParse": str(pf)})
tab("lll|cccc|c|c", ["Model & Form & Length & Lost & of which format & wrong option & pref.\\ retrieved & Gained & Same answer \\%"],
    rows, "p3_lost.tex")

# Mistral 23k: accuracy given the preference reached memory
rs = records(os.path.join(RES, "mcq", "mistral7b", "explicit", "memory", "inter*", "*.jsonl"))
d = pd.DataFrame([{"n": r["inter_turns"], "g": r["gold_in_memory"], "c": r["correct"]} for r in rs])
cond = d.groupby(["n", "g"]).c.mean().unstack()
macros.update({"qMiCondTen": f"{100 * cond.loc[28, True]:.1f}", "qMiCondTwentyThree": f"{100 * cond.loc[68, True]:.1f}",
               "qMiMissAccLo": f"{100 * cond.loc[[28, 48, 68], False].min():.0f}", "qMiMissAccHi": f"{100 * cond.loc[[28, 48, 68], False].max():.0f}"})

# =============================================================================== 5. stress test
st1 = pd.read_csv(os.path.join(RES, "stress", "retrieval.csv"))
st = s[(s.coverage == "all") & s.variant.str.startswith("memory_stress")].copy()
st["D"] = st.variant.str.extract(r"_d(\d+)_").astype(int)
st["policy"] = st.variant.str.extract(r"_(all|top5)$")
rows = []
for form in ("explicit", "implicit-choice"):
    b = acc("llama3-8b", form, "memory", 8)
    rows.append(f"{form} & 0 & 1 & {b:.1f} & 100.0 & {b:.1f}")
    for D in (5, 10, 20):
        a = st[(st.form == form) & (st.D == D) & (st.policy == "all")].acc.iloc[0]
        t = st[(st.form == form) & (st.D == D) & (st.policy == "top5")].acc.iloc[0]
        r1 = st1[(st1.form == form) & (st1.distractors == D) & (st1.policy == "top5")].iloc[0]
        fa = f"\\textbf{{{a:.1f}}}" if a > t else f"{a:.1f}"
        ft = f"\\textbf{{{t:.1f}}}" if t >= a else f"{t:.1f}"
        rows.append(f" & {D} & {D + 1} & {fa} & {100 * r1.gold_in_prompt_given_in_memory:.1f} & {ft}")
    if form == "explicit":
        rows.append(r"\midrule")
tab("lc|cc|cc", [r" & & \multicolumn{2}{c|}{Include all} & \multicolumn{2}{c}{Top-5 retrieval over memory}",
                 r"Form & $D$ & entries & accuracy & pref.\ kept (\%) & accuracy"], rows, "p3_stress.tex")
sx = st.set_index(["form", "D", "policy"]).acc
macros.update({"qStExAll": f"{sx[('explicit', 20, 'all')]:.1f}", "qStExTop": f"{sx[('explicit', 20, 'top5')]:.1f}",
               "qStImAll": f"{sx[('implicit-choice', 20, 'all')]:.1f}", "qStImTop": f"{sx[('implicit-choice', 20, 'top5')]:.1f}",
               "qStKeepEx": f"{100 * st1[(st1.form == 'explicit') & (st1.distractors == 20) & (st1.policy == 'top5')].gold_in_prompt_given_in_memory.iloc[0]:.1f}",
               "qStKeepIm": f"{100 * st1[(st1.form == 'implicit-choice') & (st1.distractors == 20) & (st1.policy == 'top5')].gold_in_prompt_given_in_memory.iloc[0]:.1f}"})

fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6), sharey=False)
for ax, form in zip(axes, ("explicit", "implicit-choice")):
    b = acc("llama3-8b", form, "memory", 8)
    for pol, c, mk, lab in (("all", "#eb6834", "s", "include all entries"), ("top5", "#2a78d6", "o", "top-5 over memory")):
        ys = [b] + [sx[(form, D, pol)] for D in (5, 10, 20)]
        ax.plot([0, 5, 10, 20], ys, color=c, lw=2, marker=mk, ms=4.5, markeredgecolor="white", markeredgewidth=0.6, label=lab)
    ax.axhline(acc("llama3-8b", form, "rag", 8), color=GRAY, ls="--", lw=1.2)
    ax.text(20, acc("llama3-8b", form, "rag", 8) + 0.6, "RAG (SimCSE)", ha="right", fontsize=7, color="#3d3c37")
    ax.set_xticks([0, 5, 10, 20])
    ax.set_xlabel("extra preferences in memory ($D$)")
    ax.set_title(f"Llama-3-8B | {form} | 3k", fontsize=8.5)
axes[0].set_ylabel("accuracy %")
axes[0].legend(frameon=False, fontsize=7, loc="lower left")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "p3_stress.pdf"))
plt.close(fig)

# =============================================================================== 6. cost / prompt size
p1 = records(os.path.join(REPO, "results", "phase1", "llama3-8b", "explicit", "zero-shot", "inter8", "*.jsonl"))
p3 = records(os.path.join(RES, "mcq", "llama3-8b", "explicit", "memory", "inter8", "*.jsonl"))
tok1 = np.mean([r["calls"][-1]["prompt_tokens"] for r in p1])
tok3 = np.mean([r["calls"][-1]["prompt_tokens"] for r in p3])
p2 = pd.read_csv(os.path.join(REPO, "results", "phase2", "eval", "summary.csv")).set_index("model")
macros.update({"qTokZs": f"{tok1:,.0f}", "qTokMem": f"{tok3:,.0f}", "qTokAdd": f"{tok3 - tok1:,.0f}",
               "qClfLat": f"{p2.loc['distilbert', 'gpu_latency_ms']:.1f}"})

n_q = sum(len(records(os.path.join(RES, "mcq", m, "*", "*", "inter*", "*.jsonl"))) for m in ("llama3-8b", "mistral7b"))
macros["qTotalQ"] = f"{n_q:,}"
man = [json.loads(l) for l in open(os.path.join(RES, "run_manifest.jsonl"))]
gpu_s = sum(m.get("elapsed_s", 0) for m in man if m.get("step") == "mcq_run" and m.get("event") == "end")
macros["qGpuHours"] = f"{gpu_s / 3600:.0f}"

write("p3_macros.tex", [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()])
print(json.dumps(macros, indent=1))
