"""Generate every figure, table and number used in the mid-term report and slides.

    python report/make_assets.py          (from the repo root, `prefeval` env)

Reads only results/phase1/ (+ phase1/paper_baselines.json) and writes
    report/figures/*.pdf          vector figures (report + slides)
    report/generated/*.tex        LaTeX tables and \\newcommand number macros
so no number in the report is typed by hand.
"""

import collections
import glob
import json
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from phase1.runner import ALL_TOPICS  # noqa: E402

RES = os.path.join(REPO, "results", "phase1")
OUT_FIG = os.path.join(REPO, "report", "figures")
OUT_GEN = os.path.join(REPO, "report", "generated")
os.makedirs(OUT_FIG, exist_ok=True)
os.makedirs(OUT_GEN, exist_ok=True)

PAPER = json.load(open(os.path.join(REPO, "phase1", "paper_baselines.json")))["explicit"]
PAPER_NAME = {"llama3-8b": "Llama3 8B Instruct", "mistral7b": "Mistral 7b"}
MODEL_LABEL = {"llama3-8b": "Llama-3-8B-Instruct", "mistral7b": "Mistral-7B-Instruct-v0.2"}
METHODS = ["zero-shot", "remind", "cot", "selfcritic", "rag"]
METHOD_LABEL = {"zero-shot": "Zero-shot", "remind": "Reminder", "cot": "CoT", "selfcritic": "Self-Critic",
                "rag": "RAG (top-5)"}
TOK = {0: "0.2k", 3: "1k", 8: "3k", 28: "10k", 48: "16k", 68: "23k"}

# Validated categorical palette (dataviz reference palette, slots 1-5, fixed order) + marker per slot
# as secondary encoding (3 slots are < 3:1 contrast on white).
C = {"zero-shot": "#2a78d6", "remind": "#eb6834", "cot": "#1baf7a", "selfcritic": "#eda100", "rag": "#e87ba4"}
MK = {"zero-shot": "o", "remind": "s", "cot": "^", "selfcritic": "D", "rag": "v"}
OURS, PAPERC, INK, MUTED, GRID = "#2a78d6", "#52514e", "#0b0b0b", "#52514e", "#e6e5e0"
SLOT2 = "#eb6834"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 0.8, "lines.linewidth": 2, "lines.markersize": 6,
    "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": "tight",
})


# ============================================================================ data
def load_records():
    rows = []
    for path in glob.glob(os.path.join(RES, "*", "*", "*", "inter*", "*.jsonl")):
        for line in open(path):
            r = json.loads(line)
            if r.get("backend_rev") != 2:
                continue
            m = re.search(r"<choice>\s*([ABCD])", r.get("first_choice") or "")
            rows.append({
                "model": r["model"], "form": r["form"], "method": r["method"], "inter": r["inter_turns"],
                "topic": r["topic"], "task": r["task_id"], "correct": bool(r["correct"]), "choice": r["choice"],
                "gold": r["correct_idx"], "parse_fail": r["choice"] is None,
                "first_correct": (m.group(1) == r["correct_idx"]) if m else False,
                "seconds": r["seconds"],
                "refusal": r["choice"] is None and bool(re.search(r"\b(I cannot|I can't|I apologize|I'm unable|I understand)",
                                                                   r["response_to_q"])),
            })
    return pd.DataFrame(rows)


df = load_records()
pt = pd.read_csv(os.path.join(RES, "summary", "per_topic.csv"))
pt = pt[pt["complete"]]
summ = pd.read_csv(os.path.join(RES, "summary", "summary.csv"))
# keep only complete cells in the record-level frame too (drops the 20 leftover smoke questions)
keep = set(map(tuple, pt[["model", "form", "method", "inter_turns", "topic"]].values))
df = df[[k in keep for k in zip(df.model, df.form, df.method, df.inter, df.topic)]]


def acc(model, form, method, inter):
    r = summ[(summ.model == model) & (summ.form == form) & (summ.method == method) & (summ.inter_turns == inter)]
    return float(r.acc_macro.iloc[0]) if len(r) else None


def paper(model, method, inter):
    return PAPER.get(PAPER_NAME[model], {}).get(method, {}).get(str(inter))


def inters(model, form, method):
    return sorted(summ[(summ.model == model) & (summ.form == form) & (summ.method == method)].inter_turns)


def save(fig, name):
    fig.savefig(os.path.join(OUT_FIG, name + ".pdf"))
    fig.savefig(os.path.join(OUT_FIG, name + ".png"), dpi=200)
    plt.close(fig)


def style_x(ax, xs):
    ax.set_xticks(range(len(xs)), [TOK[x] for x in xs])
    ax.set_ylim(0, 100)
    ax.tick_params(length=0)


# ============================================================================ figures
def fig_vs_paper(name="fig_vs_paper", figsize=(10.5, 4.6)):
    """Small multiples: rows = model, cols = method; ours (solid) vs paper (dashed, hollow)."""
    fig, axes = plt.subplots(2, 5, figsize=figsize, sharey=True)
    for row, model in enumerate(["llama3-8b", "mistral7b"]):
        all_x = sorted({x for m in METHODS for x in inters(model, "explicit", m)})
        pos = {x: i for i, x in enumerate(all_x)}
        for col, m in enumerate(METHODS):
            ax = axes[row][col]
            xs = inters(model, "explicit", m)
            ax.plot([pos[x] for x in xs], [acc(model, "explicit", m, x) for x in xs], color=OURS, marker="o",
                    label="Ours (local, Q8)")
            px = [x for x in all_x if paper(model, m, x) is not None]
            ax.plot([pos[x] for x in px], [paper(model, m, x) for x in px], color=PAPERC, ls="--", marker="o",
                    mfc="white", label="Paper (Fig. 6)")
            style_x(ax, all_x)
            if row == 0:
                ax.set_title(METHOD_LABEL[m])
            if col == 0:
                ax.set_ylabel(f"{MODEL_LABEL[model].split('-Instruct')[0]}\naccuracy (%)")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.04))
    fig.supxlabel("Context between preference and question (paper token rows)", color=MUTED, fontsize=9)
    fig.tight_layout()
    save(fig, name)


def fig_methods(model, form, name, title):
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    all_x = sorted({x for m in METHODS for x in inters(model, form, m)})
    pos = {x: i for i, x in enumerate(all_x)}
    for m in METHODS:
        xs = inters(model, form, m)
        if not xs:
            continue
        ys = [acc(model, form, m, x) for x in xs]
        ax.plot([pos[x] for x in xs], ys, color=C[m], marker=MK[m], label=METHOD_LABEL[m])
        ax.annotate(METHOD_LABEL[m], (pos[xs[-1]], ys[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=7.5, color=INK)
    style_x(ax, all_x)
    ax.set_xlim(-0.3, len(all_x) - 0.3 + 0.9)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Context length (paper token rows)")
    ax.set_title(title, loc="left")
    ax.legend(loc="lower left", ncol=2)
    save(fig, name)


def fig_explicit_vs_implicit():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.1), sharey=True)
    w = 0.38
    for ax, model in zip(axes, ["llama3-8b", "mistral7b"]):
        x = np.arange(len(METHODS))
        e = [acc(model, "explicit", m, 8) for m in METHODS]
        i = [acc(model, "implicit-choice", m, 8) for m in METHODS]
        b1 = ax.bar(x - w / 2 - 0.01, e, w, color=OURS, label="Explicit")
        b2 = ax.bar(x + w / 2 + 0.01, i, w, color=SLOT2, label="Implicit (choice-based)")
        for bars in (b1, b2):
            for b in bars:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{b.get_height():.0f}", ha="center",
                        va="bottom", fontsize=7, color=INK)
        ax.set_xticks(x, [METHOD_LABEL[m].replace(" (top-5)", "") for m in METHODS])
        ax.set_ylim(0, 100)
        ax.tick_params(length=0)
        ax.set_title(MODEL_LABEL[model], loc="left")
    axes[0].set_ylabel("Accuracy at 3k tokens (%)")
    axes[0].legend(loc="upper left", ncol=2)
    fig.tight_layout()
    save(fig, "fig_explicit_vs_implicit")


def selfcritic_stats():
    out = []
    for (model, form, inter), g in df[df.method == "selfcritic"].groupby(["model", "form", "inter"]):
        n = len(g)
        out.append({"model": model, "form": form, "inter": inter, "n": n,
                    "first": 100 * g.first_correct.mean(), "final": 100 * g.correct.mean(),
                    "r2w": int((g.first_correct & ~g.correct).sum()), "w2r": int((~g.first_correct & g.correct).sum()),
                    "empty": 100 * g.parse_fail.mean()})
    return pd.DataFrame(out)


def fig_selfcritic(sc):
    fig, axes = plt.subplots(1, 4, figsize=(10.5, 2.9), sharey=True)
    combos = [("llama3-8b", "explicit"), ("llama3-8b", "implicit-choice"), ("mistral7b", "explicit"),
              ("mistral7b", "implicit-choice")]
    w = 0.38
    for ax, (model, form) in zip(axes, combos):
        g = sc[(sc.model == model) & (sc.form == form)].sort_values("inter")
        x = np.arange(len(g))
        ax.bar(x - w / 2 - 0.01, g["first"], w, color=OURS, label="First answer (before critique)")
        ax.bar(x + w / 2 + 0.01, g["final"], w, color=SLOT2, label="Final answer (after revision)")
        ax.set_xticks(x, [TOK[i] for i in g.inter])
        ax.set_ylim(0, 100)
        ax.tick_params(length=0)
        ax.set_title(f"{MODEL_LABEL[model].split('-Instruct')[0]} | {form.replace('implicit-choice', 'implicit')}",
                     fontsize=9, loc="left")
    axes[0].set_ylabel("Accuracy (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    save(fig, "fig_selfcritic")


def position_stats(model="mistral7b", form="explicit", method="zero-shot"):
    g = df[(df.model == model) & (df.form == form) & (df.method == method)]
    rows = []
    for inter, h in g.groupby("inter"):
        r = {"inter": inter}
        for p in "ABCD":
            r[f"acc_{p}"] = 100 * h[h.gold == p].correct.mean()
            r[f"pick_{p}"] = 100 * (h.choice == p).mean()
        rows.append(r)
    return pd.DataFrame(rows).sort_values("inter")


def fig_position(pos):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.0))
    xs = list(pos.inter)
    pcol = {"A": C["zero-shot"], "B": C["remind"], "C": C["cot"], "D": C["selfcritic"]}
    pmk = {"A": "o", "B": "s", "C": "^", "D": "D"}
    for p in "ABCD":
        axes[0].plot(range(len(xs)), pos[f"acc_{p}"], color=pcol[p], marker=pmk[p], label=f"correct answer = {p}")
        axes[1].plot(range(len(xs)), pos[f"pick_{p}"], color=pcol[p], marker=pmk[p], label=f"option {p}")
    for ax in axes:
        style_x(ax, xs)
        ax.set_xlabel("Context length")
    axes[1].axhline(25, color=MUTED, lw=1, ls=":")
    axes[1].text(len(xs) - 1, 26, "uniform 25%", ha="right", fontsize=7, color=MUTED)
    axes[1].set_ylim(0, 50)
    axes[0].set_ylabel("Accuracy (%)")
    axes[1].set_ylabel("Share of model picks (%)")
    axes[0].set_title("Accuracy by position of the correct option", loc="left", fontsize=9)
    axes[1].set_title("Which option the model picks", loc="left", fontsize=9)
    axes[0].legend(loc="lower left", fontsize=7)
    axes[1].legend(loc="upper left", ncol=2, fontsize=7)
    fig.tight_layout()
    save(fig, "fig_position_bias")


def fig_topic_heatmap():
    """Per-topic zero-shot / reminder / RAG accuracy at 3k, both models (explicit)."""
    cols = [(m, meth) for m in ["llama3-8b", "mistral7b"] for meth in ["zero-shot", "remind", "rag"]]
    mat = np.full((len(ALL_TOPICS), len(cols)), np.nan)
    for j, (m, meth) in enumerate(cols):
        g = pt[(pt.model == m) & (pt.form == "explicit") & (pt.method == meth) & (pt.inter_turns == 8)]
        for _, r in g.iterrows():
            mat[ALL_TOPICS.index(r.topic), j] = 100 * r.acc
    order = np.argsort(np.nanmean(mat, axis=1))
    mat = mat[order]
    topics = [ALL_TOPICS[i] for i in order]
    fig, ax = plt.subplots(figsize=(6.2, 6.4))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blue", ["#f3f7fd", "#9ec5f4", "#2a78d6", "#104281"])
    cmap.set_bad("#f0efec")
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax.text(j, i, "–" if np.isnan(v) else f"{v:.0f}", ha="center", va="center", fontsize=7,
                    color="white" if (not np.isnan(v) and v > 60) else INK)
    ax.set_xticks(range(len(cols)), [f"{'Llama' if m == 'llama3-8b' else 'Mistral'}\n{METHOD_LABEL[k].replace(' (top-5)', '')}"
                                     for m, k in cols], fontsize=7.5)
    ax.set_yticks(range(len(topics)), [t.replace("_", " ") for t in topics], fontsize=7.5)
    ax.tick_params(length=0)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.axvline(2.5, color="white", lw=3)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Accuracy (%)", color=MUTED)
    cb.outline.set_visible(False)
    ax.set_title("Explicit preference, 3k tokens: accuracy per topic (sorted, hardest at top)", fontsize=9, loc="left")
    save(fig, "fig_topic_heatmap")
    return topics


def rag_refusals():
    g = df[(df.method == "rag") & (df.form == "explicit") & (df.inter == 8)]
    out = g.groupby(["model", "topic"]).agg(n=("correct", "size"), refusal=("refusal", "mean"),
                                            parse_fail=("parse_fail", "mean"), acc=("correct", "mean")).reset_index()
    return out


def fig_refusals(rf):
    g = rf[rf.model == "llama3-8b"].sort_values("parse_fail", ascending=True)
    g = g[g.parse_fail > 0]
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    y = np.arange(len(g))
    ax.barh(y, 100 * g.parse_fail, color=OURS, height=0.65)
    for yi, v in zip(y, 100 * g.parse_fail):
        ax.text(v + 1, yi, f"{v:.0f}%", va="center", fontsize=7, color=INK)
    ax.set_yticks(y, [t.replace("_", " ") for t in g.topic], fontsize=7.5)
    ax.set_xlabel("Answers without a valid choice (%)")
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    ax.tick_params(length=0)
    ax.set_xlim(0, 75)
    ax.set_title("Llama-3-8B, explicit RAG at 3k: refusals by topic", loc="left", fontsize=9)
    save(fig, "fig_rag_refusals")


# ============================================================================ tables / macros
def fmt(v, d=1):
    return "--" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"


def tab_vs_paper(model):
    xs = sorted({x for m in METHODS for x in inters(model, "explicit", m)})
    lines = [r"\begin{tabular}{l" + "r" * len(xs) + "}", r"\toprule",
             "Method & " + " & ".join(f"{TOK[x]} ({x})" for x in xs) + r" \\", r"\midrule"]
    for m in METHODS:
        cells = []
        for x in xs:
            o, p = acc(model, "explicit", m, x), paper(model, m, x)
            if o is None:
                cells.append("--")
            elif p is None:
                cells.append(f"{o:.1f} / --")
            else:
                d = o - p
                cells.append(f"{o:.1f} / {p} ({'+' if d >= 0 else '$-$'}{abs(d):.1f})")
        lines.append(METHOD_LABEL[m] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT_GEN, f"tab_vs_paper_{model}.tex"), "w").write("\n".join(lines) + "\n")


def tab_implicit():
    lines = [r"\begin{tabular}{l rrr rrr}", r"\toprule",
             r" & \multicolumn{3}{c}{Llama-3-8B} & \multicolumn{3}{c}{Mistral-7B} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"Method & 0.2k & 1k & 3k & 0.2k & 1k & 3k \\", r"\midrule"]
    for m in METHODS:
        cells = [fmt(acc(mod, "implicit-choice", m, x)) for mod in ["llama3-8b", "mistral7b"] for x in (0, 3, 8)]
        lines.append(METHOD_LABEL[m] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT_GEN, "tab_implicit.tex"), "w").write("\n".join(lines) + "\n")


def tab_selfcritic(sc):
    lines = [r"\begin{tabular}{ll r rrrrr}", r"\toprule",
             r"Model & Form & Context & First & Final & R$\to$W & W$\to$R & Empty (\%) \\", r"\midrule"]
    for _, r in sc.sort_values(["model", "form", "inter"]).iterrows():
        lines.append(f"{MODEL_LABEL[r.model].split('-Instruct')[0]} & {r.form.replace('implicit-choice', 'implicit')} & "
                     f"{TOK[r.inter]} & {r['first']:.1f} & {r['final']:.1f} & {r.r2w} & {r.w2r} & {r['empty']:.1f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT_GEN, "tab_selfcritic.tex"), "w").write("\n".join(lines) + "\n")


def compute_table():
    man = [json.loads(l) for l in open(os.path.join(RES, "run_manifest.jsonl"))]
    groups = {"Llama explicit (Tiers A+B)": ["tierA", "tierB"],
              "Llama implicit-choice": ["llama_implicitA", "llama_implicitB"],
              "Mistral explicit, short (Tiers A+B)": ["mistral_tierA", "mistral_tierB"],
              "Mistral explicit, long (10k--23k)": ["mistral_tierC_long_pilot", "mistral_tierC_long"],
              "Mistral implicit-choice": ["mistral_implicitA", "mistral_implicitB"],
              "Smoke tests / checks": ["smoke", "mistral_smoke", "mic"]}
    dur = collections.defaultdict(float)
    q = collections.defaultdict(int)
    for e in man:
        if e["event"] == "end":
            k = os.path.basename(e["config_file"]).replace(".yaml", "")
            dur[k] += e.get("elapsed_s", 0)
            q[k] += e.get("questions_this_run") or 0
    lines = [r"\begin{tabular}{lrrr}", r"\toprule", r"Runs & Questions & GPU hours & s / question \\", r"\midrule"]
    tq = th = 0
    for label, ks in groups.items():
        qq, hh = sum(q[k] for k in ks), sum(dur[k] for k in ks) / 3600
        tq += qq
        th += hh
        lines.append(f"{label} & {qq:,} & {hh:.1f} & {3600 * hh / max(qq, 1):.1f}" + r" \\")
    lines += [r"\midrule", f"Total & {tq:,} & {th:.1f} & {3600 * th / tq:.1f}" + r" \\", r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT_GEN, "tab_compute.tex"), "w").write("\n".join(lines) + "\n")
    return tq, th


def tab_context():
    """Measured mean prompt length (model tokens) of the final zero-shot call, explicit form."""
    paper_turns = {0: 2, 3: 5, 8: 10, 28: 30, 48: 50, 68: 70}
    lines = [r"\begin{tabular}{rrrrr}", r"\toprule",
             r"Paper row & Paper turns & \texttt{inter\_turns} & Llama-3 tokens & Mistral tokens \\", r"\midrule"]
    for x in (0, 3, 8, 28, 48, 68):
        cells = []
        for model in ["llama3-8b", "mistral7b"]:
            r = summ[(summ.model == model) & (summ.form == "explicit") & (summ.method == "zero-shot") & (summ.inter_turns == x)]
            cells.append(f"{r.avg_prompt_tokens.iloc[0]:,.0f}" if len(r) else "-- (8k limit)")
        lines.append(f"{TOK[x]} & {paper_turns[x]} & {x} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT_GEN, "tab_context.tex"), "w").write("\n".join(lines) + "\n")


def tab_slides():
    """Compact tables for the 3-slide mid-evaluation deck: 'ours (paper)' explicit grid + explicit-vs-implicit at 3k."""
    xs = [0, 3, 8, 28, 48, 68]
    lines = [r"\begin{tabular}{ll" + "c" * len(xs) + "}", r"\toprule",
             r"Model & Method & " + " & ".join(TOK[x] for x in xs) + r" \\", r"\midrule"]
    for model, short in [("llama3-8b", "Llama-3-8B"), ("mistral7b", "Mistral-7B")]:
        for k, m in enumerate(METHODS):
            cells = []
            for x in xs:
                o, p = acc(model, "explicit", m, x), paper(model, m, x)
                cells.append("" if o is None else (f"{o:.1f} ({p})" if p is not None else f"{o:.1f}"))
            lines.append((short if k == 0 else "") + " & " + METHOD_LABEL[m].replace(" (top-5)", "") + " & "
                         + " & ".join(cells) + r" \\")
        if model == "llama3-8b":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT_GEN, "tab_slide_explicit.tex"), "w").write("\n".join(lines) + "\n")

    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r" & \multicolumn{2}{c}{Llama-3-8B} & \multicolumn{2}{c}{Mistral-7B} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}", r"Method & Explicit & Implicit & Explicit & Implicit \\", r"\midrule"]
    for m in METHODS:
        cells = [fmt(acc(mod, f, m, 8)) for mod in ["llama3-8b", "mistral7b"] for f in ["explicit", "implicit-choice"]]
        lines.append(METHOD_LABEL[m].replace(" (top-5)", "") + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT_GEN, "tab_slide_implicit.tex"), "w").write("\n".join(lines) + "\n")


def tab_slide_compare():
    """Side-by-side comparison for the mid-evaluation slide: paper (left) | ours (right), explicit preference."""
    xs = [0, 3, 8, 28, 48, 68]
    n = len(xs)
    col = "c" * n
    lines = [r"\begin{tabular}{ll" + col + "@{\hspace{10pt}}" + col + "}", r"\toprule",
             r" & & \multicolumn{%d}{c}{\textbf{Paper} (Fig.\ 6, Bedrock)} & \multicolumn{%d}{c}{\textbf{Ours} (local, Ollama Q8)} \\" % (n, n),
             r"\cmidrule(lr){3-%d}\cmidrule(lr){%d-%d}" % (2 + n, 3 + n, 2 + 2 * n),
             r"Model & Method & " + " & ".join(TOK[x] for x in xs) + " & " + " & ".join(TOK[x] for x in xs) + r" \\",
             r"\midrule"]
    for model, short in [("llama3-8b", "Llama-3-8B"), ("mistral7b", "Mistral-7B")]:
        for k, m in enumerate(METHODS):
            pc = [str(paper(model, m, x)) if paper(model, m, x) is not None else "--" for x in xs]
            oc = [f"{acc(model, 'explicit', m, x):.1f}" if acc(model, "explicit", m, x) is not None else "--" for x in xs]
            lines.append((short if k == 0 else "") + " & " + METHOD_LABEL[m].replace(" (top-5)", "") + " & "
                         + " & ".join(pc) + " & " + " & ".join(oc) + r" \\")
        if model == "llama3-8b":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT_GEN, "tab_slide_compare.tex"), "w").write("\n".join(lines) + "\n")


def macros(sc, pos, rf, total_q_run, total_h):
    deltas = []
    for model in ["llama3-8b", "mistral7b"]:
        for m in METHODS:
            for x in inters(model, "explicit", m):
                p = paper(model, m, x)
                if p is not None and x > 0:
                    deltas.append(abs(acc(model, "explicit", m, x) - p))
    deltas = np.array(deltas)
    llsc = sc[(sc.model == "llama3-8b") & (sc.form == "explicit")].set_index("inter")
    ms = sc[(sc.model == "mistral7b") & (sc.form == "explicit")].set_index("inter")
    p0, p68 = pos.set_index("inter").loc[8], pos.set_index("inter").loc[68]
    ll_rf = rf[rf.model == "llama3-8b"]
    m = {
        "nQuestions": f"{int(summ.n_questions.sum()):,}",
        "nCells": str(len(summ)),
        "nQuestionsRun": f"{total_q_run:,}",
        "gpuHours": f"{total_h:.0f}",
        "nCompared": str(len(deltas)),
        "nWithinFive": str(int((deltas <= 5).sum())),
        "nWithinTen": str(int((deltas <= 10).sum())),
        "medianAbsDelta": f"{np.median(deltas):.1f}",
        "llamaSCemptyOneK": f"{llsc.loc[3, 'empty']:.1f}",
        "llamaSCrtwZero": str(int(llsc.loc[0, "r2w"])), "llamaSCwtrZero": str(int(llsc.loc[0, "w2r"])),
        "llamaSCfirstZero": f"{llsc.loc[0, 'first']:.1f}",
        "mistralSCrtwZero": str(int(ms.loc[0, "r2w"])), "mistralSCwtrZero": str(int(ms.loc[0, "w2r"])),
        "mistralSCfirstZero": f"{ms.loc[0, 'first']:.1f}",
        "posAccAthree": f"{p0['acc_A']:.0f}", "posAccDthree": f"{p0['acc_D']:.0f}",
        "posAccAlong": f"{p68['acc_A']:.0f}", "posAccDlong": f"{p68['acc_D']:.0f}",
        "posPickDthree": f"{p0['pick_D']:.0f}", "posPickDlong": f"{p68['pick_D']:.0f}",
        "llamaRagRefusalAll": f"{100 * (ll_rf.parse_fail * ll_rf.n).sum() / ll_rf.n.sum():.1f}",
    }
    with open(os.path.join(OUT_GEN, "macros.tex"), "w") as f:
        for k, v in m.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    return m


def main():
    fig_vs_paper()
    with plt.rc_context({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 11, "legend.fontsize": 12,
                         "xtick.labelsize": 10, "ytick.labelsize": 10, "lines.markersize": 7}):
        fig_vs_paper("fig_vs_paper_slide", figsize=(11, 4.4))
    fig_methods("mistral7b", "explicit", "fig_mistral_long", "Mistral-7B, explicit preference: all methods")
    fig_methods("llama3-8b", "explicit", "fig_llama_methods", "Llama-3-8B, explicit preference: all methods")
    fig_explicit_vs_implicit()
    sc = selfcritic_stats()
    fig_selfcritic(sc)
    pos = position_stats()
    fig_position(pos)
    fig_topic_heatmap()
    rf = rag_refusals()
    fig_refusals(rf)
    tab_vs_paper("llama3-8b")
    tab_vs_paper("mistral7b")
    tab_implicit()
    tab_selfcritic(sc)
    tab_context()
    tab_slides()
    tab_slide_compare()
    tq, th = compute_table()
    m = macros(sc, pos, rf, tq, th)
    print(json.dumps(m, indent=1))
    print("figures:", sorted(os.listdir(OUT_FIG)))


if __name__ == "__main__":
    main()
