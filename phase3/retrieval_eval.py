"""Metrics, tables and figures for the retrieval benchmark (reads results/phase3/retrieval/, no model calls).

    python -m phase3.retrieval_eval

Writes results/phase3/retrieval/{summary.csv, summary.md} and results/phase3/retrieval/figures/*.png.
Subsets: `common` = questions that also have stored SimCSE scores (949 explicit / 973 implicit; the like-for-like comparison,
same coverage as Phase 1 RAG) and `all` = all 1,000 questions (encoders only).
Metrics per (encoder, pool, inter_turns): recall@k of the gold preference message, MRR, mean gold rank, and the share of
top-5 slots taken by LMSYS distractor messages (what a RAG prompt would be filled with instead of the preference).
"""

import os

import numpy as np
import pandas as pd

from phase3.common import RETRIEVAL_DIR, SIMCSE, encoder_names, read_jsonl, retrieval_config
from phase3.embed import load_meta

TOKENS = {0: "0.2k", 3: "1k", 8: "3k", 28: "10k", 48: "16k", 68: "23k"}     # paper Fig. 6 rows
LABEL = {"bge_base": "bge-base-v1.5", "e5_base": "e5-base-v2", "bge_large": "bge-large-v1.5",
         "mxbai_large": "mxbai-embed-large", "embeddinggemma2": "EmbeddingGemma-2", SIMCSE: "SimCSE (paper)"}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]   # dataviz categorical slots 1-5 (validated)
MARKERS = ["o", "s", "^", "D", "v"]
GRAY = "#6b6a63"
SIMCSE_PARAMS_M = 355.4          # princeton-nlp/sup-simcse-roberta-large


def metrics(rows, ks):
    ranks = np.array([r["gold_rank"] for r in rows])
    out = {"n": len(rows), **{f"recall@{k}": float((ranks <= k).mean()) for k in ks},
           "mrr": float((1 / ranks).mean()), "mean_rank": float(ranks.mean()), "median_rank": float(np.median(ranks)),
           "distractor_share@5": float(np.mean([np.mean([s == "lmsys" for s in r["top_sources"][:5]]) for r in rows]))}
    return out


def collect(rcfg):
    recs = []
    for enc in [SIMCSE] + encoder_names():
        d = os.path.join(RETRIEVAL_DIR, enc)
        if not os.path.isdir(d):
            print(f"skip {enc}: no rankings")
            continue
        meta = load_meta(enc) if enc != SIMCSE else None
        for pool_name in sorted(os.listdir(d)):
            for n in rcfg["inter_turns"]:
                rows = read_jsonl(os.path.join(d, pool_name, f"inter{n}.jsonl"))
                if not rows:
                    continue
                form, _, reply = pool_name.partition("@")
                for subset in ("common", "all"):
                    sub = [r for r in rows if r["simcse_available"]] if subset == "common" else rows
                    if enc == SIMCSE and subset == "all":
                        continue
                    recs.append({"encoder": enc, "form": form, "reply": reply or "-", "pool": pool_name,
                                 "inter_turns": n, "tokens": TOKENS[n], "subset": subset, **metrics(sub, rcfg["recall_ks"]),
                                 "params_m": SIMCSE_PARAMS_M if enc == SIMCSE else meta["params_m"],
                                 "latency_ms": np.nan if enc == SIMCSE else meta["latency"]["median_ms"]})
    return pd.DataFrame(recs)


def headline(df, reply="llama3-8b"):
    """Like-for-like view: common subset; explicit uses `reply`'s pool for encoders and the paper pool for SimCSE."""
    c = df[df.subset == "common"]
    keep = (c.form == "implicit-choice") | (c.reply == reply) | ((c.encoder == SIMCSE) & (c.reply == "paper"))
    return c[keep]


def markdown(df):
    lines = ["# Phase 3 retrieval benchmark", "",
             "Gold = the message that states the preference; query = the final question; candidates = every message before it.",
             "Subset `common` (questions with stored SimCSE scores). Explicit pools use Llama-3-8B's own reply "
             "(SimCSE: the authors' stored reply score).", ""]
    h = headline(df)
    order = [e for e in [SIMCSE] + encoder_names() if e in set(h.encoder)]
    for metric in ("recall@5", "recall@1", "mrr", "distractor_share@5"):
        for form in ("explicit", "implicit-choice"):
            p = h[h.form == form].pivot(index="encoder", columns="inter_turns", values=metric).loc[order]
            p.columns = [TOKENS[c] for c in p.columns]
            p.index = [LABEL[e] for e in p.index]
            lines += [f"## {metric}, {form}", "", p.to_markdown(floatfmt=".3f"), ""]
    # reply-source sensitivity (explicit pools differ only in the LLM reply)
    c = df[(df.subset == "common") & (df.form == "explicit") & (df.encoder != SIMCSE)]
    if c.reply.nunique() > 1:
        p = c.pivot_table(index=["encoder"], columns="reply", values="recall@5", aggfunc="mean")
        p.index = [LABEL[e] for e in p.index]
        lines += ["## Explicit recall@5 by reply source (mean over lengths)", "", p.to_markdown(floatfmt=".3f"), ""]
    cost = df[df.subset == "common"].groupby("encoder")[["params_m", "latency_ms"]].first().loc[order]
    cost.index = [LABEL[e] for e in cost.index]
    lines += ["## Cost", "", "latency = median ms to embed one message (batch 1, RTX 3060, fp32)", "",
              cost.to_markdown(floatfmt=".1f"), ""]
    return "\n".join(lines)


def figures(df, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#9a998f", "axes.labelcolor": "#3d3c37", "xtick.color": "#5c5b54",
                         "ytick.color": "#5c5b54", "axes.grid": True, "grid.color": "#e6e5df", "grid.linewidth": 0.6})
    h = headline(df)
    encs = [e for e in encoder_names() if e in set(h.encoder)]

    def by_length(metric, ylabel, fname, ylim):
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
        for ax, form in zip(axes, ("explicit", "implicit-choice")):
            f = h[h.form == form]
            ns = sorted(f.inter_turns.unique())
            xs = range(len(ns))
            s = f[f.encoder == SIMCSE].set_index("inter_turns").reindex(ns)
            ax.plot(xs, s[metric], color=GRAY, ls="--", lw=2, marker="o", ms=5, label=LABEL[SIMCSE])
            for i, e in enumerate(encs):
                s = f[f.encoder == e].set_index("inter_turns").reindex(ns)
                ax.plot(xs, s[metric], color=COLORS[i], lw=2, marker=MARKERS[i], ms=5, label=LABEL[e],
                        markeredgecolor="white", markeredgewidth=0.8)
            ax.set_xticks(list(xs), [TOKENS[n] for n in ns])
            ax.set_xlabel("conversation length (tokens, paper Fig. 6 rows)")
            ax.set_title(form, fontsize=10, color="#1f1e1b")
            ax.set_ylim(*ylim)
        axes[0].set_ylabel(ylabel)
        axes[1].legend(frameon=False, fontsize=8, loc="best")
        fig.tight_layout()
        fig.savefig(os.path.join(out, fname), dpi=200)
        plt.close(fig)

    by_length("recall@5", "recall@5 of the preference message", "recall5_vs_length.png", (0, 1.02))
    by_length("distractor_share@5", "share of top-5 that are LMSYS messages", "distractor_share_vs_length.png", (0, 1.02))

    # recall@5 at the longest context vs per-message latency (encoders only; SimCSE has no measured latency)
    n_max = h.inter_turns.max()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, form in zip(axes, ("explicit", "implicit-choice")):
        f = h[(h.form == form) & (h.inter_turns == n_max)].set_index("encoder")
        if SIMCSE in f.index:
            ax.axhline(f.loc[SIMCSE, "recall@5"], color=GRAY, ls="--", lw=1.5)
            ax.text(0.02, f.loc[SIMCSE, "recall@5"], "SimCSE (paper)", color="#3d3c37", fontsize=8, va="bottom",
                    transform=ax.get_yaxis_transform())
        for i, e in enumerate(encs):
            ax.scatter(f.loc[e, "latency_ms"], f.loc[e, "recall@5"], color=COLORS[i], marker=MARKERS[i], s=60,
                       edgecolor="white", linewidth=1, zorder=3)
            ax.annotate(LABEL[e], (f.loc[e, "latency_ms"], f.loc[e, "recall@5"]), textcoords="offset points",
                        xytext=(6, 4), fontsize=8, color="#3d3c37")
        ax.set_xlabel("ms to embed one message (batch 1, RTX 3060)")
        ax.set_title(f"{form}, {TOKENS[n_max]} tokens", fontsize=10, color="#1f1e1b")
        ax.set_xlim(0, max(h.latency_ms.max() * 1.3, 1))
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("recall@5")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "recall5_vs_latency.png"), dpi=200)
    plt.close(fig)


def main():
    rcfg = retrieval_config()
    df = collect(rcfg)
    df.to_csv(os.path.join(RETRIEVAL_DIR, "summary.csv"), index=False)
    md = markdown(df)
    with open(os.path.join(RETRIEVAL_DIR, "summary.md"), "w") as f:
        f.write(md)
    figures(df, os.path.join(RETRIEVAL_DIR, "figures"))
    print(md)


if __name__ == "__main__":
    main()
