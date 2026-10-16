"""Step 10: all Phase 2 figures and tables (from saved results only).

    python -m phase2.report

Figures (results/phase2/figures/*.png + .pdf):
  data_sources        class counts per data source            data_lengths     message-length distribution by label
  loss_curves         train + validation loss vs epoch, mean +- range over the 5 folds, one panel per model
  val_f1_curves       validation F1 vs epoch
  model_comparison    accuracy / precision / recall / F1 with fold std error bars
  roc_pr              ROC and precision-recall curves (pooled test folds) + LLM-baseline operating point
  confusion           confusion matrix per model (pooled test folds, counts + row %)
  calibration         reliability diagram
  per_source          recall on positive sources / false-positive rate on negative sources (heatmap)
  per_topic           F1 per topic (heatmap; topic-bias check)
  ensembles           best subsets per rule + latency-vs-F1 Pareto view
  diversity           pairwise error correlation between models
Tables (results/phase2/tables/*.md): data_card, model_comparison, llm_comparison, ensembles_top, per_fold
"""

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from phase2.common import DATA_DIR, DATASET, MODEL_NAMES, RESULTS_DIR, read_jsonl
from phase2.ensemble import ENS_DIR
from phase2.evaluate import EVAL_DIR

FIG = os.path.join(RESULTS_DIR, "figures")
TAB = os.path.join(RESULTS_DIR, "tables")
# validated categorical palette (slots 1-4) + a marker per model as secondary encoding
COLOR = {"distilbert": "#2a78d6", "modernbert": "#eb6834", "tinyllama": "#1baf7a", "qwen3": "#eda100", "llm": "#52514e"}
MARK = {"distilbert": "o", "modernbert": "s", "tinyllama": "^", "qwen3": "D"}
LABEL = {"distilbert": "DistilBERT", "modernbert": "ModernBERT", "tinyllama": "TinyLlama-1.1B (LoRA)", "qwen3": "Qwen3-1.7B (LoRA)"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "axes.grid.axis": "y", "grid.color": GRID, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "legend.frameon": False, "savefig.bbox": "tight",
                     "lines.linewidth": 2, "pdf.fonttype": 42})


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=180)
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    plt.close(fig)


def md_table(df, path, floatfmt=".3f"):
    with open(path, "w") as f:
        f.write(df.to_markdown(index=False, floatfmt=floatfmt) if hasattr(df, "to_markdown") else df.to_string(index=False))
        f.write("\n")


def models_present():
    return [m for m in MODEL_NAMES if glob.glob(os.path.join(RESULTS_DIR, m, "fold*", "metrics.json"))]


# ----------------------------------------------------------------------------- data card
def data_card():
    rows = read_jsonl(DATASET)
    if not rows:
        return
    df = pd.DataFrame(rows)
    ct = df.groupby(["source", "label"]).size().unstack(fill_value=0).rename(columns={0: "label 0", 1: "label 1"})
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    y = np.arange(len(ct))
    ax.barh(y - 0.2, ct.get("label 0", 0), 0.38, color=COLOR["distilbert"], label="label 0 (no preference)")
    ax.barh(y + 0.2, ct.get("label 1", 0), 0.38, color=COLOR["modernbert"], label="label 1 (preference)")
    ax.set_yticks(y, ct.index)
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("rows")
    ax.legend(loc="lower right")
    ax.set_title(f"Training data by source ({len(df):,} rows)", loc="left", fontsize=10)
    save(fig, "data_sources")
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    lens = df["text"].str.len()
    bins = np.linspace(0, min(2000, lens.max()), 40)
    for lab, c in ((0, COLOR["distilbert"]), (1, COLOR["modernbert"])):
        ax.hist(lens[df["label"] == lab], bins=bins, alpha=0.65, color=c, label=f"label {lab}")
    ax.set_xlabel("message length (characters)")
    ax.set_ylabel("rows")
    ax.legend()
    save(fig, "data_lengths")
    card = json.load(open(os.path.join(DATA_DIR, "data_card.json"))) if os.path.exists(os.path.join(DATA_DIR, "data_card.json")) else {}
    tab = ct.reset_index()
    tab["total"] = tab.get("label 0", 0) + tab.get("label 1", 0)
    md_table(tab, os.path.join(TAB, "data_card.md"), ".0f")
    if card:
        with open(os.path.join(TAB, "data_card.md"), "a") as f:
            f.write(f"\nTotal {card['rows']} rows: {card['positives']} positive / {card['negatives']} negative. "
                    f"Label origin: {card['by_label_origin']}. Excluded: {card['excluded']}. "
                    f"Per-topic positive rate range: {card['topic_positive_rate_min_max']}.\n")


# ----------------------------------------------------------------------------- training curves
def curves(models):
    n_folds = max(len(glob.glob(os.path.join(RESULTS_DIR, m, "fold*", "log_history.csv"))) for m in models)
    fig, axes = plt.subplots(1, len(models), figsize=(3.3 * len(models), 3.0), sharey=True, squeeze=False)
    fig2, axes2 = plt.subplots(1, len(models), figsize=(3.3 * len(models), 3.0), sharey=True, squeeze=False)
    for ax, ax2, m in zip(axes[0], axes2[0], models):
        tr, va, vf = [], [], []
        for p in sorted(glob.glob(os.path.join(RESULTS_DIR, m, "fold*", "log_history.csv"))):
            h = pd.read_csv(p)
            if "loss" in h:
                tr.append(h[h["loss"].notna()][["epoch", "loss"]].to_numpy())
            if "eval_loss" in h:
                e = h[h["eval_loss"].notna()]
                va.append(e[["epoch", "eval_loss"]].to_numpy())
                vf.append(e[["epoch", "eval_f1"]].to_numpy())
        for series, a, col, lab in ((tr, ax, COLOR[m], "train loss"), (va, ax, MUTED, "validation loss"), (vf, ax2, COLOR[m], "validation F1")):
            if not series:
                continue
            grid = np.linspace(max(s[0, 0] for s in series), min(s[-1, 0] for s in series), 50)
            ys = np.array([np.interp(grid, s[:, 0], s[:, 1]) for s in series])
            a.plot(grid, ys.mean(0), color=col, label=lab, ls="-" if lab != "validation loss" else "--")
            a.fill_between(grid, ys.min(0), ys.max(0), color=col, alpha=0.15, lw=0)
        for a, yl in ((ax, "loss"), (ax2, "F1")):
            a.set_title(LABEL[m], fontsize=9, loc="left")
            a.set_xlabel("epoch")
        ax.legend(fontsize=7)
    axes[0][0].set_ylabel("cross-entropy loss")
    axes2[0][0].set_ylabel("validation F1")
    fig.suptitle(f"Training and validation loss (mean over {n_folds} fold(s), band = min-max)", fontsize=10, x=0.01, ha="left")
    fig2.suptitle(f"Validation F1 during training (mean over {n_folds} fold(s), band = min-max)", fontsize=10, x=0.01, ha="left")
    save(fig, "loss_curves")
    save(fig2, "val_f1_curves")


# ----------------------------------------------------------------------------- comparison / curves / confusion
def comparison(models):
    s = pd.read_csv(os.path.join(EVAL_DIR, "summary.csv")).set_index("model").loc[models]
    pf = pd.read_csv(os.path.join(EVAL_DIR, "per_fold.csv"))
    metrics = ["accuracy", "precision", "recall", "f1"]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    w = 0.8 / len(models)
    for i, m in enumerate(models):
        vals = [s.loc[m, k] for k in metrics]
        errs = [pf[pf.model == m][k].std(ddof=0) for k in metrics]
        x = np.arange(len(metrics)) + (i - (len(models) - 1) / 2) * w
        ax.bar(x, vals, w * 0.92, color=COLOR[m], label=LABEL[m], yerr=errs, capsize=2, error_kw={"lw": 0.8, "ecolor": INK})
    ax.set_xticks(range(len(metrics)), [k.capitalize() if k != "f1" else "F1" for k in metrics])
    lo = max(0, min(s[metrics].min()) - 0.1)
    ax.set_ylim(lo, 1.0)
    ax.legend(ncol=2, fontsize=7, loc="lower right")
    ax.set_title("Pooled test metrics (error bars = std over the 5 topic folds)", loc="left", fontsize=10)
    save(fig, "model_comparison")
    cols = ["accuracy", "precision", "recall", "f1", "f1_fold_mean", "f1_fold_std", "roc_auc", "pr_auc", "ece",
            "ood_flag_rate", "gpu_latency_ms", "params_total", "params_trainable", "train_time_min", "peak_gpu_mem_gib"]
    md_table(s.reset_index()[["model"] + [c for c in cols if c in s.columns]], os.path.join(TAB, "model_comparison.md"))
    md_table(pf, os.path.join(TAB, "per_fold.md"))


def roc_pr_conf_calib(models):
    c = json.load(open(os.path.join(EVAL_DIR, "curves.json")))
    llm = None
    p = os.path.join(RESULTS_DIR, "llm_baseline", "metrics.json")
    if os.path.exists(p):
        llm = json.load(open(p))["metrics"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.4))
    for m in models:
        a1.plot(c[m]["roc"]["fpr"], c[m]["roc"]["tpr"], color=COLOR[m], label=LABEL[m], lw=1.6)
        a2.plot(c[m]["pr"]["recall"], c[m]["pr"]["precision"], color=COLOR[m], label=LABEL[m], lw=1.6)
    if llm:
        a1.scatter([llm["fpr"]], [llm["recall"]], color=COLOR["llm"], marker="X", s=60, zorder=5, label="LLM zero-shot")
        a2.scatter([llm["recall"]], [llm["precision"]], color=COLOR["llm"], marker="X", s=60, zorder=5, label="LLM zero-shot")
    a1.plot([0, 1], [0, 1], ls=":", color=MUTED, lw=1)
    a1.set(xlabel="false-positive rate", ylabel="true-positive rate (recall)", title="ROC")
    a2.set(xlabel="recall", ylabel="precision", title="Precision-recall")
    a2.legend(fontsize=7, loc="lower left")
    save(fig, "roc_pr")

    fig, axes = plt.subplots(1, len(models), figsize=(2.6 * len(models), 2.6), squeeze=False)
    for i_ax, (ax, m) in enumerate(zip(axes[0], models)):
        cm = c[m]["confusion"]
        mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        pct = mat / mat.sum(1, keepdims=True)
        ax.imshow(pct, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#f3f7fd", "#2a78d6"]), vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{mat[i, j]:,}\n{pct[i, j]:.1%}", ha="center", va="center", fontsize=8,
                        color="white" if pct[i, j] > 0.6 else INK)
        ax.set_xticks([0, 1], ["pred 0", "pred 1"])
        ax.set_yticks([0, 1], ["true 0", "true 1"] if i_ax == 0 else ["", ""])
        ax.grid(False)
        ax.set_title(LABEL[m], fontsize=8.5)
    fig.suptitle("Confusion matrices (pooled test folds; % of each true class)", fontsize=10, x=0.01, ha="left")
    save(fig, "confusion")

    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    ax.plot([0, 1], [0, 1], ls=":", color=MUTED, lw=1)
    for m in models:
        b = c[m]["calibration"]
        ax.plot([x["mean_prob"] for x in b], [x["frac_pos"] for x in b], color=COLOR[m], marker=MARK[m], ms=4, label=LABEL[m])
    ax.set(xlabel="predicted probability", ylabel="observed fraction positive", title="Calibration (reliability)")
    ax.legend(fontsize=7)
    save(fig, "calibration")


def heatmaps(models):
    ps = pd.read_csv(os.path.join(EVAL_DIR, "per_source.csv"))
    ps["metric"] = np.where(ps["positives"] > 0, ps.get("recall"), ps.get("false_positive_rate"))
    ps["row"] = ps["source"] + np.where(ps["positives"] > 0, "  (recall)", "  (false-pos. rate)")
    piv = ps.pivot_table(index="row", columns="model", values="metric")[models]
    pt = pd.read_csv(os.path.join(EVAL_DIR, "per_topic.csv")).pivot_table(index="topic", columns="model", values="f1")[models]
    for name, tab, title, fmt in (("per_source", piv, "Per-source behaviour (pooled test folds)", "{:.2f}"),
                                  ("per_topic", pt, "F1 per topic (topic-bias check)", "{:.2f}")):
        fig, ax = plt.subplots(figsize=(1.3 * len(models) + 3, 0.32 * len(tab) + 1.2))
        ax.imshow(tab.to_numpy(dtype=float), cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#f3f7fd", "#2a78d6"]),
                  aspect="auto", vmin=0, vmax=1)
        for i in range(tab.shape[0]):
            for j in range(tab.shape[1]):
                v = tab.iat[i, j]
                ax.text(j, i, "" if pd.isna(v) else fmt.format(v), ha="center", va="center", fontsize=7,
                        color="white" if (not pd.isna(v) and v > 0.6) else INK)
        ax.set_xticks(range(len(models)), [LABEL[m].split(" ")[0] for m in models], fontsize=8)
        ax.set_yticks(range(len(tab)), tab.index, fontsize=7.5)
        ax.grid(False)
        ax.set_title(title, loc="left", fontsize=10)
        save(fig, name)


def ensembles():
    p = os.path.join(ENS_DIR, "ensembles.csv")
    if not os.path.exists(p):
        return
    e = pd.read_csv(p)
    md_table(e.head(20), os.path.join(TAB, "ensembles_top.md"))
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    rule_style = {"single": ("o", INK), "soft": ("s", COLOR["distilbert"]), "hard": ("^", COLOR["modernbert"]),
                  "stacking": ("D", COLOR["tinyllama"])}
    for rule, (mk, col) in rule_style.items():
        g = e[e["rule"] == rule]
        if len(g) and g["latency_ms_sum"].notna().any():
            ax.scatter(g["latency_ms_sum"], g["f1"], marker=mk, color=col, s=34, label=rule, zorder=3)
    for _, r in e[e["rule"] == "single"].iterrows():
        ax.annotate(r["members"], (r["latency_ms_sum"], r["f1"]), xytext=(4, -10), textcoords="offset points", fontsize=7)
    best = e.iloc[0]
    ax.annotate(f"best: {best['members']} ({best['rule']})", (best["latency_ms_sum"], best["f1"]), xytext=(5, 6),
                textcoords="offset points", fontsize=7)
    ax.set_xscale("log")
    ax.set(xlabel="summed GPU latency per message (ms, log scale)", ylabel="pooled test F1",
           title="All model subsets: accuracy vs cost")
    ax.legend(fontsize=7)
    save(fig, "ensembles")
    d = os.path.join(ENS_DIR, "diversity.csv")
    if os.path.exists(d):
        dv = pd.read_csv(d)
        ms = sorted(set(dv.model_a) | set(dv.model_b), key=MODEL_NAMES.index)
        mat = pd.DataFrame(np.eye(len(ms)), index=ms, columns=ms)
        for _, r in dv.iterrows():
            mat.loc[r.model_a, r.model_b] = mat.loc[r.model_b, r.model_a] = r.error_correlation
        fig, ax = plt.subplots(figsize=(3.8, 3.2))
        ax.imshow(mat.to_numpy(dtype=float), cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#f3f7fd", "#2a78d6"]),
                  vmin=0, vmax=1)
        for i in range(len(ms)):
            for j in range(len(ms)):
                ax.text(j, i, f"{mat.iat[i, j]:.2f}", ha="center", va="center", fontsize=8)
        ax.set_xticks(range(len(ms)), [LABEL[m].split(" ")[0] for m in ms], rotation=30, fontsize=7.5)
        ax.set_yticks(range(len(ms)), [LABEL[m].split(" ")[0] for m in ms], fontsize=7.5)
        ax.grid(False)
        ax.set_title("Error correlation between models", fontsize=10, loc="left")
        save(fig, "diversity")


def main():
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(TAB, exist_ok=True)
    data_card()
    models = models_present()
    if not models:
        print("No trained models yet - only the data card was produced.")
        return
    curves(models)
    if os.path.exists(os.path.join(EVAL_DIR, "summary.csv")):
        comparison(models)
        roc_pr_conf_calib(models)
        heatmaps(models)
    lc = os.path.join(EVAL_DIR, "llm_comparison.csv")
    if os.path.exists(lc):
        md_table(pd.read_csv(lc), os.path.join(TAB, "llm_comparison.md"))
    ensembles()
    print(f"figures -> {FIG}\ntables  -> {TAB}")


if __name__ == "__main__":
    main()
