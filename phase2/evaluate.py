"""Step 8: evaluate every trained model from its saved predictions (no model reloading).

    python -m phase2.evaluate

For each model (all 5 test folds pooled; each fold uses the threshold tuned on its own validation fold):
  summary.csv        accuracy, precision, recall, F1, macro-F1, ROC-AUC, PR-AUC, ECE, confusion counts,
                     fold mean +- std of F1, LMSYS-OOD flag rate, latency, params, training time, peak GPU memory
  per_fold.csv       test metrics per fold (variance across unseen topic groups)
  per_source.csv     recall on each positive source / false-positive rate on each negative source
  per_topic.csv      F1 per PrefEval topic (topic-bias check)
  llm_comparison.csv every model vs the LLM-prompted baseline on the identical 2,000-row sample
  curves.json        ROC + precision-recall curve points and calibration bins (for figures)
  errors/<model>.md  most confident false positives / false negatives (error analysis)
Outputs go to results/phase2/eval/.
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from phase2.common import DATASET, MODEL_NAMES, RESULTS_DIR, read_jsonl, write_run_manifest
from phase2.metrics import metrics_from_preds

EVAL_DIR = os.path.join(RESULTS_DIR, "eval")


def available_models():
    out = []
    for m in MODEL_NAMES:
        d = os.path.join(RESULTS_DIR, m)
        if os.path.isdir(d) and any(os.path.exists(os.path.join(d, f, "metrics.json")) for f in os.listdir(d)):
            out.append(m)
    return out


def load_model_results(name):
    d = os.path.join(RESULTS_DIR, name)
    folds = sorted(f for f in os.listdir(d) if f.startswith("fold") and os.path.exists(os.path.join(d, f, "metrics.json")))
    metrics = [json.load(open(os.path.join(d, f, "metrics.json"))) for f in folds]
    preds = {s: pd.DataFrame([r for f in folds for r in read_jsonl(os.path.join(d, f, f"predictions_{s}.jsonl"))])
             for s in ("val", "test", "ood")}
    return metrics, preds


def calibration_bins(y, p, n=10):
    bins = np.linspace(0, 1, n + 1)
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p <= hi) if lo == 0 else (p > lo) & (p <= hi)
        if m.any():
            out.append({"bin_lo": float(lo), "bin_hi": float(hi), "n": int(m.sum()), "mean_prob": float(p[m].mean()),
                        "frac_pos": float(y[m].mean())})
    return out


def main():
    os.makedirs(os.path.join(EVAL_DIR, "errors"), exist_ok=True)
    texts = {r["uid"]: r["text"] for r in read_jsonl(DATASET)}
    models = available_models()
    if not models:
        raise SystemExit("No trained models found in results/phase2/.")
    summary, per_fold, per_source, per_topic, curves = [], [], [], [], {}
    llm_path = os.path.join(RESULTS_DIR, "llm_baseline", "predictions.jsonl")
    llm = pd.DataFrame(read_jsonl(llm_path)) if os.path.exists(llm_path) else None
    llm_rows = []

    for name in models:
        metrics, preds = load_model_results(name)
        te = preds["test"]
        pooled = metrics_from_preds(te["label"], te["prob"], te["pred"])
        f1s = [m["test"]["f1"] for m in metrics]
        lat = next((m["latency"] for m in metrics if m.get("latency")), {})
        ood = preds["ood"]
        thr = {m["fold"]: m["threshold"] for m in metrics}
        row = {"model": name, "folds": len(metrics), **{k: pooled[k] for k in
               ("n", "accuracy", "precision", "recall", "f1", "f1_macro", "roc_auc", "pr_auc", "ece", "tp", "fp", "fn", "tn")},
               "f1_fold_mean": float(np.mean(f1s)), "f1_fold_std": float(np.std(f1s)),
               "ood_flag_rate": float(ood["pred"].mean()) if len(ood) else None,
               "gpu_latency_ms": lat.get("gpu", {}).get("median_ms"),
               "params_total": metrics[0]["params_total"], "params_trainable": metrics[0]["params_trainable"],
               "train_time_min": sum(m["train_time_s"] for m in metrics) / 60,
               "peak_gpu_mem_gib": max((m["peak_gpu_mem_gib"] or 0) for m in metrics),
               "revision": metrics[0]["revision"]}
        if len(ood) and ood["pseudo_label"].notna().any():
            o = ood[ood["pseudo_label"].notna()]
            om = metrics_from_preds(o["pseudo_label"].astype(int), o["prob"], o["pred"])
            row.update({"ood_precision_vs_llm": om["precision"], "ood_recall_vs_llm": om["recall"]})
        summary.append(row)
        for m in metrics:
            per_fold.append({"model": name, "fold": m["fold"], "threshold": m["threshold"],
                             **{k: m["test"][k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc") if k in m["test"]},
                             "train_time_min": m["train_time_s"] / 60})
        for (src, form), g in te.groupby(["source", "form"], dropna=False):
            gl = g["label"].astype(int)
            entry = {"model": name, "source": src, "form": form, "n": len(g), "positives": int(gl.sum())}
            if gl.sum():
                entry["recall"] = float((g["pred"][gl == 1] == 1).mean())
            if (gl == 0).any():
                entry["false_positive_rate"] = float((g["pred"][gl == 0] == 1).mean())
            per_source.append(entry)
        for topic, g in te[te["topic"].notna()].groupby("topic"):
            mm = metrics_from_preds(g["label"], g["prob"], g["pred"])
            per_topic.append({"model": name, "topic": topic, "n": mm["n"], "f1": mm["f1"],
                              "precision": mm["precision"], "recall": mm["recall"]})
        y, p = te["label"].astype(int).to_numpy(), te["prob"].to_numpy()
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        idx = np.linspace(0, len(fpr) - 1, min(len(fpr), 400)).astype(int)
        jdx = np.linspace(0, len(prec) - 1, min(len(prec), 400)).astype(int)
        curves[name] = {"roc": {"fpr": fpr[idx].tolist(), "tpr": tpr[idx].tolist()},
                        "pr": {"precision": prec[jdx].tolist(), "recall": rec[jdx].tolist()},
                        "calibration": calibration_bins(y, p),
                        "confusion": {k: pooled[k] for k in ("tn", "fp", "fn", "tp")}}
        # error analysis: most confident mistakes
        fp_rows = te[(te["label"] == 0) & (te["pred"] == 1)].sort_values("prob", ascending=False).head(15)
        fn_rows = te[(te["label"] == 1) & (te["pred"] == 0)].sort_values("prob").head(15)
        with open(os.path.join(EVAL_DIR, "errors", f"{name}.md"), "w") as f:
            f.write(f"# {name}: most confident errors (pooled test folds)\n\n")
            for title, rs in (("False positives (label 0, predicted 1)", fp_rows), ("False negatives (label 1, predicted 0)", fn_rows)):
                f.write(f"## {title}\n\n| prob | source | label origin | text |\n|---|---|---|---|\n")
                for _, r in rs.iterrows():
                    t = texts.get(r["uid"], "").replace("|", "/").replace("\n", " ")[:220]
                    f.write(f"| {r['prob']:.3f} | {r['source']} | {r['label_origin']} | {t} |\n")
                f.write("\n")
        # same-sample comparison with the LLM baseline
        if llm is not None and len(llm):
            sub = te.set_index("uid").loc[llm["uid"]]
            mm = metrics_from_preds(sub["label"], sub["prob"], sub["pred"])
            llm_rows.append({"model": name, **{k: mm[k] for k in ("n", "accuracy", "precision", "recall", "f1")},
                             "median_latency_ms": row["gpu_latency_ms"]})

    if llm is not None and len(llm):
        lm = json.load(open(os.path.join(RESULTS_DIR, "llm_baseline", "metrics.json")))
        llm_rows.append({"model": "llm_baseline (Llama-3-8B zero-shot)",
                         **{k: lm["metrics"][k] for k in ("n", "accuracy", "precision", "recall", "f1")},
                         "median_latency_ms": lm["latency"]["median_ms"]})
        pd.DataFrame(llm_rows).to_csv(os.path.join(EVAL_DIR, "llm_comparison.csv"), index=False, float_format="%.4f")
    pd.DataFrame(summary).to_csv(os.path.join(EVAL_DIR, "summary.csv"), index=False, float_format="%.4f")
    pd.DataFrame(per_fold).to_csv(os.path.join(EVAL_DIR, "per_fold.csv"), index=False, float_format="%.4f")
    pd.DataFrame(per_source).to_csv(os.path.join(EVAL_DIR, "per_source.csv"), index=False, float_format="%.4f")
    pd.DataFrame(per_topic).to_csv(os.path.join(EVAL_DIR, "per_topic.csv"), index=False, float_format="%.4f")
    with open(os.path.join(EVAL_DIR, "curves.json"), "w") as f:
        json.dump(curves, f)
    print(pd.DataFrame(summary)[["model", "folds", "accuracy", "precision", "recall", "f1", "f1_fold_std", "roc_auc",
                                 "ood_flag_rate", "gpu_latency_ms"]].to_string(index=False, float_format="%.3f"))
    if llm_rows:
        print("\nSame-sample comparison with the LLM-prompted detector:")
        print(pd.DataFrame(llm_rows).to_string(index=False, float_format="%.3f"))
    write_run_manifest({"event": "evaluate", "models": models})


if __name__ == "__main__":
    main()
