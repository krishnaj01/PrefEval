"""Step 9: ensembles of every non-empty subset of the trained models (4 models -> 15 subsets).

    python -m phase2.ensemble

Combination rules (per cross-validation round; everything is fitted on the VALIDATION fold, applied to the TEST fold):
  soft      mean of member probabilities; threshold tuned on validation (max F1)
  hard      majority vote of members' own decisions (each at its own tuned threshold); ties -> soft-vote decision
  stacking  logistic regression on members' validation log-odds; its threshold is tuned with 5-fold cross-validated
            predictions on the validation fold (so the threshold is not fitted on the same predictions it is scored on)
Single-model subsets reproduce the individual results (sanity check).

Outputs (results/phase2/ensemble/):
  ensembles.csv  pooled test metrics + fold mean/std F1 + summed latency/params for every subset x rule
  diversity.csv  pairwise disagreement, error correlation and Yule's Q between models (why ensembling helps or not)
"""

import itertools
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

from phase2.common import MODEL_NAMES, RESULTS_DIR, data_config, read_jsonl, write_run_manifest
from phase2.evaluate import EVAL_DIR
from sklearn.metrics import f1_score

from phase2.metrics import metrics_from_preds, tune_threshold

ENS_DIR = os.path.join(RESULTS_DIR, "ensemble")


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def load_round(models, k):
    """Aligned val/test frames for round k: columns uid, label, p_<model>, d_<model> (decision at the model's tau)."""
    out = {}
    for split in ("val", "test"):
        frame = None
        for m in models:
            rows = read_jsonl(os.path.join(RESULTS_DIR, m, f"fold{k}", f"predictions_{split}.jsonl"))
            df = pd.DataFrame(rows)[["uid", "label", "prob", "pred"]].rename(columns={"prob": f"p_{m}", "pred": f"d_{m}"})
            frame = df if frame is None else frame.merge(df.drop(columns="label"), on="uid", how="inner")
        out[split] = frame.sort_values("uid").reset_index(drop=True)
    return out


def combine(rule, members, val, test, seed):
    pv = val[[f"p_{m}" for m in members]].to_numpy()
    pt = test[[f"p_{m}" for m in members]].to_numpy()
    yv = val["label"].astype(int).to_numpy()
    if rule == "soft" or len(members) == 1:
        tau = tune_threshold(yv, pv.mean(1))
        prob = pt.mean(1)
        return prob, (prob >= tau).astype(int)
    if rule == "hard":
        dt = test[[f"d_{m}" for m in members]].to_numpy()
        votes = dt.mean(1)
        tau = tune_threshold(yv, pv.mean(1))
        soft = (pt.mean(1) >= tau).astype(int)
        pred = np.where(votes > 0.5, 1, np.where(votes < 0.5, 0, soft))
        return votes, pred
    if rule == "stacking":
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
        cv_prob = cross_val_predict(clf, logit(pv), yv, cv=5, method="predict_proba")[:, 1]
        tau = tune_threshold(yv, cv_prob)
        clf.fit(logit(pv), yv)
        prob = clf.predict_proba(logit(pt))[:, 1]
        return prob, (prob >= tau).astype(int)
    raise ValueError(rule)


def main():
    os.makedirs(ENS_DIR, exist_ok=True)
    cfg = data_config()
    n_folds, seed = cfg["n_folds"], cfg["seed"]
    done = {m: {k for k in range(n_folds) if os.path.exists(os.path.join(RESULTS_DIR, m, f"fold{k}", "metrics.json"))}
            for m in MODEL_NAMES}
    models = [m for m in MODEL_NAMES if done[m]]
    if not models:
        raise SystemExit("No trained model found.")
    folds = sorted(set.intersection(*(done[m] for m in models)))       # rounds every model has finished
    if len(folds) < n_folds:
        print(f"NOTE: using rounds {folds} only (not every model has all {n_folds} folds yet)")
    summary = pd.read_csv(os.path.join(EVAL_DIR, "summary.csv")).set_index("model") if os.path.exists(
        os.path.join(EVAL_DIR, "summary.csv")) else None
    rounds = {k: load_round(models, k) for k in folds}
    subsets = [s for r in range(1, len(models) + 1) for s in itertools.combinations(models, r)]
    rows = []
    for subset in subsets:
        for rule in (["single"] if len(subset) == 1 else ["soft", "hard", "stacking"]):
            ys, ps, ds, fold_f1 = [], [], [], []
            for k in folds:
                val, test = rounds[k]["val"], rounds[k]["test"]
                prob, pred = combine("soft" if rule == "single" else rule, subset, val, test, seed)
                y = test["label"].astype(int).to_numpy()
                ys.append(y), ps.append(prob), ds.append(pred)
                fold_f1.append(f1_score(y, pred, zero_division=0))
            m = metrics_from_preds(np.concatenate(ys), np.concatenate(ps), np.concatenate(ds))
            lat = params = None
            if summary is not None:
                lat = float(sum(summary.loc[s, "gpu_latency_ms"] or 0 for s in subset))
                params = int(sum(summary.loc[s, "params_total"] for s in subset))
            rows.append({"members": "+".join(subset), "n_members": len(subset), "rule": rule,
                         **{k: m[k] for k in ("accuracy", "precision", "recall", "f1", "f1_macro", "roc_auc", "pr_auc",
                                              "tp", "fp", "fn", "tn")},
                         "f1_fold_mean": float(np.mean(fold_f1)), "f1_fold_std": float(np.std(fold_f1)),
                         "latency_ms_sum": lat, "params_sum": params})
    ens = pd.DataFrame(rows).sort_values(["f1", "n_members"], ascending=[False, True])
    ens.to_csv(os.path.join(ENS_DIR, "ensembles.csv"), index=False, float_format="%.4f")

    # pairwise diversity on pooled test decisions
    pooled = pd.concat([rounds[k]["test"] for k in folds], ignore_index=True)
    y = pooled["label"].astype(int).to_numpy()
    div = []
    for a, b in itertools.combinations(models, 2):
        da, db = pooled[f"d_{a}"].to_numpy(), pooled[f"d_{b}"].to_numpy()
        ea, eb = (da != y).astype(int), (db != y).astype(int)
        n11, n00 = int(((ea == 0) & (eb == 0)).sum()), int(((ea == 1) & (eb == 1)).sum())
        n10, n01 = int(((ea == 0) & (eb == 1)).sum()), int(((ea == 1) & (eb == 0)).sum())
        q = (n11 * n00 - n01 * n10) / (n11 * n00 + n01 * n10) if (n11 * n00 + n01 * n10) else 0.0
        div.append({"model_a": a, "model_b": b, "disagreement": float((da != db).mean()),
                    "error_correlation": float(np.corrcoef(ea, eb)[0, 1]) if ea.std() and eb.std() else None,
                    "yules_q": float(q), "both_wrong": n00, "only_a_wrong": n01, "only_b_wrong": n10})
    pd.DataFrame(div).to_csv(os.path.join(ENS_DIR, "diversity.csv"), index=False, float_format="%.4f")
    print(ens.head(12)[["members", "rule", "accuracy", "precision", "recall", "f1", "f1_fold_std", "latency_ms_sum"]]
          .to_string(index=False, float_format="%.3f"))
    write_run_manifest({"event": "ensemble", "models": models, "subsets": len(subsets)})


if __name__ == "__main__":
    main()
