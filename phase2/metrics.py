"""Metric helpers shared by train / evaluate / ensemble / llm_baseline (positive class = 1 = 'states a preference')."""

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)


def _tune_threshold_slow(y, p):
    """Reference implementation (kept for verification): try every distinct probability as the threshold."""
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    cands = np.unique(np.concatenate([p, [0.5]]))
    best, best_t = -1.0, 0.5
    for t in cands:
        f = f1_score(y, p >= t, zero_division=0)
        if f > best + 1e-12 or (abs(f - best) <= 1e-12 and abs(t - 0.5) < abs(best_t - 0.5)):
            best, best_t = f, float(t)
    return best_t


def tune_threshold(y, p):
    """Threshold maximising F1 on (validation) labels y and probabilities p; ties -> the value closest to 0.5.
    Vectorised: identical result to `_tune_threshold_slow`, computed in O(n log n) instead of O(n^2)."""
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    cands = np.unique(np.concatenate([p, [0.5]]))                 # ascending
    order = np.sort(p)
    pos_sorted = np.sort(p[y == 1])
    n_pos = int(y.sum())
    # predictions are p >= t: count of positives / all rows with p >= t
    tp = n_pos - np.searchsorted(pos_sorted, cands, side="left")
    pred_pos = len(p) - np.searchsorted(order, cands, side="left")
    denom = pred_pos + n_pos
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    best = f1.max()
    ties = np.where(f1 >= best - 1e-12)[0]
    return float(cands[ties[np.argmin(np.abs(cands[ties] - 0.5))]])


def ece(y, p, n_bins=10):
    """Expected calibration error of P(y=1)."""
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def metrics_from_preds(y, prob, pred):
    """Pooled metrics when every fold used its own threshold: hard metrics from `pred`, ranking metrics from `prob`."""
    y, prob, pred = np.asarray(y).astype(int), np.asarray(prob, dtype=float), np.asarray(pred).astype(int)
    out = {"n": int(len(y)), "positives": int(y.sum()),
           "accuracy": float(accuracy_score(y, pred)),
           "precision": float(precision_score(y, pred, zero_division=0)),
           "recall": float(recall_score(y, pred, zero_division=0)),
           "f1": float(f1_score(y, pred, zero_division=0)),
           "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0))}
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
                "fpr": float(fp / (fp + tn)) if fp + tn else 0.0})
    if 0 < y.sum() < len(y):
        out["roc_auc"] = float(roc_auc_score(y, prob))
        out["pr_auc"] = float(average_precision_score(y, prob))
    out["ece"] = ece(y, prob)
    return out


def binary_metrics(y, p, tau=0.5):
    """All reported metrics for labels y, probabilities p and decision threshold tau."""
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    yhat = (p >= tau).astype(int)
    out = {"n": int(len(y)), "positives": int(y.sum()), "threshold": float(tau),
           "accuracy": float(accuracy_score(y, yhat)),
           "precision": float(precision_score(y, yhat, zero_division=0)),
           "recall": float(recall_score(y, yhat, zero_division=0)),
           "f1": float(f1_score(y, yhat, zero_division=0)),
           "f1_macro": float(f1_score(y, yhat, average="macro", zero_division=0))}
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    out.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
                "fpr": float(fp / (fp + tn)) if fp + tn else 0.0})
    if 0 < y.sum() < len(y):
        out["roc_auc"] = float(roc_auc_score(y, p))
        out["pr_auc"] = float(average_precision_score(y, p))
    out["ece"] = ece(y, p)
    return out
