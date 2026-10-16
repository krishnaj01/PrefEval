"""Step 6: train one model on every cross-validation round (resumable per fold).

    python -m phase2.train --config configs/phase2/distilbert.yaml --fold all
    python -m phase2.train --config configs/phase2/qwen3.yaml --fold 2
    python -m phase2.train --config configs/phase2/distilbert.yaml --smoke     # tiny end-to-end check

Per fold k (test = fold k, val = fold k+1, train = rest):
  * HF Trainer; validation every 1/`evals_per_epoch` epoch -> loss curves + early stopping on validation F1
  * decision threshold tau tuned on the validation set (max F1)
  * saved to results/phase2/<model>/fold<k>/ :
        metrics.json         val/test metrics (at tau and at 0.5), timings, memory, params, revision, latency
        log_history.csv      every logged train loss / validation metric (loss curves)
        predictions_{val,test,ood}.jsonl   uid, label, prob, pred, fold, source, form, topic
  * best model -> models/phase2/<model>/fold<k>/best (git-ignored; Phase 3 loads it)
A fold whose metrics.json exists is skipped (use --force to retrain).
"""

import argparse
import csv
import json
import math
import os
import shutil
import sys
import time

import numpy as np
import torch
from datasets import Dataset
from transformers import DataCollatorWithPadding, EarlyStoppingCallback, Trainer, TrainingArguments
from transformers import set_seed as hf_set_seed

from phase2.common import (DATASET, LMSYS_OOD, MODELS_DIR, REPO, RESULTS_DIR, data_config, model_config, read_jsonl,
                           set_seed, write_jsonl, write_run_manifest)
from phase2.metrics import binary_metrics, tune_threshold
from phase2.models import build, param_counts, resolve_revision
from phase2.pseudolabel import LMSYS_CACHE
from phase2.splits import fold_roles

KEEP_FIELDS = ("uid", "fold", "source", "form", "topic", "label_origin")


def softmax_pos(logits):
    logits = np.asarray(logits, dtype=np.float64)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return (e[:, 1] / e.sum(axis=1)).astype(float)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    p = softmax_pos(logits)
    m = binary_metrics(labels, p, 0.5)
    return {k: m[k] for k in ("accuracy", "precision", "recall", "f1", "f1_macro", "roc_auc") if k in m}


def gpu_check():
    if not torch.cuda.is_available():
        print("WARNING: no CUDA GPU visible - training will be very slow.")
        return
    free, total = torch.cuda.mem_get_info()
    if free / total < 0.7:
        print(f"WARNING: only {free / 2**30:.1f} of {total / 2**30:.1f} GiB GPU memory free. "
              f"If Ollama still holds a model, free it with: ollama stop <model>")


def measure_latency(model, tok, texts, device, max_len, n=200):
    """Median / mean ms per single message (batch size 1), after 5 warm-up calls."""
    model.eval()
    times = []
    with torch.no_grad():
        for i, t in enumerate(texts[: n + 5]):
            enc = tok(t, truncation=True, max_length=max_len, return_tensors="pt").to(device)
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(**enc)
            if device == "cuda":
                torch.cuda.synchronize()
            if i >= 5:
                times.append((time.perf_counter() - t0) * 1000)
    return {"device": device, "n": len(times), "median_ms": float(np.median(times)), "mean_ms": float(np.mean(times))}


def to_dataset(rows, tok, max_len):
    ds = Dataset.from_dict({"text": [r["text"] for r in rows], "label": [int(r.get("label") or 0) for r in rows]})
    return ds.map(lambda b: tok(b["text"], truncation=True, max_length=max_len), batched=True, remove_columns=["text"])


def smoke_subset(rows, n, seed):
    rng = np.random.default_rng(seed)
    pos = [r for r in rows if r["label"] == 1]
    neg = [r for r in rows if r["label"] == 0]
    k = min(len(pos), n // 3)
    pick = [pos[i] for i in rng.choice(len(pos), k, replace=False)] + \
           [neg[i] for i in rng.choice(len(neg), min(len(neg), n - k), replace=False)]
    return pick


def train_fold(cfg, k, args, all_rows, ood_rows, ood_labels, revision):
    dcfg = data_config()
    seed, n_folds = dcfg["seed"], dcfg["n_folds"]
    name = cfg["name"]
    res_root = os.path.join(REPO, "results", "phase2_smoke") if args.smoke else RESULTS_DIR
    mod_root = os.path.join(REPO, "models", "phase2_smoke") if args.smoke else MODELS_DIR
    out_dir = os.path.join(res_root, name, f"fold{k}")
    if os.path.exists(os.path.join(out_dir, "metrics.json")) and not args.force:
        print(f"[{name}] fold {k}: already done, skipping")
        return
    os.makedirs(out_dir, exist_ok=True)
    set_seed(seed)
    hf_set_seed(seed)

    tr_f, va_f, te_f = fold_roles(k, n_folds)
    train = [r for r in all_rows if r["fold"] in tr_f]
    val = [r for r in all_rows if r["fold"] == va_f]
    test = [r for r in all_rows if r["fold"] == te_f]
    ood = list(ood_rows)
    epochs = cfg["epochs"]
    if args.smoke:
        train, val, test, ood = smoke_subset(train, 300, seed), smoke_subset(val, 150, seed), \
            smoke_subset(test, 150, seed), ood[:50]
        epochs = 1
    print(f"[{name}] fold {k}: train={len(train)} val={len(val)} test={len(test)} ood={len(ood)}")

    tok, model = build(cfg, revision)
    total_p, train_p = param_counts(model)
    ds_train, ds_val = to_dataset(train, tok, cfg["max_len"]), to_dataset(val, tok, cfg["max_len"])
    steps_per_epoch = math.ceil(len(train) / (cfg["batch_size"] * cfg.get("grad_accum", 1)))
    eval_steps = max(1, steps_per_epoch // cfg.get("evals_per_epoch", 4))
    warmup_steps = int(round(cfg.get("warmup_ratio", 0.0) * steps_per_epoch * epochs))   # transformers v5: steps only
    ckpt_dir = os.path.join(mod_root, name, f"fold{k}", "ckpt")
    targs = TrainingArguments(
        output_dir=ckpt_dir, per_device_train_batch_size=cfg["batch_size"], per_device_eval_batch_size=cfg["batch_size"] * 2,
        gradient_accumulation_steps=cfg.get("grad_accum", 1), learning_rate=float(cfg["lr"]), num_train_epochs=epochs,
        weight_decay=cfg.get("weight_decay", 0.0), warmup_steps=warmup_steps, lr_scheduler_type="linear",
        eval_strategy="steps", eval_steps=eval_steps, save_strategy="steps", save_steps=eval_steps, save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="f1", greater_is_better=True,
        logging_steps=max(1, eval_steps // 4), bf16=cfg.get("precision") == "bf16", seed=seed, data_seed=seed,
        report_to="none", dataloader_num_workers=2, disable_tqdm=False)
    trainer = Trainer(model=model, args=targs, train_dataset=ds_train, eval_dataset=ds_val,
                      data_collator=DataCollatorWithPadding(tok), compute_metrics=compute_metrics,
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.get("early_stopping_patience", 4))])
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    resume = None
    if os.path.isdir(ckpt_dir) and any(d.startswith("checkpoint-") for d in os.listdir(ckpt_dir)):
        resume = True                                   # crashed mid-fold: continue from the last checkpoint
    t0 = time.time()
    trainer.train(resume_from_checkpoint=resume)
    train_time = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None

    def predict(rows):
        if not rows:
            return np.array([])
        out = trainer.predict(to_dataset(rows, tok, cfg["max_len"]))
        logits = out.predictions[0] if isinstance(out.predictions, tuple) else out.predictions
        return softmax_pos(logits)

    p_val, p_test, p_ood = predict(val), predict(test), predict(ood)
    y_val = np.array([r["label"] for r in val])
    y_test = np.array([r["label"] for r in test])
    tau = tune_threshold(y_val, p_val)
    metrics = {"model": name, "hf_id": cfg["hf_id"], "revision": revision, "fold": k, "test_fold": te_f, "val_fold": va_f,
               "train_folds": tr_f, "n_train": len(train), "n_val": len(val), "n_test": len(test), "threshold": tau,
               "val": binary_metrics(y_val, p_val, tau), "test": binary_metrics(y_test, p_test, tau),
               "test_at_0.5": binary_metrics(y_test, p_test, 0.5), "train_time_s": round(train_time, 1),
               "peak_gpu_mem_gib": peak_mem, "params_total": total_p, "params_trainable": train_p,
               "best_checkpoint_step": trainer.state.best_global_step if hasattr(trainer.state, "best_global_step") else None,
               "epochs_run": trainer.state.epoch, "config": cfg, "smoke": bool(args.smoke)}
    ood_lab = [(ood_labels.get(r["uid"]), p) for r, p in zip(ood, p_ood)]
    metrics["ood_lmsys"] = {"n": len(ood), "flag_rate": float(np.mean(p_ood >= tau)) if len(ood) else None}
    labelled = [(y, p) for y, p in ood_lab if y is not None]
    if labelled and 0 < sum(y for y, _ in labelled) < len(labelled):
        metrics["ood_lmsys"]["vs_pseudo"] = binary_metrics([y for y, _ in labelled], [p for _, p in labelled], tau)

    # latency (batch 1) measured once per model, on the first fold trained
    if k == 0 or args.smoke:
        texts = [r["text"] for r in test]
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        metrics["latency"] = {"gpu" if dev == "cuda" else "cpu": measure_latency(trainer.model, tok, texts, dev, cfg["max_len"],
                                                                                 n=50 if args.smoke else 200)}

    def dump(split, rows, probs):
        write_jsonl(os.path.join(out_dir, f"predictions_{split}.jsonl"),
                    [{**{f: r.get(f) for f in KEEP_FIELDS}, "label": r.get("label"),
                      "pseudo_label": ood_labels.get(r["uid"]) if split == "ood" else None,
                      "prob": float(p), "pred": int(p >= tau)} for r, p in zip(rows, probs)])
    dump("val", val, p_val)
    dump("test", test, p_test)
    dump("ood", ood, p_ood)
    hist = trainer.state.log_history
    keys = sorted({kk for h in hist for kk in h})
    with open(os.path.join(out_dir, "log_history.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(hist)
    best_dir = os.path.join(mod_root, name, f"fold{k}", "best")
    trainer.save_model(best_dir)
    tok.save_pretrained(best_dir)
    with open(os.path.join(best_dir, "phase2_threshold.json"), "w") as f:
        json.dump({"threshold": tau, "revision": revision, "fold": k}, f)
    shutil.rmtree(ckpt_dir, ignore_errors=True)                      # intermediate checkpoints no longer needed
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)
    t = metrics["test"]
    print(f"[{name}] fold {k}: tau={tau:.3f} test F1={t['f1']:.3f} P={t['precision']:.3f} R={t['recall']:.3f} "
          f"acc={t['accuracy']:.3f} AUC={t.get('roc_auc', float('nan')):.3f}  ({train_time / 60:.1f} min)")
    write_run_manifest({"event": "train_fold", "model": name, "fold": k, "revision": revision, "smoke": bool(args.smoke),
                        "train_time_s": round(train_time, 1), "test_f1": t["f1"], "threshold": tau},
                       path=os.path.join(res_root, "run_manifest.jsonl"))
    del trainer, model
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--fold", default="all", help="fold index or 'all'")
    ap.add_argument("--smoke", action="store_true", help="fold 0, ~600 rows, 1 epoch, separate output folders")
    ap.add_argument("--force", action="store_true", help="retrain folds that already have metrics.json")
    args = ap.parse_args()
    cfg = model_config(args.config)
    if not os.path.exists(DATASET):
        sys.exit("data/phase2/dataset.jsonl not found - run python -m phase2.splits first.")
    gpu_check()
    revision = resolve_revision(cfg["hf_id"], cfg.get("revision"))
    print(f"model {cfg['name']} = {cfg['hf_id']} @ {revision}")
    rows = read_jsonl(DATASET)
    ood_rows = read_jsonl(LMSYS_OOD)
    ood_labels = {r["uid"]: r["label"] for r in read_jsonl(LMSYS_CACHE)}
    n_folds = data_config()["n_folds"]
    folds = [0] if args.smoke else (range(n_folds) if args.fold == "all" else [int(args.fold)])
    for k in folds:
        train_fold(cfg, k, args, rows, ood_rows, ood_labels, revision)


if __name__ == "__main__":
    main()
