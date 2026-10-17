"""Step 5: assemble the final labelled dataset and assign topic-level cross-validation folds.

    python -m phase2.splits

Inputs : data/phase2/prefeval_turns.jsonl (gold), pseudo/persona.jsonl, oasst_sample.jsonl + pseudo/oasst.jsonl,
         audit_sample.csv (manual labels override pseudo labels when present).
Outputs: data/phase2/dataset.jsonl  - every trainable row with `label`, `label_origin` and `fold`
         data/phase2/splits.json    - topics per fold + row counts per role, with sha256 of dataset.jsonl
         data/phase2/data_card.json - counts per source x label, class balance per topic (topic-bias check)

Fold roles for fold k (5 folds):  test = k,  val = (k+1) % 5,  train = the other 3.
PrefEval rows are grouped by TOPIC (4 topics per fold, so the test topics are unseen in training); OASST rows are grouped by
conversation tree. LMSYS turns are excluded entirely (evaluation only).
"""

import collections
import csv
import hashlib
import json
import os
import random

from phase1.upstream.common_utils import ALL_TOPICS
from phase2.common import (AUDIT_CSV, DATA_DIR, DATASET, OASST_RAW, PREFEVAL_TURNS, SPLITS, data_config, read_jsonl,
                           sha256_file, update_data_manifest, write_jsonl)
from phase2.pseudolabel import (OASST_CACHE, OASST_STRICT_CACHE, PERSONA_CACHE, PERSONA_FILTERS, PERSONA_OVERLAP_MIN)

DATA_CARD = os.path.join(DATA_DIR, "data_card.json")


def fold_roles(k, n_folds):
    """Return (train_folds, val_fold, test_fold) for cross-validation round k."""
    val = (k + 1) % n_folds
    return [f for f in range(n_folds) if f not in (k, val)], val, k


def topic_folds(seed, n_folds):
    topics = list(ALL_TOPICS)
    random.Random(seed).shuffle(topics)
    per = len(topics) // n_folds
    return {f: sorted(topics[f * per:(f + 1) * per]) for f in range(n_folds)}


def assemble():
    """Combine gold rows, (refined) pseudo-labels and manual audit labels.

    Persona turn  -> 1 if it is the LLM's pick OR the overlap check's pick (>= PERSONA_OVERLAP_MIN shared words);
                     otherwise 0, unless the generic yes/no check says it states a preference -> excluded (ambiguous).
    OASST message -> LLM YES/NO; a YES is kept only if the stricter re-check confirms it, else it becomes 0 (hard negative).
    Audit         -> manual labels override everything (label_origin = 'audited').
    Returns rows plus `pre_audit` = the label each audited uid had before the override (to measure refinement quality).
    """
    rows = []
    persona = {r["key"]: r for r in read_jsonl(PERSONA_CACHE)}
    filt = data_config().get("persona_negative_filter", "generic")      # generic | strict | none
    pcheck = {} if filt == "none" else {r["uid"]: r["label"] for r in read_jsonl(PERSONA_FILTERS[filt])}
    excluded = collections.Counter()
    for r in read_jsonl(PREFEVAL_TURNS):
        if r["source"] == "persona_turn":
            p = persona.get(f"{r['topic']}/{r['task_id']}")
            if p is None or p["turn_idx"] is None:          # dialogue unlabelled / LLM said NONE
                excluded["persona_turn_unlabelled"] += 1
                continue
            ov = p.get("overlap_idx") if (p.get("overlap_score") or 0) >= PERSONA_OVERLAP_MIN else None
            lab = int(r["turn_idx"] in (p["turn_idx"], ov))
            if not lab and pcheck.get(r["uid"]) == 1:
                excluded["persona_turn_ambiguous"] += 1
                continue
            r = {**r, "label": lab, "label_origin": "pseudo", "form": "implicit-persona" if lab else "none"}
        rows.append(r)
    oasst = {r["uid"]: r["label"] for r in read_jsonl(OASST_CACHE)}
    strict = {r["uid"]: r["label"] for r in read_jsonl(OASST_STRICT_CACHE)}
    for r in read_jsonl(OASST_RAW):
        lab = oasst.get(r["uid"])
        if lab is None:
            excluded["oasst_unlabelled"] += 1
            continue
        if lab == 1 and r["uid"] in strict:
            lab = 1 if strict[r["uid"]] == 1 else 0
        rows.append({**r, "label": lab, "label_origin": "pseudo", "form": "chat" if lab else "none"})
    # manual audit overrides
    n_over, pre_audit = 0, {}
    if os.path.exists(AUDIT_CSV):
        with open(AUDIT_CSV) as f:
            human = {a["uid"]: int(a["your_label"]) for a in csv.DictReader(f) if a["your_label"].strip() in ("0", "1")}
        for r in rows:
            if r["uid"] in human:
                pre_audit[r["uid"]] = r["label"]
                n_over += r["label"] != human[r["uid"]]
                r["label"], r["label_origin"] = human[r["uid"]], "audited"
                if r["source"] == "persona_turn":
                    r["form"] = "implicit-persona" if r["label"] else "none"
    return rows, excluded, n_over, pre_audit


def compare_persona_filters():
    """How well each persona-negative filter matches the human audit on the persona_neg stratum.
    'flag precision' = of the audited turns the filter excluded, the share that humans also said contain a preference."""
    if not os.path.exists(AUDIT_CSV):
        return {}
    with open(AUDIT_CSV) as f:
        rows = [a for a in csv.DictReader(f) if a["stratum"] == "persona_neg" and a["your_label"].strip() in ("0", "1")]
    out = {}
    for name, path in PERSONA_FILTERS.items():
        chk = {r["uid"]: r["label"] for r in read_jsonl(path)}
        if not chk:
            continue
        flagged = [a for a in rows if chk.get(a["uid"]) == 1]
        kept = [a for a in rows if chk.get(a["uid"]) == 0]
        total = len(read_jsonl(path))
        out[name] = {"excluded_total": sum(1 for r in read_jsonl(path) if r["label"] == 1), "checked_total": total,
                     "audited_flagged": len(flagged), "flag_precision": round(sum(a["your_label"] == "1" for a in flagged) / len(flagged), 3) if flagged else None,
                     "audited_kept": len(kept), "kept_clean_rate": round(sum(a["your_label"] == "0" for a in kept) / len(kept), 3) if kept else None,
                     "missed_preferences_in_kept": sum(a["your_label"] == "1" for a in kept)}
    return out


def refined_audit_agreement(pre_audit):
    """Agreement between the human audit and the REFINED pseudo-labels (before the override), per stratum."""
    if not os.path.exists(AUDIT_CSV):
        return {}
    with open(AUDIT_CSV) as f:
        rows = [a for a in csv.DictReader(f) if a["your_label"].strip() in ("0", "1")]
    out = {}
    for s in ["overall"] + sorted({a["stratum"] for a in rows}):
        rs = [a for a in rows if (s == "overall" or a["stratum"] == s)]
        scored = [a for a in rs if a["uid"] in pre_audit]
        out[s] = {"n_audited": len(rs), "n_still_in_dataset": len(scored),
                  "agreement": round(sum(pre_audit[a["uid"]] == int(a["your_label"]) for a in scored) / len(scored), 4)
                  if scored else None}
    return out


def main():
    cfg = data_config()
    n_folds, seed = cfg["n_folds"], cfg["seed"]
    rows, excluded, n_over, pre_audit = assemble()
    refined = refined_audit_agreement(pre_audit)
    tf = topic_folds(seed, n_folds)
    topic_to_fold = {t: f for f, ts in tf.items() for t in ts}
    for r in rows:
        if r["topic"] is not None:
            r["fold"] = topic_to_fold[r["topic"]]
        else:                                                   # OASST: stable hash of the conversation tree id
            r["fold"] = int(hashlib.sha256(r["group"].encode()).hexdigest(), 16) % n_folds
    rows.sort(key=lambda r: r["uid"])
    write_jsonl(DATASET, rows)

    # ---------------------------------------------------------------- leakage assertions
    assert not any(r["source"] == "lmsys_ood" for r in rows), "LMSYS turns must never enter the training data"
    for k in range(n_folds):
        tr, va, te = fold_roles(k, n_folds)
        topics = {role: {r["topic"] for r in rows if r["fold"] in fs and r["topic"]} for role, fs in
                  (("train", tr), ("val", [va]), ("test", [te]))}
        assert not (topics["train"] & topics["test"]) and not (topics["train"] & topics["val"]) \
            and not (topics["val"] & topics["test"]), f"topic overlap in fold {k}"
        groups = {role: {r["group"] for r in rows if r["fold"] in fs} for role, fs in (("train", tr), ("test", [te]))}
        assert not (groups["train"] & groups["test"]), f"group overlap in fold {k}"

    # ---------------------------------------------------------------- splits + data card
    roles = {}
    for k in range(n_folds):
        tr, va, te = fold_roles(k, n_folds)
        roles[k] = {"train_folds": tr, "val_fold": va, "test_fold": te,
                    "n_train": sum(r["fold"] in tr for r in rows), "n_val": sum(r["fold"] == va for r in rows),
                    "n_test": sum(r["fold"] == te for r in rows)}
    splits = {"seed": seed, "n_folds": n_folds, "topic_folds": tf, "oasst_fold_rule": "sha256(conversation tree id) % n_folds",
              "rounds": roles, "dataset_sha256": sha256_file(DATASET), "rows": len(rows)}
    with open(SPLITS, "w") as f:
        json.dump(splits, f, indent=1)

    by_src = collections.Counter((r["source"], r["label"]) for r in rows)
    by_origin = collections.Counter(r["label_origin"] for r in rows)
    per_topic = {}
    for t in ALL_TOPICS:
        rs = [r for r in rows if r["topic"] == t]
        per_topic[t] = {"n": len(rs), "positive_rate": round(sum(r["label"] for r in rs) / len(rs), 4)}
    rates = [v["positive_rate"] for v in per_topic.values()]
    card = {"rows": len(rows), "positives": sum(r["label"] for r in rows),
            "negatives": sum(1 - r["label"] for r in rows),
            "by_source_label": {f"{s}|{l}": n for (s, l), n in sorted(by_src.items())},
            "by_label_origin": dict(by_origin), "excluded": dict(excluded), "audit_overrides_changed": n_over,
            "audit_agreement_of_refined_labels": refined,
            "persona_negative_filter": data_config().get("persona_negative_filter", "generic"),
            "persona_filter_comparison": compare_persona_filters(),
            "per_topic": per_topic, "topic_positive_rate_min_max": [min(rates), max(rates)],
            "per_fold": {f: {"n": sum(r["fold"] == f for r in rows), "pos": sum(r["label"] for r in rows if r["fold"] == f)}
                         for f in range(n_folds)}}
    with open(DATA_CARD, "w") as f:
        json.dump(card, f, indent=1)
    update_data_manifest("splits", {"dataset_sha256": splits["dataset_sha256"], "rows": len(rows),
                                    "splits_sha256": sha256_file(SPLITS)})
    print(f"dataset: {len(rows)} rows ({card['positives']} positive / {card['negatives']} negative); "
          f"label origin {dict(by_origin)}; excluded {dict(excluded)}; audit changed {n_over} labels")
    print(f"per-topic positive rate range: {min(rates):.3f} - {max(rates):.3f}")
    cmp = card["persona_filter_comparison"]
    if cmp:
        print(f"persona-negative filter in use: {card['persona_negative_filter']}")
        for k, v in cmp.items():
            print(f"  {k:8s} excluded {v['excluded_total']}/{v['checked_total']} | on audit: flagged {v['audited_flagged']} "
                  f"(precision {v['flag_precision']}), kept {v['audited_kept']} (clean {v['kept_clean_rate']}, "
                  f"missed preferences {v['missed_preferences_in_kept']})")
    if refined:
        print("audit agreement of the REFINED pseudo-labels (rows excluded as ambiguous are not scored):")
        for k, v in refined.items():
            print(f"  {k:34s} agreement={v['agreement']}  ({v['n_still_in_dataset']}/{v['n_audited']} audited rows still in dataset)")
    for k, v in roles.items():
        print(f"round {k}: test fold {v['test_fold']} {tf[v['test_fold']]}  train={v['n_train']} val={v['n_val']} test={v['n_test']}")


if __name__ == "__main__":
    main()
