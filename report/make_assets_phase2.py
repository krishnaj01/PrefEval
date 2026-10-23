"""Generate every table and number used in the Phase 2 report (report/phase2_report.tex).

    python report/make_assets_phase2.py      (repo root, `prefeval-p2` or `prefeval` env)

Reads only data/phase2/ and results/phase2/ and writes report/generated/p2_*.tex.
Figures are taken directly from results/phase2/figures/ (made by `python -m phase2.report`).
"""

import json
import os

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data", "phase2")
RES = os.path.join(REPO, "results", "phase2")
OUT = os.path.join(REPO, "report", "generated")
os.makedirs(OUT, exist_ok=True)

LABEL = {"distilbert": "DistilBERT", "modernbert": "ModernBERT", "tinyllama": "TinyLlama-1.1B (LoRA)",
         "qwen3": "Qwen3-1.7B (LoRA)"}
SHORT = {"distilbert": "DistilBERT", "modernbert": "ModernBERT", "tinyllama": "TinyLlama", "qwen3": "Qwen3"}
ORDER = ["distilbert", "modernbert", "tinyllama", "qwen3"]
SRC = {"explicit_pref": "Explicit preference", "persona_pref": "Persona preference field",
       "choice_selection": "Choice selection (implicit)", "persona_turn": "Persona dialogue turn",
       "choice_query": "Choice initial query", "question": "Final question", "oasst": "OpenAssistant message"}
STRATA = {"persona_pos": "Persona, LLM-picked turn (+)", "persona_neg": "Persona, other turns (-)",
          "persona_llm_vs_overlap_disagree": "Persona, LLM vs overlap disagree",
          "oasst_pos": "OpenAssistant, LLM yes (+)", "oasst_neg": "OpenAssistant, LLM no (-)"}


def j(*p):
    with open(os.path.join(*p)) as f:
        return json.load(f)


def write(name, lines):
    with open(os.path.join(OUT, name), "w") as f:
        f.write("\n".join(lines) + "\n")


def tab(cols, header, rows, name):
    write(name, [r"\begin{tabular}{" + cols + "}", r"\toprule", header + r" \\", r"\midrule",
                 *[r + r" \\" for r in rows], r"\bottomrule", r"\end{tabular}"])


card = j(DATA, "data_card.json")
audit = j(DATA, "audit_report.json")
man = j(DATA, "manifest.json")
summ = pd.read_csv(os.path.join(RES, "eval", "summary.csv")).set_index("model").loc[ORDER]
llm = pd.read_csv(os.path.join(RES, "eval", "llm_comparison.csv"))
ens = pd.read_csv(os.path.join(RES, "ensemble", "ensembles.csv"))
div = pd.read_csv(os.path.join(RES, "ensemble", "diversity.csv"))
ps = pd.read_csv(os.path.join(RES, "eval", "per_source.csv"))
llm_m = j(RES, "llm_baseline", "metrics.json")

# ------------------------------------------------------------------ data composition
rows = []
for s in SRC:
    n0 = card["by_source_label"].get(f"{s}|0", 0)
    n1 = card["by_source_label"].get(f"{s}|1", 0)
    origin = "gold" if s in ("explicit_pref", "persona_pref", "choice_selection", "choice_query", "question") else "pseudo (LLM)"
    rows.append(f"{SRC[s]} & {origin} & {n1:,} & {n0:,} & {n0 + n1:,}")
rows.append(r"\midrule Total & & " + f"{card['positives']:,} & {card['negatives']:,} & {card['rows']:,}")
tab("llrrr", r"Source & Label origin & Label 1 & Label 0 & Rows", rows, "p2_data.tex")

# ------------------------------------------------------------------ audit before / after refinement
ref = card["audit_agreement_of_refined_labels"]
rows = []
for k, name in STRATA.items():
    rows.append(f"{name} & {audit[k]['n']} & {audit[k]['agreement']:.3f} & {ref[k]['agreement']:.3f}")
rows.append(r"\midrule All strata & " + f"{audit['overall']['n']} & {audit['overall']['agreement']:.3f} "
            f"($\\kappa$={audit['overall']['cohen_kappa']:.2f}) & {ref['overall']['agreement']:.3f}")
tab("lrrr", r"Audit stratum & $n$ & Before refinement & After refinement", rows, "p2_audit.tex")

# ------------------------------------------------------------------ model comparison
rows = []
for m in ORDER:
    r = summ.loc[m]
    rows.append(f"{LABEL[m]} & {r.f1:.3f} & {r.precision:.3f} & {r.recall:.3f} & {r.f1_fold_std:.3f} & {r.roc_auc:.3f} & "
                f"{r.ece:.3f} & {100 * r.ood_flag_rate:.1f}\\% & {r.gpu_latency_ms:.1f} & {r.params_trainable / 1e6:.0f}M & "
                f"{r.train_time_min:.0f}")
tab("lrrrrrrrrrr", r"Model & F1 & Prec. & Rec. & F1 std & AUC & ECE & OOD flags & ms/msg & Trainable & Train min", rows,
    "p2_models.tex")

# ------------------------------------------------------------------ LLM comparison (same 2,000 rows)
rows = []
for _, r in llm.iterrows():
    name = "Llama-3-8B, zero-shot prompt" if r.model.startswith("llm") else LABEL[r.model]
    rows.append(f"{name} & {r.f1:.3f} & {r.precision:.3f} & {r.recall:.3f} & {r.accuracy:.3f} & {r.median_latency_ms:.1f}")
tab("lrrrrr", r"Detector & F1 & Precision & Recall & Accuracy & ms / message", rows, "p2_llm.tex")

# ------------------------------------------------------------------ per-source behaviour (from test predictions)
def test_preds(m):
    rows = []
    for k in range(5):
        with open(os.path.join(RES, m, f"fold{k}", "predictions_test.jsonl")) as f:
            rows += [json.loads(l) for l in f]
    return pd.DataFrame(rows)


preds = {m: test_preds(m) for m in ORDER}
src_rows, src_stats = [], {}
for s in ["explicit_pref", "choice_selection", "persona_turn", "oasst", "question", "choice_query"]:
    rec, fpr = [], []
    for m in ORDER:
        g = preds[m][preds[m].source == s]
        p1, p0 = g[g.label == 1], g[g.label == 0]
        rec.append(p1.pred.mean() if len(p1) else None)
        fpr.append(p0.pred.mean() if len(p0) else None)
        src_stats[(s, m)] = (rec[-1], fpr[-1], len(p1), len(p0))
    fmt = lambda xs: " & ".join("--" if x is None else f"{x:.3f}" for x in xs)
    if any(x is not None for x in rec):
        src_rows.append(f"{SRC[s]} & recall & {fmt(rec)}")
    if any(x is not None for x in fpr):
        src_rows.append(f"{SRC[s]} & false-pos. rate & {fmt(fpr)}")
tab("llrrrr", r"Source & Metric & " + " & ".join(SHORT[m] for m in ORDER), src_rows, "p2_source.tex")


# ------------------------------------------------------------------ ensembles: best per rule + all singles
best = ens.sort_values("f1", ascending=False)
rows = []
for rule in ["soft", "hard", "stacking"]:
    r = best[best.rule == rule].iloc[0]
    rows.append(f"Best {rule} & {' + '.join(SHORT[x] for x in r.members.split('+'))} & {r.f1:.3f} & {r.precision:.3f} & {r.recall:.3f} & {r.latency_ms_sum:.1f}")
for i, m in enumerate(ORDER):
    r = ens[(ens.rule == "single") & (ens.members == m)].iloc[0]
    rows.append(("\\midrule " if i == 0 else "") + f"Single & {SHORT[m]} & {r.f1:.3f} & {r.precision:.3f} & {r.recall:.3f} & {r.latency_ms_sum:.1f}")
tab("llrrrr", r"Rule & Members & F1 & Prec. & Rec. & ms (sum)", rows, "p2_ensemble.tex")

# ------------------------------------------------------------------ macros
d = {r.model_a + r.model_b: r for _, r in div.iterrows()}
f1s = summ["f1"]
best_ens = best.iloc[0]
db = llm[llm.model == "distilbert"].iloc[0]
macros = {
    "pRows": f"{card['rows']:,}", "pPos": f"{card['positives']:,}", "pNeg": f"{card['negatives']:,}",
    "pGold": f"{card['by_label_origin']['gold']:,}", "pPseudo": f"{card['by_label_origin']['pseudo']:,}",
    "pAudited": f"{card['by_label_origin']['audited']:,}",
    "pDupDropped": f"{man['build_data']['prefeval_turns']['duplicates_dropped']:,}",
    "pPrefevalTurns": f"{man['build_data']['prefeval_turns']['rows']:,}",
    "pOasstPool": f"{man['oasst_sample']['pool']:,}", "pOasstRev": man["oasst_sample"]["revision"][:8],
    "pPersonaFound": str(man["pseudo_persona"]["found"]),
    "pOverlapAgree": f"{100 * man['pseudo_persona']['overlap_agreement']:.1f}",
    "pOasstYes": str(man["pseudo_oasst"]["yes"]),
    "pStrictConfirmed": str(man["pseudo_refine"]["oasst_strict"]["1"]),
    "pStrictRejected": str(man["pseudo_refine"]["oasst_strict"]["0"]),
    "pGenericExcl": f"{card['persona_filter_comparison']['generic']['excluded_total']:,}",
    "pStrictExcl": str(card["persona_filter_comparison"]["strict"]["excluded_total"]),
    "pGenericPrec": f"{card['persona_filter_comparison']['generic']['flag_precision']:.2f}",
    "pStrictPrec": f"{card['persona_filter_comparison']['strict']['flag_precision']:.2f}",
    "pAuditBefore": f"{100 * audit['overall']['agreement']:.1f}", "pAuditAfter": f"{100 * ref['overall']['agreement']:.1f}",
    "pKappa": f"{audit['overall']['cohen_kappa']:.2f}",
    "pTopicRateLo": f"{card['topic_positive_rate_min_max'][0]:.2f}", "pTopicRateHi": f"{card['topic_positive_rate_min_max'][1]:.2f}",
    "pFOneLo": f"{f1s.min():.3f}", "pFOneHi": f"{f1s.max():.3f}",
    "pBestEnsFOne": f"{best_ens.f1:.3f}", "pBestEnsMembers": " + ".join(SHORT[x] for x in best_ens.members.split("+")),
    "pErrCorrLo": f"{div.error_correlation.min():.2f}", "pErrCorrHi": f"{div.error_correlation.max():.2f}",
    "pDisagreeMax": f"{100 * div.disagreement.max():.1f}",
    "pLlmFOne": f"{llm_m['metrics']['f1']:.3f}", "pLlmPrec": f"{llm_m['metrics']['precision']:.3f}",
    "pLlmRec": f"{llm_m['metrics']['recall']:.3f}", "pLlmFpr": f"{100 * llm_m['metrics']['fpr']:.1f}", "pLlmLat": f"{llm_m['latency']['median_ms']:.0f}",
    "pDistilLat": f"{summ.loc['distilbert', 'gpu_latency_ms']:.1f}",
    "pSpeedup": f"{llm_m['latency']['median_ms'] / summ.loc['distilbert', 'gpu_latency_ms']:.0f}",
    "pDistilSameFOne": f"{db.f1:.3f}",
    "pOodLo": f"{100 * summ.ood_flag_rate.min():.1f}", "pOodHi": f"{100 * summ.ood_flag_rate.max():.1f}",
    "pTotalTrainH": f"{summ.train_time_min.sum() / 60:.1f}",
}
oasst_rec = [src_stats[("oasst", m)][0] for m in ORDER]
macros["pOasstRecLo"] = f"{100 * min(oasst_rec):.0f}"
macros["pOasstRecHi"] = f"{100 * max(oasst_rec):.0f}"
macros["pOasstPosTest"] = str(src_stats[("oasst", "distilbert")][2])
write("p2_macros.tex", [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()])
print(json.dumps(macros, indent=1))
