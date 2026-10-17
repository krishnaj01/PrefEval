"""Steps 2-4: LLM pseudo-labels (local Ollama) + manual audit.

    python -m phase2.pseudolabel --what persona     # which persona-dialogue user turn reveals the known preference
    python -m phase2.pseudolabel --what oasst       # sample OpenAssistant user turns + yes/no label
    python -m phase2.pseudolabel --what lmsys       # yes/no label for the LMSYS distractor turns (evaluation only)
    python -m phase2.pseudolabel --audit            # write data/phase2/audit_sample.csv for manual review
    python -m phase2.pseudolabel --audit-report     # agreement of the filled-in audit with the pseudo-labels

Every LLM answer is appended (fsync) to a cache in data/phase2/pseudo/*.jsonl together with the prompt hash and
model digest, so runs are resumable and a finished label is never re-queried. Labeller = Llama-3-8B-Instruct (Q8)
via phase1.backend.OllamaBackend, temperature 0, under its Phase 1 Ollama settings (enforced by the settings guard).
"""

import argparse
import collections
import csv
import os
import random
import re
import sys
import time

from tqdm import tqdm

from phase1.backend import OllamaBackend
from phase1.runner import check_ollama_env, load_yaml, MODELS_YAML
from phase2.common import (AUDIT_CSV, AUDIT_REPORT, DATASET_DIR, LMSYS_OOD, OASST_RAW, PREFEVAL_TURNS, PSEUDO_DIR,
                           append_jsonl, data_config, read_jsonl, sha256_file, sha256_text, update_data_manifest,
                           write_jsonl)

PERSONA_CACHE = os.path.join(PSEUDO_DIR, "persona.jsonl")
OASST_CACHE = os.path.join(PSEUDO_DIR, "oasst.jsonl")
LMSYS_CACHE = os.path.join(PSEUDO_DIR, "lmsys.jsonl")
# refinement caches (added after the manual audit showed noisy OASST positives / persona negatives)
OASST_STRICT_CACHE = os.path.join(PSEUDO_DIR, "oasst_strict.jsonl")     # stricter re-check of OASST "yes" labels
PERSONA_CHECK_CACHE = os.path.join(PSEUDO_DIR, "persona_check.jsonl")   # generic yes/no on persona negatives
PERSONA_CHECK_STRICT_CACHE = os.path.join(PSEUDO_DIR, "persona_check_strict.jsonl")   # strict prompt on the same turns
PERSONA_FILTERS = {"generic": PERSONA_CHECK_CACHE, "strict": PERSONA_CHECK_STRICT_CACHE}
PERSONA_OVERLAP_MIN = 2      # overlap pick counts as a positive turn only with >= 2 shared content words

# ----------------------------------------------------------------------------- prompts (hashed into every cache row)
DEFINITION = ("A message STATES A PREFERENCE if it reveals a personal preference, like, dislike, restriction or constraint "
              "of the user (e.g. diet, allergy, budget, accessibility, taste, values, things they avoid or love) that "
              "should shape future recommendations or answers - stated directly or implied by a choice or a passing remark. "
              "Plain requests, questions, instructions, greetings, thanks and small talk do NOT state a preference.")

YESNO_TEMPLATE = (DEFINITION + "\n\nExamples:\n"
                  "\"I'm vegan, so please keep that in mind.\" -> YES\n"
                  "\"I'll take the second option, the others are too loud for me.\" -> YES\n"
                  "\"As someone with a severe nut allergy, what snacks are safe on a flight?\" -> YES\n"
                  "\"What are some good hotels in Rome?\" -> NO\n"
                  "\"Write a Python function that reverses a string.\" -> NO\n"
                  "\"Thanks, that was helpful!\" -> NO\n\n"
                  "Message:\n\"\"\"\n{text}\n\"\"\"\n\nDoes this message state a preference? Answer YES or NO only.")

PERSONA_TEMPLATE = ("A user has the following preference:\n\"{preference}\"\n\n"
                    "Below are the user's messages from one conversation, numbered. In exactly one of them (sometimes "
                    "none) the user reveals, states or implies this preference, often in passing.\n\n{turns}\n\n"
                    "Which message number reveals the preference? Answer with the number only, or NONE.")

_STOP = set("i a an the and or to of in on for with my me is am are be by as at it this that from all any due so do not no "
             "avoid prefer only very really strictly strict".split())


def overlap_turn(preference, turns):
    """Independent, LLM-free cross-check: the user turn sharing most content words (>3 letters) with the known preference."""
    words = lambda t: {w for w in re.findall(r"[a-z]+", t.lower()) if len(w) > 3 and w not in _STOP}
    pw = words(preference)
    scores = [len(pw & words(v["user"])) for _, v in turns]
    best = max(range(len(scores)), key=lambda i: scores[i])
    return (best if scores[best] > 0 else None), scores[best]


STRICT_TEMPLATE = ("Decide whether the message reveals a LASTING PERSONAL preference or constraint of the USER THEMSELVES - "
                   "e.g. their diet, allergy, health, budget, accessibility needs, tastes, likes and dislikes, or values - that an "
                   "assistant should remember and apply to FUTURE, DIFFERENT requests.\n\n"
                   "Answer NO if the message only contains: requirements for the current task (programming language, length, "
                   "format, style, tools), feedback on the assistant's previous answer, role-play or fiction instructions, "
                   "statements about other people, questions or requests without a personal preference, or one-off wishes.\n\n"
                   "Message:\n\"\"\"\n{text}\n\"\"\"\n\nDoes this message reveal a lasting personal preference or "
                   "constraint of the user? Answer YES or NO only.")

SYSTEM = "You are a careful annotator. Follow the instructions exactly and answer with the requested format only."


def llama3_prompt(user_msg):
    """Raw Llama-3 chat template (the backend strips the literal BOS; Ollama adds one)."""
    return ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" + SYSTEM + "<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n" + user_msg + "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n")


def make_backend():
    cfg = data_config()["labeler"]
    models = load_yaml(MODELS_YAML)
    mc = models[cfg["model"]]
    err = check_ollama_env(cfg["model"], mc)
    if err:
        sys.exit(f"{err}\n(Pseudo-labels are produced under the labeller's Phase 1 Ollama settings for reproducibility.)")
    be = OllamaBackend(mc["ollama_tag"], cfg.get("num_ctx", mc["num_ctx"]))
    return be, be.model_info()


def ask(backend, text, max_tokens):
    t0 = time.time()
    raw = backend.generate(llama3_prompt(text), max_tokens, purpose="label").strip()
    return raw, round(time.time() - t0, 3)


def parse_yesno(raw):
    m = re.match(r"\W*(yes|no)\b", raw, re.I)
    return None if not m else int(m.group(1).lower() == "yes")


# ----------------------------------------------------------------------------- persona
def label_persona():
    import json
    backend, info = make_backend()
    done = {r["key"] for r in read_jsonl(PERSONA_CACHE)}
    jobs = []
    from phase1.upstream.common_utils import ALL_TOPICS
    for topic in ALL_TOPICS:
        with open(os.path.join(DATASET_DIR, "implicit_preference", "persona-driven", f"{topic}.json")) as f:
            data = json.load(f)
        for i, d in enumerate(data):
            key = f"{topic}/{i}"
            if key not in done:
                jobs.append((key, d))
    print(f"persona dialogues to label: {len(jobs)} (already cached: {len(done)})")
    for key, d in tqdm(jobs, desc="persona"):
        turns = sorted(d["conversation"].items(), key=lambda kv: int(kv[0]))
        listing = "\n".join(f"[{int(t) + 1}] {v['user'].strip()}" for t, v in turns)   # 1-based for the LLM
        prompt = PERSONA_TEMPLATE.format(preference=d["preference"], turns=listing)
        raw, sec = ask(backend, prompt, 6)
        m = re.match(r"\W*(\d+)", raw)
        idx = int(m.group(1)) - 1 if m and 1 <= int(m.group(1)) <= len(turns) else None
        ov_idx, ov_score = overlap_turn(d["preference"], turns)
        append_jsonl(PERSONA_CACHE, {"key": key, "turn_idx": idx, "raw": raw, "n_turns": len(turns), "seconds": sec,
                                     "overlap_idx": ov_idx, "overlap_score": ov_score,
                                     "prompt_sha": sha256_text(PERSONA_TEMPLATE + SYSTEM), "model": info["tag"],
                                     "digest": info["digest"]})
    rows = read_jsonl(PERSONA_CACHE)
    found = sum(r["turn_idx"] is not None for r in rows)
    both = [r for r in rows if r["turn_idx"] is not None and r.get("overlap_idx") is not None]
    agree = sum(r["turn_idx"] == r["overlap_idx"] for r in both) / max(len(both), 1)
    pos = collections.Counter(r["turn_idx"] for r in rows if r["turn_idx"] is not None)
    print(f"persona labelled: {len(rows)} dialogues; preference turn found in {found}, NONE/unparseable in {len(rows) - found}")
    print(f"agreement with the word-overlap cross-check: {agree:.1%} of {len(both)}; LLM picks by turn: {dict(sorted(pos.items()))}")
    update_data_manifest("pseudo_persona", {"cache": os.path.relpath(PERSONA_CACHE), "sha256": sha256_file(PERSONA_CACHE),
                                            "dialogues": len(rows), "found": found, "overlap_agreement": round(agree, 4),
                                            "llm_turn_distribution": {str(k): v for k, v in pos.items()}, "model": info})


# ----------------------------------------------------------------------------- OASST
def sample_oasst():
    """Deterministic sample of English OpenAssistant user ('prompter') messages."""
    if os.path.exists(OASST_RAW):
        return read_jsonl(OASST_RAW)
    from datasets import load_dataset
    from huggingface_hub import HfApi
    cfg = data_config()
    oc = cfg["oasst"]
    revision = oc.get("revision") or HfApi().dataset_info(oc["dataset"]).sha     # pin the exact dataset commit
    ds = load_dataset(oc["dataset"], revision=revision)
    from phase2.build_data import norm
    seen, pool = set(), []
    for split in ("train", "validation"):
        for m in ds[split]:
            if m["role"] != "prompter" or m["lang"] != oc["lang"] or m.get("deleted"):
                continue
            text = (m["text"] or "").strip()
            if not text or len(text) > oc["max_chars"] or norm(text) in seen:
                continue
            seen.add(norm(text))
            pool.append({"uid": f"oasst/{m['message_id']}", "text": text, "label": None, "label_origin": None,
                         "source": "oasst", "form": None, "topic": None, "task_id": None, "turn_idx": None,
                         "group": f"oasst_tree/{m['message_tree_id']}"})
    pool.sort(key=lambda r: r["uid"])
    sample = random.Random(cfg["seed"]).sample(pool, min(oc["n_samples"], len(pool)))
    write_jsonl(OASST_RAW, sample)
    update_data_manifest("oasst_sample", {"dataset": oc["dataset"], "revision": revision, "pool": len(pool),
                                          "sampled": len(sample), "seed": cfg["seed"], "sha256": sha256_file(OASST_RAW)})
    return sample


def label_yesno(rows, cache, name):
    backend, info = make_backend()
    done = {r["uid"] for r in read_jsonl(cache)}
    todo = [r for r in rows if r["uid"] not in done]
    print(f"{name}: {len(todo)} to label (already cached: {len(done)})")
    for r in tqdm(todo, desc=name):
        raw, sec = ask(backend, YESNO_TEMPLATE.format(text=r["text"][:4000]), 3)
        append_jsonl(cache, {"uid": r["uid"], "label": parse_yesno(raw), "raw": raw, "seconds": sec,
                             "prompt_sha": sha256_text(YESNO_TEMPLATE + SYSTEM), "model": info["tag"],
                             "digest": info["digest"]})
    got = read_jsonl(cache)
    c = collections.Counter(r["label"] for r in got)
    print(f"{name} labels: yes={c[1]} no={c[0]} unparseable={c[None]}")
    update_data_manifest(f"pseudo_{name}", {"cache": os.path.relpath(cache), "sha256": sha256_file(cache),
                                            "rows": len(got), "yes": c[1], "no": c[0], "unparseable": c[None],
                                            "model": info})


# ----------------------------------------------------------------------------- refinement (post-audit)
def persona_negative_candidates():
    """Persona turns that neither the LLM nor the overlap check picked (candidate negatives)."""
    persona = {r["key"]: r for r in read_jsonl(PERSONA_CACHE)}
    cand = []
    for r in read_jsonl(PREFEVAL_TURNS):
        if r["source"] != "persona_turn":
            continue
        p = persona.get(f"{r['topic']}/{r['task_id']}")
        if p is None or p["turn_idx"] is None:
            continue
        ov = p.get("overlap_idx") if (p.get("overlap_score") or 0) >= PERSONA_OVERLAP_MIN else None
        if r["turn_idx"] not in (p["turn_idx"], ov):
            cand.append(r)
    return cand


def _check(rows, cache, template, desc, backend, info):
    done = {r["uid"] for r in read_jsonl(cache)}
    for r in tqdm([r for r in rows if r["uid"] not in done], desc=desc):
        raw, sec = ask(backend, template.format(text=r["text"][:4000]), 3)
        append_jsonl(cache, {"uid": r["uid"], "label": parse_yesno(raw), "raw": raw, "seconds": sec,
                             "prompt_sha": sha256_text(template + SYSTEM), "model": info["tag"], "digest": info["digest"]})


def refine():
    """(a) Stricter re-check of OASST rows the LLM labelled YES (rejected ones become hard negatives).
    (b) Generic YES/NO check on persona candidate negatives (a YES excludes the turn as ambiguous)."""
    backend, info = make_backend()
    oasst_yes = {r["uid"] for r in read_jsonl(OASST_CACHE) if r["label"] == 1}
    _check([r for r in read_jsonl(OASST_RAW) if r["uid"] in oasst_yes], OASST_STRICT_CACHE, STRICT_TEMPLATE,
           "oasst-strict", backend, info)
    _check(persona_negative_candidates(), PERSONA_CHECK_CACHE, YESNO_TEMPLATE, "persona-neg-check", backend, info)
    st = collections.Counter(r["label"] for r in read_jsonl(OASST_STRICT_CACHE))
    pc = collections.Counter(r["label"] for r in read_jsonl(PERSONA_CHECK_CACHE))
    print(f"OASST strict re-check of {sum(st.values())} YES rows: confirmed={st[1]} rejected={st[0]} unparseable={st[None]}")
    print(f"persona negatives checked (generic): {sum(pc.values())}: flagged={pc[1]}, kept={pc[0]}")
    update_data_manifest("pseudo_refine", {"oasst_strict": {"sha256": sha256_file(OASST_STRICT_CACHE), **{str(k): v for k, v in st.items()}},
                                           "persona_check": {"sha256": sha256_file(PERSONA_CHECK_CACHE), **{str(k): v for k, v in pc.items()}},
                                           "persona_overlap_min": PERSONA_OVERLAP_MIN, "model": info})


def refine_persona_strict():
    """Same candidate persona negatives, checked with STRICT_TEMPLATE (lasting personal preference only)."""
    backend, info = make_backend()
    _check(persona_negative_candidates(), PERSONA_CHECK_STRICT_CACHE, STRICT_TEMPLATE, "persona-neg-strict", backend, info)
    pc = collections.Counter(r["label"] for r in read_jsonl(PERSONA_CHECK_STRICT_CACHE))
    print(f"persona negatives checked (strict): {sum(pc.values())}: flagged={pc[1]}, kept={pc[0]}, unparseable={pc[None]}")
    update_data_manifest("pseudo_refine_persona_strict", {"sha256": sha256_file(PERSONA_CHECK_STRICT_CACHE),
                                                          **{str(k): v for k, v in pc.items()}, "model": info})


# ----------------------------------------------------------------------------- audit
AUDIT_FIELDS = ["audit_id", "stratum", "uid", "reference_preference", "text", "pseudo_label", "your_label", "notes"]


def persona_rows_with_labels():
    """Persona turns with their pseudo label (1 = the turn the LLM chose; 0 = other turns of a dialogue with a choice)."""
    import json
    full = {r["key"]: r for r in read_jsonl(PERSONA_CACHE)}
    cache = {k: r["turn_idx"] for k, r in full.items()}
    prefs = {}
    out = []
    for r in read_jsonl(PREFEVAL_TURNS):
        if r["source"] != "persona_turn":
            continue
        key = f"{r['topic']}/{r['task_id']}"
        if cache.get(key) is None:
            continue   # NONE / unparseable dialogue -> excluded from training
        if r["topic"] not in prefs:
            with open(os.path.join(DATASET_DIR, "implicit_preference", "persona-driven", f"{r['topic']}.json")) as f:
                prefs[r["topic"]] = [d["preference"] for d in json.load(f)]
        ov = full[key].get("overlap_idx")
        out.append({**r, "pseudo": int(r["turn_idx"] == cache[key]), "ref": prefs[r["topic"]][r["task_id"]],
                    "disagree": ov is not None and ov != cache[key] and r["turn_idx"] in (ov, cache[key])})
    return out


def write_audit():
    if os.path.exists(AUDIT_CSV):
        sys.exit(f"{AUDIT_CSV} already exists (it may contain your manual labels) - not overwriting.")
    cfg = data_config()
    rng = random.Random(cfg["seed"])
    k = cfg["audit"]["n_per_stratum"]
    persona = persona_rows_with_labels()
    oasst_lab = {r["uid"]: r["label"] for r in read_jsonl(OASST_CACHE)}
    oasst = [{**r, "pseudo": oasst_lab[r["uid"]], "ref": ""} for r in read_jsonl(OASST_RAW) if oasst_lab.get(r["uid"]) is not None]
    strata = {"persona_pos": [r for r in persona if r["pseudo"] == 1 and not r["disagree"]],
              "persona_neg": [r for r in persona if r["pseudo"] == 0 and not r["disagree"]],
              "persona_llm_vs_overlap_disagree": [r for r in persona if r["disagree"]],
              "oasst_pos": [r for r in oasst if r["pseudo"] == 1], "oasst_neg": [r for r in oasst if r["pseudo"] == 0]}
    rows = []
    for name, pool in strata.items():
        for r in rng.sample(pool, min(k, len(pool))):
            rows.append({"stratum": name, "uid": r["uid"], "reference_preference": r["ref"], "text": r["text"],
                         "pseudo_label": r["pseudo"], "your_label": "", "notes": ""})
    rng.shuffle(rows)                     # mixed order so the reviewer is not anchored by strata
    for i, r in enumerate(rows):
        r["audit_id"] = i + 1
    with open(AUDIT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=AUDIT_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {AUDIT_CSV}.\nFill `your_label` with 1 (states a preference) or 0, using the "
          f"definition in docs/PHASE2.md. Ignore `pseudo_label` while judging if you can. Then run --audit-report.")


def audit_report():
    import json
    with open(AUDIT_CSV) as f:
        rows = [r for r in csv.DictReader(f) if r["your_label"].strip() in ("0", "1")]
    if not rows:
        sys.exit("No rows with your_label filled in yet.")

    def stats(rs):
        n = len(rs)
        a = [int(r["pseudo_label"]) for r in rs]
        b = [int(r["your_label"]) for r in rs]
        agree = sum(x == y for x, y in zip(a, b)) / n
        pa, pb = sum(a) / n, sum(b) / n
        pe = pa * pb + (1 - pa) * (1 - pb)
        kappa = (agree - pe) / (1 - pe) if pe < 1 else 1.0
        return {"n": n, "agreement": round(agree, 4), "cohen_kappa": round(kappa, 4),
                "pseudo1_human0": sum(x == 1 and y == 0 for x, y in zip(a, b)),
                "pseudo0_human1": sum(x == 0 and y == 1 for x, y in zip(a, b))}
    sure = [r for r in rows if "unsure" not in r.get("notes", "").lower()]
    rep = {"overall": stats(rows)}
    if len(sure) < len(rows):
        rep["overall_excluding_unsure"] = stats(sure)
    for s in sorted({r["stratum"] for r in rows}):
        rep[s] = stats([r for r in rows if r["stratum"] == s])
    rep["n_unsure"] = len(rows) - len(sure)
    with open(AUDIT_REPORT, "w") as f:
        json.dump(rep, f, indent=1)
    for k, v in rep.items():
        if not isinstance(v, dict):
            print(f"{k}: {v}")
            continue
        print(f"{k:12s} n={v['n']:3d} agreement={v['agreement']:.3f} kappa={v['cohen_kappa']:.3f} "
              f"(LLM yes/human no: {v['pseudo1_human0']}, LLM no/human yes: {v['pseudo0_human1']})")
    update_data_manifest("audit", {"csv_sha256": sha256_file(AUDIT_CSV), **rep["overall"]})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--what", choices=["persona", "oasst", "lmsys", "refine", "refine-persona-strict"])
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--audit-report", action="store_true")
    a = ap.parse_args()
    if a.what == "persona":
        label_persona()
    elif a.what == "oasst":
        label_yesno(sample_oasst(), OASST_CACHE, "oasst")
    elif a.what == "refine":
        refine()
    elif a.what == "refine-persona-strict":
        refine_persona_strict()
    elif a.what == "lmsys":
        label_yesno(read_jsonl(LMSYS_OOD), LMSYS_CACHE, "lmsys")
    elif a.audit:
        write_audit()
    elif a.audit_report:
        audit_report()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
