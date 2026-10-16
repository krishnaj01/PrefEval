"""Step 1: build the turn-level table from PrefEval (deterministic, no model calls).

    python -m phase2.build_data

Output
  data/phase2/prefeval_turns.jsonl : one row per user message. Gold labels for explicit preferences,
                                     persona `preference` paraphrases, choice selections (1) and
                                     questions / initial queries (0). Persona dialogue turns have label=null
                                     and are labelled by phase2.pseudolabel.
  data/phase2/lmsys_ood.jsonl      : the 316 LMSYS distractor user turns, kept for EVALUATION ONLY
                                     (Phase 3 replays them in every question, so they must never be trained on).

Row schema: uid, text, label, label_origin, source, form, topic, task_id, turn_idx, group.
Only `text` is ever given to a model; every other field is bookkeeping (splitting / analysis).
"""

import collections
import json
import os
import re

from phase2.common import DATASET_DIR, LMSYS_OOD, PREFEVAL_TURNS, update_data_manifest, sha256_file, write_jsonl
from phase1.upstream.common_utils import ALL_TOPICS


def norm(text):
    """Normalisation used only for duplicate detection (the stored text is unchanged)."""
    return re.sub(r"\s+", " ", text).strip().lower()


def load(rel):
    with open(os.path.join(DATASET_DIR, rel)) as f:
        return json.load(f)


def row(topic, task_id, source, text, label, form, turn_idx=None):
    uid = f"prefeval/{topic}/{task_id}/{source}" + (f"/{turn_idx}" if turn_idx is not None else "")
    return {"uid": uid, "text": text.strip(), "label": label, "label_origin": "gold" if label is not None else None,
            "source": source, "form": form, "topic": topic, "task_id": task_id, "turn_idx": turn_idx,
            "group": topic}


def build_prefeval_rows():
    rows = []
    for topic in ALL_TOPICS:   # fixed topic order = Phase 1 order; the duplicate "travel_hotel copy.json" is never read
        explicit = load(f"explicit_preference/{topic}.json")
        choice = load(f"implicit_preference/choice-based/{topic}.json")
        persona = load(f"implicit_preference/persona-driven/{topic}.json")
        assert len(explicit) == len(choice) == len(persona), topic
        for i, (e, c, p) in enumerate(zip(explicit, choice, persona)):
            rows.append(row(topic, i, "explicit_pref", e["preference"], 1, "explicit"))
            rows.append(row(topic, i, "question", e["question"], 0, "none"))
            rows.append(row(topic, i, "persona_pref", p["preference"], 1, "explicit"))
            conv = c["conversation"]
            rows.append(row(topic, i, "choice_query", conv["query"], 0, "none"))
            rows.append(row(topic, i, "choice_selection", conv["user_selection"], 1, "implicit-choice"))
            for t, turn in sorted(p["conversation"].items(), key=lambda kv: int(kv[0])):
                rows.append(row(topic, i, "persona_turn", turn["user"], None, None, int(t)))
    return rows


def dedupe(rows):
    """Drop exact (normalised) duplicate texts. The first occurrence wins; conflicting gold labels drop both."""
    first, conflicts, dropped = {}, set(), 0
    for r in rows:
        k = norm(r["text"])
        if k not in first:
            first[k] = r
            continue
        dropped += 1
        a = first[k]
        if a["label"] is not None and r["label"] is not None and a["label"] != r["label"]:
            conflicts.add(k)
    kept = [r for k, r in first.items() if k not in conflicts]
    return kept, dropped, len(conflicts)


def build_lmsys_rows():
    sessions = load("filtered_inter_turns.json")
    rows = []
    for s_idx, s in enumerate(sessions):
        u = 0
        for m in s["conversation"]:
            if m["role"] != "user" or not m["content"].strip():
                continue
            rows.append({"uid": f"lmsys/{s['conversation_id']}/{u}", "text": m["content"].strip(), "label": None,
                         "label_origin": None, "source": "lmsys_ood", "form": None, "topic": None,
                         "task_id": s_idx, "turn_idx": u, "group": "lmsys_ood"})
            u += 1
    return rows


def main():
    rows, dropped, conflicts = dedupe(build_prefeval_rows())
    write_jsonl(PREFEVAL_TURNS, rows)
    lmsys = build_lmsys_rows()
    write_jsonl(LMSYS_OOD, lmsys)

    by_src = collections.Counter((r["source"], r["label"]) for r in rows)
    print(f"PrefEval turns: {len(rows)} (dropped {dropped} duplicates, {conflicts} conflicting texts removed)")
    for (src, lab), n in sorted(by_src.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        print(f"  {src:18s} label={str(lab):5s} {n:5d}")
    print(f"LMSYS OOD turns (eval only): {len(lmsys)}")
    update_data_manifest("build_data", {
        "prefeval_turns": {"rows": len(rows), "sha256": sha256_file(PREFEVAL_TURNS),
                           "duplicates_dropped": dropped, "conflicts_removed": conflicts},
        "lmsys_ood": {"rows": len(lmsys), "sha256": sha256_file(LMSYS_OOD)},
        "source_counts": {f"{s}|{l}": n for (s, l), n in by_src.items()},
    })


if __name__ == "__main__":
    main()
