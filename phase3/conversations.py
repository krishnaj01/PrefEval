"""Per-question candidate pools for the retrieval benchmark, rebuilt exactly as Phase 1 (upstream) lays out the conversation.

A *pool* is every message that comes before the final question, in conversation order:
  explicit        : [preference (user), the LLM's own reply to it (assistant)] + the first 2N LMSYS messages
  implicit-choice : [query, assistant_options, user_selection, assistant_acknowledgment] + the first 2N LMSYS messages
The *gold* message is the one that states the preference: the explicit preference sentence, or the choice `user_selection`.
The query is the final question (`mcq_options/<topic>.json`, as in Phase 1).

Pool order equals the index order of the stored SimCSE scores (explicit: 0 = pref, 1 = reply, 2.. = LMSYS;
implicit: 4 dialogue scores, then `inter_sentence_scores`), so `simcse_scores` lines up with `pool` one-to-one.
Only message *content* is embedded (no "user:" prefix); the role is kept as metadata.

    python -m phase3.conversations          # prints pool statistics and runs the assertions
"""

import functools
import json
import os

from phase1.runner import rag_available_count, rag_paths, rag_unavailable_reason
from phase1.upstream.common_utils import ALL_TOPICS
from phase3.common import DATASET_DIR, PHASE1_RESULTS, read_jsonl

FORMS = ["explicit", "implicit-choice"]
CHOICE_KEYS = ["query", "assistant_options", "user_selection", "assistant_acknowledgment"]
CHOICE_ROLES = {"query": "user", "assistant_options": "assistant", "user_selection": "user",
                "assistant_acknowledgment": "assistant"}
GOLD_CHOICE_KEY = "user_selection"


def _load(path):
    with open(path) as f:
        return json.load(f)


@functools.lru_cache(maxsize=None)
def lmsys_messages():
    """The 632 LMSYS distractor messages in the order upstream concatenates them (`extract_multi_turn_message`)."""
    turns = _load(os.path.join(DATASET_DIR, "filtered_inter_turns.json"))
    return [m for t in turns for m in t["conversation"]]


@functools.lru_cache(maxsize=None)
def mcq(topic):
    return _load(os.path.join(DATASET_DIR, "mcq_options", f"{topic}.json"))


@functools.lru_cache(maxsize=None)
def choice_data(topic):
    data = _load(os.path.join(DATASET_DIR, "implicit_preference", "choice-based", f"{topic}.json"))
    assert len(data) == len(mcq(topic)), topic
    return data


@functools.lru_cache(maxsize=None)
def llm_replies(llm, topic):
    """task_id -> the LLM's reply to the stated preference, exactly as inserted in its Phase 1 prompts.
    The reply is identical across methods and context lengths (decoding is deterministic), so inter0/zero-shot is used."""
    rows = read_jsonl(os.path.join(PHASE1_RESULTS, llm, "explicit", "zero-shot", "inter0", f"{topic}.jsonl"))
    out = {r["task_id"]: r["response_to_pref"] for r in rows if r.get("backend_rev") == 2}
    assert len(out) == len(mcq(topic)), (llm, topic, len(out))
    return out


@functools.lru_cache(maxsize=None)
def simcse_data(topic, form):
    """Stored SimCSE scores for (topic, form), or None when upstream's file is broken (explicit entertain_games)."""
    if rag_unavailable_reason(topic, form):
        return None
    p = rag_paths(topic, form)
    if form == "explicit":
        return {"pref": _load(p["rag_data"])}
    return {"pref": _load(p["rag_data"]), "inter": _load(p["msg_rag_data"])}


def simcse_available(topic, form, task_id):
    return simcse_data(topic, form) is not None and task_id < rag_available_count(topic, form)


def question_ids():
    """All (topic, task_id) pairs, topic order = ALL_TOPICS (1,000 questions)."""
    return [(t, i) for t in ALL_TOPICS for i in range(len(mcq(t)))]


def query(topic, task_id):
    return mcq(topic)[task_id]["question"]


def _lmsys_part(n_inter):
    msgs = lmsys_messages()
    assert 2 * n_inter <= len(msgs)
    return [{"id": f"lmsys:{i}", "role": m["role"], "source": "lmsys", "text": m["content"], "gold": False}
            for i, m in enumerate(msgs[:2 * n_inter])]


def pool(topic, task_id, form, n_inter, reply_llm=None):
    """Candidate messages before the question (see module docstring). `reply_llm` is required for explicit."""
    if form == "explicit":
        assert reply_llm, "explicit pools contain the LLM's own reply: pass reply_llm"
        head = [{"id": "pref", "role": "user", "source": "pref", "text": mcq(topic)[task_id]["preference"], "gold": True},
                {"id": "reply", "role": "assistant", "source": "reply", "text": llm_replies(reply_llm, topic)[task_id],
                 "gold": False}]
    elif form == "implicit-choice":
        conv = choice_data(topic)[task_id]["conversation"]
        assert list(conv) == CHOICE_KEYS, (topic, task_id, list(conv))
        head = [{"id": f"dlg:{k}", "role": CHOICE_ROLES[k], "source": "dialogue", "text": conv[k],
                 "gold": k == GOLD_CHOICE_KEY} for k in CHOICE_KEYS]
    else:
        raise ValueError(form)
    out = head + _lmsys_part(n_inter)
    assert len(out) == len(head) + 2 * n_inter and sum(m["gold"] for m in out) == 1
    return out


def simcse_scores(topic, task_id, form, n_inter):
    """SimCSE similarity per pool position (same order as `pool`), from upstream's stored files; None if unavailable.
    Note: the explicit reply score was computed by the authors on *their* reply text, not on our LLM's reply."""
    if not simcse_available(topic, form, task_id):
        return None
    d = simcse_data(topic, form)
    if form == "explicit":
        ss = d["pref"][task_id]["sentence_scores"][:2 + 2 * n_inter]
        assert [s[0] for s in ss] == list(range(len(ss)))
        return [s[1] for s in ss]
    head = d["pref"][task_id]["sentence_scores"]
    inter = d["inter"][task_id]["inter_sentence_scores"][:2 * n_inter]
    assert len(head) == 4 and [s[0] for s in inter] == list(range(len(inter)))
    return [s[1] for s in head] + [s[1] for s in inter]


def all_texts(reply_llms, max_inter):
    """Every unique text an encoder must embed: (queries, documents)."""
    queries, docs = set(), set()
    for topic, tid in question_ids():
        queries.add(query(topic, tid))
        for llm in reply_llms:
            docs.update(m["text"] for m in pool(topic, tid, "explicit", 0, llm))
        docs.update(m["text"] for m in pool(topic, tid, "implicit-choice", 0))
    docs.update(m["content"] for m in lmsys_messages()[:2 * max_inter])
    return sorted(queries), sorted(docs)


def main():
    from phase3.common import retrieval_config
    cfg = retrieval_config()
    qs = question_ids()
    for form in FORMS:
        avail = sum(simcse_available(t, form, i) for t, i in qs)
        for n in cfg["inter_turns"]:
            for t, i in qs:
                p = pool(t, i, form, n, cfg["reply_llms"][0])
                s = simcse_scores(t, i, form, n)
                assert s is None or len(s) == len(p)
        print(f"{form:16s} questions={len(qs)}  with SimCSE scores={avail}")
    q, d = all_texts(cfg["reply_llms"], max(cfg["inter_turns"]))
    print(f"unique texts: {len(q)} queries, {len(d)} documents")


if __name__ == "__main__":
    main()
