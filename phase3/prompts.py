"""Phase 3 prompts. Each method changes exactly ONE thing relative to its closest Phase 1 baseline:

  memory      = upstream zero-shot prompt, with the "Known user preferences" block before the question
  rag_gemma   = upstream RAG prompt (same builder and wording), top-5 messages retrieved by EmbeddingGemma-2 instead of SimCSE
  memory_rag  = rag_gemma prompt, with the memory block between the retrieved context and the question

The full conversation stays in every prompt (as in all baselines); the additions go into the final user turn, exactly
where upstream's RAG puts its retrieved context. Templates come from the vendored upstream builders (phase1/upstream).

Memory policy: include all entries if there are <= memory_topk, else the memory_topk entries most similar to the question
(EmbeddingGemma-2, an entry scores the max over its messages); selected entries keep conversation order.
"""

import functools
import os
import random

from phase1.upstream.implicit_utils import extract_implicit_messages
from phase1.upstream.mcq_implicit_utils import get_implicit_question_prompt_rag_mcq
from phase1.upstream.utils_mcq import get_implicit_question_prompt_mcq, get_question_prompt_mcq, get_question_prompt_mcq_rag
from phase3.common import MAIN_RESULTS_DIR, read_jsonl, sha256_text
from phase3.conversations import choice_data, mcq, pool, question_ids

METHODS = ["memory", "rag_gemma", "memory_rag"]
RAG_METHODS = {"rag_gemma", "memory_rag"}


# ----------------------------------------------------------------------------- embeddings / retrieval results
@functools.lru_cache(maxsize=None)
def vectors(encoder):
    from phase3.embed import load_cache
    return load_cache(encoder, "query"), load_cache(encoder, "doc")


@functools.lru_cache(maxsize=None)
def rag_top(encoder, form, reply_llm, n):
    """EmbeddingGemma-2 top-10 per question from the retrieval benchmark (always the committed results)."""
    pool_name = f"explicit@{reply_llm}" if form == "explicit" else form
    path = os.path.join(MAIN_RESULTS_DIR, "retrieval", encoder, pool_name, f"inter{n}.jsonl")
    rows = read_jsonl(path)
    assert rows, f"missing {path}: run the retrieval benchmark (phase3.embed / phase3.retrieve)"
    return {(r["topic"], r["task_id"]): r["top_ids"] for r in rows}


# ----------------------------------------------------------------------------- memory entries
def entries_from_memory(row, cands):
    """Memory row (phase3.memory) -> entries with their message texts."""
    by_id = {m["id"]: m for m in cands}
    return [{"msgs": [(by_id[i]["role"], by_id[i]["text"]) for i in e["ids"]], "ids": e["ids"], "turn": e["turn"],
             "gold": e["gold"], "origin": "conversation"} for e in row["entries"]]


def stress_memory(real, topic, task_id, form, d, seed):
    """Stress-test memory: the real entries plus d distractor preferences, in a seeded random order with no turn labels
    (so the true preference has no position or recency cue; a real memory would hold entries from many sessions)."""
    entries = [dict(e, turn=None) for e in real] + stress_distractors(topic, task_id, form, d, seed)
    random.Random(f"stress-order:{seed}:{form}:{topic}:{task_id}:{d}").shuffle(entries)
    return entries


def stress_distractors(topic, task_id, form, d, seed):
    """d preferences from questions of OTHER topics (as if stated in earlier sessions). Nested: the first 5 of d=10
    are the d=5 set, and so on, so larger d only adds entries."""
    others = [(t, i) for t, i in question_ids() if t != topic]
    picks = random.Random(f"stress:{seed}:{form}:{topic}:{task_id}").sample(others, 20)[:d]
    out = []
    for t, i in picks:
        if form == "explicit":
            msgs = [("user", mcq(t)[i]["preference"])]
        else:
            conv = choice_data(t)[i]["conversation"]
            msgs = [("assistant", conv["assistant_options"]), ("user", conv["user_selection"])]
        out.append({"msgs": msgs, "ids": [f"other:{t}:{i}"], "turn": None, "gold": False, "origin": "distractor"})
    return out


def select(entries, q_text, topk, policy, encoder):
    """policy: 'auto' (all if <= topk else top-k), 'all', 'top5' (top-k by similarity). Returns (selected, scores)."""
    if policy == "all" or (policy == "auto" and len(entries) <= topk):
        return list(entries)
    qv, dv = vectors(encoder)
    q = qv[sha256_text(q_text)]
    scores = [max(float(dv[sha256_text(t)] @ q) for _, t in e["msgs"]) for e in entries]
    keep = sorted(sorted(range(len(entries)), key=lambda i: -scores[i])[:topk])
    return [entries[i] for i in keep]


def memory_block(entries):
    if not entries:
        return ""
    lines = []
    for k, e in enumerate(entries, 1):
        where = f", turn {e['turn']}" if e.get("turn") else ""
        body = "\n".join(f"{role.capitalize()}: {text}" for role, text in e["msgs"])
        lines.append(f"[Preference {k}{where}]\n{body}")
    return ("Known user preferences: these are the preferences I have stated before.\n\n"
            "#Start of Known Preferences#\n" + "\n\n".join(lines) + "\n#End of Known Preferences#\n\n"
            "Please follow these preferences wherever they are relevant to my question.\n\n")


# ----------------------------------------------------------------------------- prompts
def rag_sentences(top_ids, cands, form):
    """Retrieved messages formatted exactly like upstream's RAG (explicit: 'User: '/'Assistant: ' for the
    preference exchange, 'user: '/'assistant: ' for LMSYS messages as in msg_index; implicit: lower-case roles)."""
    by_id = {m["id"]: m for m in cands}
    out = []
    for i in top_ids:
        m = by_id[i]
        if form == "explicit" and m["source"] in ("pref", "reply"):
            out.append(f"{m['role'].capitalize()}: {m['text']}")
        else:
            out.append(f"{m['role']}: {m['text']}")
    return out


def build_prompt(ctx, method, task_id, options, pref_generation, memory_entries, rag_ids, system_prompt):
    """ctx = phase1.runner.TopicContext (conversation, LMSYS turns, family). Returns the raw prompt string."""
    a, fam = ctx.args, ctx.family
    question = ctx.mcq[task_id]["question"]
    mem = memory_block(memory_entries) if method in ("memory", "memory_rag") else ""
    cands = pool(a.topic, task_id, ctx.form, a.inter_turns, ctx.reply_llm) if method in RAG_METHODS else None
    if ctx.form == "explicit":
        preference = ctx.mcq[task_id]["preference"]
        if method == "memory":
            return get_question_prompt_mcq(preference, options, pref_generation, mem + question, ctx.multi_inter_message,
                                           fam, a.inter_turns, remind=False, cot=False, system_prompt=system_prompt)
        return get_question_prompt_mcq_rag(preference, options, pref_generation, mem + question, ctx.multi_inter_message,
                                           fam, top_k_sentences=rag_sentences(rag_ids, cands, "explicit"),
                                           turn_number=a.inter_turns, system_prompt=system_prompt)
    conv_messages, _ =extract_implicit_messages(a, ctx.pref_data[task_id]["conversation"], fam)
    if method == "memory":
        return get_implicit_question_prompt_mcq(conv_messages, mem + question, ctx.multi_inter_message, fam,
                                                a.inter_turns, remind=False, cot=False, options=options)
    return get_implicit_question_prompt_rag_mcq(conv_messages, mem + question, ctx.multi_inter_message, fam,
                                                turn_number=a.inter_turns,
                                                top_k_sentences=rag_sentences(rag_ids, cands, "implicit-choice"),
                                                system_prompt=system_prompt, options=options)


def gold_position(entries):
    """1-based position of the true preference among the entries put in the prompt, or None."""
    return next((k for k, e in enumerate(entries, 1) if e["gold"]), None)