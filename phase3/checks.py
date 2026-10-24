"""Verification: our SimCSE rankings reproduce exactly the top-5 messages upstream's RAG handlers put in the prompt.

    PHASE3_RESULTS_DIR=results/phase3_smoke python -m phase3.checks      # after `phase3.retrieve --encoder simcse_paper`

Calls the vendored `handle_rag_task_mcq` / `handle_rag_task_implicit_mcq` with the prompt builder intercepted,
and compares their `top_k_sentences` with the top-5 ids stored by `phase3.retrieve`, for every question and length.
"""

import os
from types import SimpleNamespace

from phase1.runner import load_rag
from phase1.upstream import baselines_handling_classification as up
from phase1.upstream.common_utils import ALL_TOPICS
from phase3.common import SIMCSE, read_jsonl, retrieval_config
from phase3.conversations import CHOICE_KEYS, choice_data, llm_replies, lmsys_messages, mcq
from phase3.retrieve import out_path

TOPK = 5


def _capture(**kw):
    return kw["top_k_sentences"]


def main():
    originals = up.get_question_prompt_mcq_rag, up.get_implicit_question_prompt_rag_mcq
    up.get_question_prompt_mcq_rag = lambda *a, **kw: _capture(**kw)
    up.get_implicit_question_prompt_rag_mcq = lambda *a, **kw: _capture(**kw)
    try:
        check_simcse_top5()
    finally:   # restore: check_prompts needs the real builders
        up.get_question_prompt_mcq_rag, up.get_implicit_question_prompt_rag_mcq = originals


def check_simcse_top5():
    lm = lmsys_messages()
    checked = 0
    for n in retrieval_config()["inter_turns"]:
        for form, pool_name in (("explicit", "explicit@paper"), ("implicit-choice", "implicit-choice")):
            rows = {(r["topic"], r["task_id"]): r for r in read_jsonl(out_path(SIMCSE, pool_name, n))}
            for topic in ALL_TOPICS:
                if not any(t == topic for t, _ in rows):
                    continue
                rag_data, msg_idx, msg_rag = load_rag(topic, form)
                args = SimpleNamespace(inter_turns=n, topk=TOPK)
                for (t, tid), r in rows.items():
                    if t != topic:
                        continue
                    if form == "explicit":
                        pref, reply = mcq(topic)[tid]["preference"], llm_replies("llama3-8b", topic)[tid]
                        got = up.handle_rag_task_mcq(args, pref, reply, "q", "", "llama", 5, "", rag_data, msg_idx, tid, [])
                        text = {"pref": "User: " + pref, "reply": "Assistant: " + reply}
                        text.update({f"lmsys:{i}": msg_idx[str(i + 2)] for i in range(2 * n)})
                    else:
                        conv = choice_data(topic)[tid]["conversation"]
                        conv_list = [{"role": "user" if k in ("query", "user_selection") else "assistant",
                                      "content": conv[k]} for k in CHOICE_KEYS]
                        got = up.handle_rag_task_implicit_mcq(args, [], "q", "", "llama", 5, "", rag_data, msg_rag, tid,
                                                              conv_list, lm, [])
                        text = {f"dlg:{k}": c["role"] + ": " + c["content"] for k, c in zip(CHOICE_KEYS, conv_list)}
                        text.update({f"lmsys:{i}": lm[i]["role"] + ": " + lm[i]["content"] for i in range(2 * n)})
                    ours = [text[i] for i in r["top_ids"][:TOPK]]
                    assert ours == got, (form, n, topic, tid, r["top_ids"][:TOPK])
                    checked += 1
    print(f"OK: SimCSE top-{TOPK} identical to upstream's RAG selection for {checked} (question, length) pairs")


def check_prompts(n_per=3):
    """Phase 3 prompts reduce exactly to Phase 1's: `rag_gemma` fed SimCSE's top-5 == Phase 1 RAG prompt, and
    `memory` with an empty memory == Phase 1 zero-shot prompt (byte-identical, both families, both forms)."""
    from phase1 import runner as p1
    from phase1.runner import TopicContext, load_json, repo_settings, run_question, shuffle_options
    from phase3.memory import memory_config
    from phase3.prompts import build_prompt
    from phase3.runner import MODELS_YAML

    settings, models = repo_settings(), p1.load_yaml(MODELS_YAML)
    turns = load_json(os.path.join(p1.DATA, "filtered_inter_turns.json"))
    checked = 0
    for model in ("llama3-8b", "mistral7b"):
        cfg = {"model": model, "model_cfg": models[model], "rag_topk": TOPK, "mem": memory_config()}
        for form, pool_name in (("explicit", "explicit@paper"), ("implicit-choice", "implicit-choice")):
            for n in (3, 8):
                simcse = {(r["topic"], r["task_id"]): r["top_ids"] for r in read_jsonl(out_path(SIMCSE, pool_name, n))}
                # explicit: upstream decodes SimCSE indices with a per-topic msg_index file that, in 13 of 20 topics, is
                # shifted by one LMSYS exchange (see docs/PHASE3.md); compare on topics where it follows conversation order
                topics = ("travel_hotel", "pet_ownership") if form == "explicit" else ("travel_hotel", "shop_motors")
                for topic in topics:
                    for method_p1, method_p3 in (("rag", "rag_gemma"), ("zero-shot", "memory")):
                        ctx = TopicContext(cfg, topic, form, n, turns, need_rag=method_p1 == "rag")
                        ctx.reply_llm = model
                        for tid in range(n_per):
                            options, _ = shuffle_options(ctx.mcq[tid]["classification_task_options"], 41, topic, tid)
                            reply = llm_replies(model, topic)[tid] if form == "explicit" else None

                            class Capture:
                                calls, prompt = [], None

                                def generate(self, prompt, max_tokens, purpose="answer"):
                                    return reply

                                def upstream_generate_message(self, c, mid, mt, sp=None, messages=None, *a, **kw):
                                    Capture.prompt = messages
                                    return "A"

                            run_question(ctx, Capture(), method_p1, tid, options, settings)
                            ours = build_prompt(ctx, method_p3, tid, options, reply, [],
                                                simcse[(topic, tid)][:TOPK] if method_p1 == "rag" else None,
                                                settings["system_prompt"])
                            assert ours == Capture.prompt, (model, form, n, topic, tid, method_p3)
                            checked += 1
    print(f"OK: {checked} Phase 3 prompts byte-identical to Phase 1 (rag_gemma+SimCSE ids == rag, empty memory == zero-shot)")


if __name__ == "__main__":
    main()
    check_prompts()
