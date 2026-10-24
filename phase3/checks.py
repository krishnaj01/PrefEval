"""Verification: our SimCSE rankings reproduce exactly the top-5 messages upstream's RAG handlers put in the prompt.

    PHASE3_RESULTS_DIR=results/phase3_smoke python -m phase3.checks      # after `phase3.retrieve --encoder simcse_paper`

Calls the vendored `handle_rag_task_mcq` / `handle_rag_task_implicit_mcq` with the prompt builder intercepted,
and compares their `top_k_sentences` with the top-5 ids stored by `phase3.retrieve`, for every question and length.
"""

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
    up.get_question_prompt_mcq_rag = lambda *a, **kw: _capture(**kw)
    up.get_implicit_question_prompt_rag_mcq = lambda *a, **kw: _capture(**kw)
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


if __name__ == "__main__":
    main()
