"""Phase 3 MCQ runs: preference memory (Phase 2 classifier) + EmbeddingGemma-2 retrieval, on the Phase 1 grid.

    python -m phase3.runner run --config configs/phase3/tierA.yaml           # resumable (one fsync'd line per question)
    python -m phase3.runner run --config configs/phase3/tierA.yaml --dry-run # cells and progress only
    python -m phase3.runner status --config configs/phase3/tierA.yaml        # progress, accuracy, ETA (read-only)
    python -m phase3.runner show-prompt --model llama3-8b --form implicit-choice --method memory_rag --inter-turns 3

Prerequisites: `python -m phase3.memory` (memories) and the retrieval benchmark (EmbeddingGemma-2 rankings).
Reuses Phase 1 unchanged: Ollama backend (raw mode, temperature 0, seed 41), option shuffle, strict `extract_choice`,
resume/repair, and the Ollama server-settings guard (each model runs under the settings of its own baselines).
Explicit form: the LLM's reply to the stated preference is taken from Phase 1 (deterministic; identical in every Phase 1
method and length), which saves one call per question. `--verify-reply N` regenerates it for the first N questions per
cell and stops on any mismatch.

Results: results/phase3/mcq/<model>/<form>/<variant>/inter<N>/<topic>.jsonl, variant = method, or for the stress test
memory_stress_d<D>_<policy>.
"""

import argparse
import datetime as dt
import logging
import os
import time

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from phase1.backend import BACKEND_REV, OllamaBackend, OllamaUnavailable
from phase1.runner import (DATA, MODELS_YAML, TopicContext, append_record, check_ollama_env, lenient_choice, load_json,
                           ollama_server_env, read_done, repo_settings, shuffle_options)
from phase1.upstream.common_utils import ALL_TOPICS
from phase1.upstream.explicit_utils import create_user_pref_message
from phase1.upstream.utils_mcq import extract_choice
from phase3.common import LOGS_DIR, RESULTS_DIR, load_yaml, sha256_text, write_run_manifest
from phase3.conversations import FORMS, llm_replies, mcq, pool
from phase3.memory import load_memory, memory_config
from phase3.prompts import (METHODS, RAG_METHODS, build_prompt, entries_from_memory, gold_position, rag_top, select,
                            stress_memory)

MCQ_DIR = os.path.join(RESULTS_DIR, "mcq")
CONSECUTIVE_FAILURE_LIMIT = 10
log = logging.getLogger("phase3.runner")


# ----------------------------------------------------------------------------- config / cells
def load_tier(path):
    cfg = load_yaml(path)
    cfg.setdefault("name", os.path.splitext(os.path.basename(path))[0])
    if cfg.get("topics") in (None, "all"):
        cfg["topics"] = list(ALL_TOPICS)
    models = load_yaml(MODELS_YAML)
    cfg["model_cfg"] = dict(models[cfg["model"]])
    if cfg.get("num_ctx"):
        cfg["model_cfg"]["num_ctx"] = cfg["num_ctx"]
    cfg["mem"] = memory_config()
    cfg["rag_topk"] = cfg["mem"]["rag_topk"]
    if cfg.get("stress") is True:      # stress grid (distractors x policies) lives in memory.yaml
        cfg["stress"] = cfg["mem"]["stress"]
    assert set(cfg["forms"]) <= set(FORMS) and set(cfg["methods"]) <= set(METHODS), cfg
    return cfg


def iter_cells(cfg):
    """topic -> form -> inter_turns -> variant. RAG methods need distractor turns (none at inter_turns 0, as Phase 1)."""
    stress = cfg.get("stress")
    for topic in cfg["topics"]:
        for form in cfg["forms"]:
            for n in cfg["inter_turns"]:
                if stress:
                    for d in stress["distractors"]:
                        for pol in stress["policies"]:
                            yield {"topic": topic, "form": form, "inter_turns": n, "method": "memory",
                                   "variant": f"memory_stress_d{d}_{pol}", "stress_d": d, "policy": pol}
                    continue
                for m in cfg["methods"]:
                    if m in RAG_METHODS and n == 0:
                        continue
                    yield {"topic": topic, "form": form, "inter_turns": n, "method": m, "variant": m,
                           "stress_d": 0, "policy": "auto"}


def cell_path(model, c):
    return os.path.join(MCQ_DIR, model, c["form"], c["variant"], f"inter{c['inter_turns']}", f"{c['topic']}.jsonl")


def n_questions(topic, limit):
    n = len(mcq(topic))
    return min(n, limit) if limit else n


# ----------------------------------------------------------------------------- one question
class Cell:
    """Everything one (topic, form, inter_turns, variant) needs: upstream context, memories, retrieval results."""

    def __init__(self, cfg, c, turns_data):
        self.c, self.cfg = c, cfg
        self.ctx = TopicContext(cfg, c["topic"], c["form"], c["inter_turns"], turns_data, need_rag=False)
        self.ctx.reply_llm = cfg["model"]
        self.memory = load_memory(c["form"], c["inter_turns"])
        assert self.memory, f"no memories for {c['form']} inter{c['inter_turns']}: run `python -m phase3.memory`"
        self.rag = rag_top(cfg["mem"]["retriever"], c["form"], cfg["model"], c["inter_turns"]) \
            if c["method"] in RAG_METHODS else None
        self.replies = llm_replies(cfg["model"], c["topic"]) if c["form"] == "explicit" else None


def run_question(cell, backend, task_id, options, settings, verify_reply=False):
    c, cfg, ctx, mem_cfg = cell.c, cell.cfg, cell.ctx, cell.cfg["mem"]
    topic, form, n = c["topic"], c["form"], c["inter_turns"]
    sp, max_tokens, fam = settings["system_prompt"], settings["max_mcq_tokens"], ctx.family
    out = {}
    pref_generation = None
    if form == "explicit":
        pref_generation = cell.replies[task_id]
        out["response_to_pref"] = pref_generation
        if verify_reply:
            fresh = backend.generate(create_user_pref_message(ctx.mcq[task_id]["preference"], fam, sp), max_tokens,
                                     "pref_response")
            if fresh != pref_generation:
                raise RuntimeError(f"Phase 1 reply mismatch {topic}/{task_id}: {fresh!r} vs {pref_generation!r}")
    # memory: classifier-built entries (+ stress distractors), then the selection policy
    row = cell.memory[(topic, task_id)]
    cands = pool(topic, task_id, form, n, cfg["model"])
    entries = entries_from_memory(row, cands)
    if c["stress_d"]:
        entries = stress_memory(entries, topic, task_id, form, c["stress_d"], mem_cfg["seed"])
    used = select(entries, ctx.mcq[task_id]["question"], mem_cfg["memory_topk"], c["policy"], mem_cfg["retriever"])
    rag_ids = cell.rag[(topic, task_id)][:mem_cfg["rag_topk"]] if cell.rag is not None else None
    prompt = build_prompt(ctx, c["method"], task_id, options, pref_generation, used, rag_ids, sp)
    end = backend.upstream_generate_message(None, None, fam, sp, prompt, max_tokens)
    if fam == "mistral":  # upstream prefills "<choice>" for mistral and re-attaches it (as Phase 1)
        end = "<choice>" + end
    out.update(response_to_q=end, prompt_sha=sha256_text(prompt)[:16],
               memory_size=len(entries), memory_used=len(used), gold_in_memory=row["gold_in_memory"],
               gold_position=gold_position(used) if c["method"] != "rag_gemma" else None,
               memory_ids=[e["ids"] for e in used] if c["method"] != "rag_gemma" else None,
               rag_ids=rag_ids, gold_in_rag=("pref" in rag_ids or "dlg:user_selection" in rag_ids) if rag_ids else None)
    return out


# ----------------------------------------------------------------------------- commands
def setup_logging(name):
    os.makedirs(LOGS_DIR, exist_ok=True)
    path = os.path.join(LOGS_DIR, f"{name}_{dt.datetime.now():%Y%m%d_%H%M%S}.log")
    fh = logging.FileHandler(path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%H:%M:%S"))
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers = [fh, ch]
    for noisy in ("urllib3", "httpx", "httpx2"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return path


def cmd_run(o):
    cfg = load_tier(o.config)
    limit = o.limit if o.limit is not None else cfg.get("limit")
    if (limit or o.topics) and not os.environ.get("PHASE3_RESULTS_DIR"):
        raise SystemExit("partial runs (--limit/--topics) must set PHASE3_RESULTS_DIR=results/phase3_smoke")
    if o.topics:
        cfg["topics"] = o.topics.split(",")
    settings = repo_settings()
    log_path = setup_logging(cfg["name"])
    mc = cfg["model_cfg"]
    backend = OllamaBackend(mc["ollama_tag"], mc["num_ctx"], host=o.host, seed=cfg["mem"]["seed"])
    cells = list(iter_cells(cfg))
    todo = sum(n_questions(c["topic"], limit) - len(read_done(cell_path(cfg["model"], c))) for c in cells)
    log.info("Config %s | model %s (%s, num_ctx %d) | %d cells | %d questions remaining | log: %s",
             o.config, cfg["model"], mc["ollama_tag"], mc["num_ctx"], len(cells), todo, log_path)
    env_error = check_ollama_env(cfg["model"], mc)
    if env_error:
        log.error(env_error)
        if not o.dry_run:
            return 3
    if o.dry_run:
        for c in cells:
            print(f"{c['topic']:34s} {c['form']:16s} inter{c['inter_turns']:<3d} {c['variant']:24s} "
                  f"{len(read_done(cell_path(cfg['model'], c)))}/{n_questions(c['topic'], limit)}")
        return 0
    try:
        start = {"step": "mcq_run", "event": "start", "config_file": o.config,
                 "config": {k: v for k, v in cfg.items() if k != "topics"}, "topics": cfg["topics"], "settings": settings,
                 "backend_rev": BACKEND_REV, "ollama_version": backend.server_version(),
                 "ollama_env": ollama_server_env(), "model": backend.model_info(),
                 "log": os.path.relpath(log_path, os.path.dirname(MCQ_DIR))}
    except Exception as e:
        log.error("Cannot reach Ollama at %s (%s). Start it with `systemctl start ollama`.", o.host, e)
        return 2
    write_run_manifest(start)
    turns_data = load_json(os.path.join(DATA, "filtered_inter_turns.json"))
    status, fails, n_run, t_start = "completed", 0, 0, time.time()
    try:
        with logging_redirect_tqdm():
            for ci, c in enumerate(cells, 1):
                path = cell_path(cfg["model"], c)
                done = read_done(path, repair=True)
                total = n_questions(c["topic"], limit)
                remaining = [i for i in range(total) if i not in done]
                tag = f"[{ci}/{len(cells)}] {c['topic']} | {c['form']} | inter{c['inter_turns']} | {c['variant']}"
                if not remaining:
                    continue
                log.info("%s: %d/%d done, running %d", tag, len(done), total, len(remaining))
                cell = Cell(cfg, c, turns_data)
                correct = sum(r["correct"] for r in done.values())
                for task_id in tqdm(remaining, desc=f"{c['variant']}/inter{c['inter_turns']}", leave=False):
                    options, letter = shuffle_options(cell.ctx.mcq[task_id]["classification_task_options"],
                                                      cfg["mem"]["seed"], c["topic"], task_id)
                    backend.calls = []
                    t0 = time.time()
                    try:
                        out = run_question(cell, backend, task_id, options, settings,
                                           verify_reply=task_id < o.verify_reply)
                    except OllamaUnavailable:
                        raise
                    except Exception as e:
                        if "reply mismatch" in str(e):
                            raise
                        fails += 1
                        log.exception("%s task %d failed (%d consecutive): %s", tag, task_id, fails, e)
                        if fails >= CONSECUTIVE_FAILURE_LIMIT:
                            raise RuntimeError("too many consecutive failures; aborting") from e
                        continue
                    fails = 0
                    choice = extract_choice(out["response_to_q"])
                    rec = {"task_id": task_id, "topic": c["topic"], "form": c["form"], "method": c["method"],
                           "variant": c["variant"], "stress_d": c["stress_d"], "policy": c["policy"],
                           "inter_turns": c["inter_turns"], "model": cfg["model"], "backend_rev": BACKEND_REV,
                           "shuffled_options": options, "correct_idx": letter, **out,
                           "choice": choice, "choice_lenient": lenient_choice(out["response_to_q"]),
                           "correct": choice == letter, "seconds": round(time.time() - t0, 3), "calls": backend.calls,
                           "truncated": any(x["truncated"] for x in backend.calls),
                           "timestamp": dt.datetime.now().isoformat(timespec="seconds")}
                    append_record(path, rec)
                    correct += rec["correct"]
                    n_run += 1
                log.info("%s finished: accuracy %.1f%% (%d/%d)", tag, 100 * correct / total, correct, total)
    except KeyboardInterrupt:
        status = "interrupted"
        log.warning("Interrupted. Everything up to the last finished question is saved; rerun the same command.")
    except (OllamaUnavailable, RuntimeError) as e:
        status = "aborted"
        log.error("Aborted: %s. Rerun the same command to resume.", e)
    except Exception:
        status = "crashed"
        log.exception("Crashed. Finished questions are saved; fix the cause and rerun to resume.")
        raise
    finally:
        write_run_manifest({"step": "mcq_run", "event": "end", "config_file": o.config, "status": status,
                            "questions_this_run": n_run, "elapsed_s": round(time.time() - t_start, 1)})
    log.info("Run %s: %d questions in %.1f min. Next: python -m phase3.aggregate", status, n_run,
             (time.time() - t_start) / 60)
    return {"completed": 0, "interrupted": 130}.get(status, 1)


def cmd_status(o):
    cfg = load_tier(o.config)
    limit = cfg.get("limit")
    done_q = total_q = 0
    secs = []
    for c in iter_cells(cfg):
        recs = read_done(cell_path(cfg["model"], c))
        total = n_questions(c["topic"], limit)
        done_q, total_q = done_q + len(recs), total_q + total
        secs += [r["seconds"] for r in recs.values()]
        acc = f"{100 * sum(r['correct'] for r in recs.values()) / len(recs):5.1f}%" if recs else "    -"
        print(f"{c['topic']:34s} {c['form']:16s} inter{c['inter_turns']:<3d} {c['variant']:24s} "
              f"{len(recs):4d}/{total:<4d} acc {acc} {'done' if len(recs) == total else ''}")
    per_q = sum(secs) / len(secs) if secs else float("nan")
    print(f"\n{done_q}/{total_q} questions done ({100 * done_q / max(total_q, 1):.1f}%). "
          f"Mean {per_q:.2f} s/question -> about {(total_q - done_q) * per_q / 3600:.1f} h remaining")
    return 0


def cmd_show_prompt(o):
    models = load_yaml(MODELS_YAML)
    cfg = {"model": o.model, "model_cfg": models[o.model], "mem": memory_config(), "rag_topk": 5}
    c = {"topic": o.topic, "form": o.form, "inter_turns": o.inter_turns, "method": o.method, "variant": o.method,
         "stress_d": o.stress_d, "policy": o.policy}
    cell = Cell(cfg, c, load_json(os.path.join(DATA, "filtered_inter_turns.json")))
    options, letter = shuffle_options(cell.ctx.mcq[o.task_id]["classification_task_options"], 41, o.topic, o.task_id)

    class Capture:
        calls = []

        def generate(self, prompt, max_tokens, purpose="answer"):
            return ""

        def upstream_generate_message(self, cl, mid, mt, sp=None, messages=None, max_tokens=None, **kw):
            print(messages)
            return "<choice>A</choice>"

    out = run_question(cell, Capture(), o.task_id, options, repo_settings())
    print(f"\n(correct answer: {letter}) {({k: v for k, v in out.items() if k != 'response_to_q'})}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", required=True)
    r.add_argument("--limit", type=int, default=None, help="max questions per cell (smoke; needs PHASE3_RESULTS_DIR)")
    r.add_argument("--topics", help="comma-separated topic subset (smoke; needs PHASE3_RESULTS_DIR)")
    r.add_argument("--verify-reply", type=int, default=0, help="regenerate the explicit reply for the first N questions")
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--host", default=os.environ.get("OLLAMA_HOST_URL", "http://localhost:11434"))
    s = sub.add_parser("status")
    s.add_argument("--config", required=True)
    sp = sub.add_parser("show-prompt")
    sp.add_argument("--model", default="llama3-8b")
    sp.add_argument("--form", default="explicit", choices=FORMS)
    sp.add_argument("--method", default="memory_rag", choices=METHODS)
    sp.add_argument("--inter-turns", type=int, default=3)
    sp.add_argument("--topic", default="travel_restaurant", choices=ALL_TOPICS)
    sp.add_argument("--task-id", type=int, default=0)
    sp.add_argument("--stress-d", type=int, default=0)
    sp.add_argument("--policy", default="auto", choices=["auto", "all", "top5"])
    o = p.parse_args(argv)
    return {"run": cmd_run, "status": cmd_status, "show-prompt": cmd_show_prompt}[o.cmd](o)


if __name__ == "__main__":
    raise SystemExit(main())
