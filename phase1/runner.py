"""Phase 1 runner: PrefEval classification (MCQ) baselines on a local Ollama model.

Usage (from the repo root, inside the `prefeval` conda env):
    python -m phase1.runner run         --config configs/phase1/smoke.yaml
    python -m phase1.runner run         --config configs/phase1/tierA_pilot.yaml --dry-run
    python -m phase1.runner status      --config configs/phase1/tierA_pilot.yaml
    python -m phase1.runner show-prompt --model llama3-8b --form explicit --method remind --inter-turns 3

Every question's result is appended (and fsync'd) to
    results/phase1/<model>/<form>/<method>/inter<N>/<topic>.jsonl
Re-running the same command resumes: questions already in the file are skipped.
"""

import argparse
import datetime as dt
import json
import logging
import os
import random
import re
import subprocess
import sys
import time
from types import SimpleNamespace

import yaml
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from phase1.backend import BACKEND_REV, OllamaBackend, OllamaUnavailable
from phase1.upstream import baselines_handling_classification as upstream_baselines
from phase1.upstream.common_utils import ALL_TOPICS, extract_multi_turn_message
from phase1.upstream.explicit_utils import create_user_pref_message
from phase1.upstream.implicit_utils import extract_implicit_messages
from phase1.upstream.utils_mcq import extract_choice, get_implicit_question_prompt_mcq, get_question_prompt_mcq

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "benchmark_dataset")
RESULTS = os.path.join(REPO, "results", "phase1")
LOGS = os.path.join(REPO, "logs", "phase1")
MODELS_YAML = os.path.join(REPO, "configs", "phase1", "models.yaml")

FORMS = ["explicit", "implicit-choice", "implicit-persona"]
METHODS = ["zero-shot", "remind", "cot", "selfcritic", "rag"]
CONSECUTIVE_FAILURE_LIMIT = 10
# Seconds/question used for ETA before a method has measurements (smoke test, RTX 3060, Llama3-8B Q8,
# averaged over inter_turns 0 and 8).
DEFAULT_SEC_PER_Q = {"zero-shot": 0.8, "remind": 0.6, "cot": 0.9, "rag": 1.9, "selfcritic": 4.7}

log = logging.getLogger("phase1")


# =============================================================================== config / data
def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_experiment(config_path):
    cfg = load_yaml(config_path)
    cfg.setdefault("name", os.path.splitext(os.path.basename(config_path))[0])
    cfg.setdefault("seed", 41)
    cfg.setdefault("rag_topk", 5)
    cfg.setdefault("limit", None)
    if cfg.get("topics") in (None, "all"):
        cfg["topics"] = list(ALL_TOPICS)
    for key, allowed in (("forms", FORMS), ("methods", METHODS), ("topics", ALL_TOPICS)):
        bad = [x for x in cfg[key] if x not in allowed]
        if bad:
            raise ValueError(f"{config_path}: unknown {key} {bad}; allowed: {allowed}")
    models = load_yaml(MODELS_YAML)
    if cfg["model"] not in models:
        raise ValueError(f"model '{cfg['model']}' not in {MODELS_YAML}")
    cfg["model_cfg"] = models[cfg["model"]]
    # Upstream picks prompt templates via substring checks on the model name ("llama" in args.model).
    assert cfg["model_cfg"]["family"] in cfg["model"], "model key must contain its family name (llama/mistral)"
    return cfg


def repo_settings():
    """system_prompt / max_mcq_tokens from the upstream config.yaml (unchanged from the paper)."""
    return load_yaml(os.path.join(REPO, "config.yaml"))


def iter_cells(cfg):
    """Experiment cells in run order: topic -> form -> inter_turns -> method.

    Topic is outermost so that an interrupted run still leaves complete method x length
    grids for the first topics (most useful for a staged pilot).
    """
    for topic in cfg["topics"]:
        for form in cfg["forms"]:
            for inter in cfg["inter_turns"]:
                for method in cfg["methods"]:
                    if method == "rag" and inter == 0:
                        # Upstream asserts >= topk retrievable exchanges, which fails without distractor
                        # turns (explicit, implicit-choice); paper reports no RAG value at 0.2k either.
                        continue
                    yield {"topic": topic, "form": form, "inter_turns": inter, "method": method}


def cell_path(model, cell):
    return os.path.join(RESULTS, model, cell["form"], cell["method"], f"inter{cell['inter_turns']}",
                        f"{cell['topic']}.jsonl")


def n_questions(topic, limit=None):
    n = len(load_json(os.path.join(DATA, "mcq_options", f"{topic}.json")))
    return min(n, limit) if limit else n


# =============================================================================== results I/O
def read_done(path, repair=False):
    """Return {task_id: record} from a results file.

    Unreadable lines (torn write from a crash) and records from another BACKEND_REV are ignored.
    With repair=True (only used by `run`, never by `status`, which may run concurrently with a job)
    the file is rewritten without them.
    """
    if not os.path.exists(path):
        return {}
    records, bad = {}, 0
    with open(path) as f:
        lines = f.readlines()
    for line in lines:
        try:
            rec = json.loads(line)
            if rec.get("backend_rev") != BACKEND_REV:
                raise KeyError("stale backend_rev")
            records[rec["task_id"]] = rec
        except (json.JSONDecodeError, KeyError):
            bad += 1
    if bad and repair:
        log.warning("%s: dropped %d unreadable/stale line(s) (interrupted write or old backend_rev); rewriting file", path, bad)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            for rec in sorted(records.values(), key=lambda r: r["task_id"]):
                f.write(json.dumps(rec) + "\n")
        os.replace(tmp, path)
    return records


def append_record(path, rec):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def git_state():
    try:
        commit = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "-C", REPO, "status", "--porcelain"], text=True).strip())
        branch = subprocess.check_output(["git", "-C", REPO, "branch", "--show-current"], text=True).strip()
        return {"commit": commit, "branch": branch, "dirty": dirty}
    except Exception as e:  # git missing or not a repo
        return {"error": str(e)}


def write_manifest(entry):
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "run_manifest.jsonl"), "a") as f:
        f.write(json.dumps(entry) + "\n")


# =============================================================================== prompting
def shuffle_options(options, seed, topic, task_id):
    """Upstream semantics (options[0] is correct) but with a per-question RNG, so the shuffle is
    identical across methods / context lengths and unaffected by resuming."""
    rng = random.Random(f"{seed}:{topic}:{task_id}")
    shuffled = rng.sample(options, len(options))
    return shuffled, "ABCD"[shuffled.index(options[0])]


def lenient_choice(response):
    """Diagnostic only (NOT the reported metric): accept 'B', 'B.', '<choice> B' etc."""
    m = re.search(r"<choice>\s*([ABCD])", response) or re.match(r"\W*([ABCD])\b", response.strip())
    return m.group(1) if m else None


class TopicContext:
    """All per-(topic, form, inter_turns) data the upstream prompt builders need."""

    def __init__(self, cfg, topic, form, inter_turns, turns_data):
        self.family = cfg["model_cfg"]["family"]
        self.form = form
        pref_type = {"explicit": "", "implicit-choice": "choice", "implicit-persona": "persona"}[form]
        # Namespace mimicking upstream argparse args (fields read by the vendored code).
        self.args = SimpleNamespace(model=cfg["model"], topic=topic, inter_turns=inter_turns,
                                    topk=cfg["rag_topk"], pref_type=pref_type)
        self.mcq = load_json(os.path.join(DATA, "mcq_options", f"{topic}.json"))
        self.multi_inter_message, self.multi_turn_message = extract_multi_turn_message(
            turns_data, self.args, self.family)
        rag_dir = os.path.join(DATA, "rag_retrieval")
        if form == "explicit":
            self.rag_data = load_json(os.path.join(rag_dir, "simcse_explicit_pref", f"{topic}_overall300_topk_history.json"))
            self.msg_idx = load_json(os.path.join(rag_dir, "simcse_explicit_pref", f"msg_index_{topic}_overall300_topk_history.json"))
        else:
            sub = "choice-based" if pref_type == "choice" else "persona-driven"
            self.pref_data = load_json(os.path.join(DATA, "implicit_preference", sub, f"{topic}.json"))
            suffix = "mcq" if pref_type == "choice" else "persona"
            self.rag_data = load_json(os.path.join(rag_dir, f"simcse_implicit_{pref_type}",
                                                   f"{topic}_overall300_topk_history_{suffix}.json"))
            self.msg_rag_data = load_json(os.path.join(
                rag_dir, "simcse_question_inter_conversation_similarities", f"{topic}_300_inter_similarities.json"))


def run_question(ctx, backend, method, task_id, options, settings):
    """One PrefEval classification question; mirrors upstream benchmark_classification.py."""
    a, fam = ctx.args, ctx.family
    sp, max_tokens = settings["system_prompt"], settings["max_mcq_tokens"]
    question = ctx.mcq[task_id]["question"]
    out = {}
    gen = backend.upstream_generate_message  # same signature as upstream generate_message

    if ctx.form == "explicit":
        preference = ctx.mcq[task_id]["preference"]
        # Upstream generates the reply to the stated preference with max_mcq_tokens (=5) too.
        pref_generation = backend.generate(create_user_pref_message(preference, fam, sp), max_tokens, "pref_response")
        out["response_to_pref"] = pref_generation
        if method == "rag":
            prompt = upstream_baselines.handle_rag_task_mcq(
                a, preference, pref_generation, question, ctx.multi_inter_message, fam, max_tokens, sp,
                ctx.rag_data, ctx.msg_idx, task_id, options)
        elif method == "selfcritic":
            first, end, critic = upstream_baselines.handle_selfcritic_task_mcq(
                a, preference, pref_generation, ctx.multi_inter_message, question, sp, None, None, fam,
                max_tokens, options)
            out.update(first_choice=first, self_critic=critic)
        else:
            prompt = get_question_prompt_mcq(
                preference, options, pref_generation, question, ctx.multi_inter_message, fam, a.inter_turns,
                remind=method == "remind", cot=method == "cot", system_prompt=sp)
    else:
        conversation = ctx.pref_data[task_id]["conversation"]
        conv_messages, conv_list = extract_implicit_messages(a, conversation, fam)
        if method == "rag":
            prompt = upstream_baselines.handle_rag_task_implicit_mcq(
                a, conv_messages, question, ctx.multi_inter_message, fam, max_tokens, sp, ctx.rag_data,
                ctx.msg_rag_data, task_id, conv_list, ctx.multi_turn_message, options)
        elif method == "selfcritic":
            first, end, critic = upstream_baselines.handle_selfcritic_task_implicit_mcq(
                a, conv_messages, ctx.multi_inter_message, question, sp, None, None, fam, max_tokens, options)
            out.update(first_choice=first, self_critic=critic)
        else:
            prompt = get_implicit_question_prompt_mcq(
                conv_messages, question, ctx.multi_inter_message, fam, a.inter_turns,
                remind=method == "remind", cot=method == "cot", options=options)

    if method != "selfcritic":
        end = gen(None, None, fam, sp, prompt, max_tokens)
    if fam == "mistral":  # upstream prefills "<choice>" for mistral and re-attaches it
        end = "<choice>" + end
    out["response_to_q"] = end
    return out


# =============================================================================== commands
def setup_logging(name):
    os.makedirs(LOGS, exist_ok=True)
    path = os.path.join(LOGS, f"{name}_{dt.datetime.now():%Y%m%d_%H%M%S}.log")
    fh = logging.FileHandler(path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%H:%M:%S"))
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers = [fh, ch]
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    return path


def cmd_run(opts):
    cfg = load_experiment(opts.config)
    if opts.limit is not None:
        cfg["limit"] = opts.limit
    settings = repo_settings()
    log_path = setup_logging(cfg["name"])
    mc = cfg["model_cfg"]
    backend = OllamaBackend(mc["ollama_tag"], mc["num_ctx"], host=opts.host, seed=cfg["seed"])
    # Upstream self-critic handlers call their module-level generate_message (Bedrock) -> route to Ollama.
    upstream_baselines.generate_message = backend.upstream_generate_message
    cells = list(iter_cells(cfg))
    todo = sum(n_questions(c["topic"], cfg["limit"]) - len(read_done(cell_path(cfg["model"], c))) for c in cells)
    log.info("Config %s | model %s (%s) | %d cells | %d questions remaining | log: %s",
             opts.config, cfg["model"], mc["ollama_tag"], len(cells), todo, log_path)
    if opts.dry_run:
        for c in cells:
            done = len(read_done(cell_path(cfg["model"], c)))
            print(f"{c['topic']:34s} {c['form']:17s} inter{c['inter_turns']:<3d} {c['method']:10s} "
                  f"{done}/{n_questions(c['topic'], cfg['limit'])}")
        return 0

    try:
        manifest = {"event": "start", "time": dt.datetime.now().isoformat(timespec="seconds"),
                    "config_file": opts.config, "config": {k: v for k, v in cfg.items()},
                    "settings": settings, "backend_rev": BACKEND_REV, "git": git_state(), "ollama_version": backend.server_version(),
                    "model": backend.model_info(), "log": os.path.relpath(log_path, REPO)}
    except Exception as e:
        log.error("Cannot reach Ollama at %s (%s). Start it with `ollama serve` / `systemctl start ollama`.",
                  opts.host, e)
        return 2
    write_manifest(manifest)

    turns_data = load_json(os.path.join(DATA, "filtered_inter_turns.json"))
    status, consecutive_failures, n_done_run = "completed", 0, 0
    t_start = time.time()
    try:
        with logging_redirect_tqdm():
            for ci, cell in enumerate(cells, 1):
                path = cell_path(cfg["model"], cell)
                done = read_done(path, repair=True)
                total = n_questions(cell["topic"], cfg["limit"])
                remaining = [i for i in range(total) if i not in done]
                tag = f"[{ci}/{len(cells)}] {cell['topic']} | {cell['form']} | inter{cell['inter_turns']} | {cell['method']}"
                if not remaining:
                    log.debug("%s already complete (%d/%d)", tag, total, total)
                    continue
                log.info("%s: %d/%d done, running %d", tag, len(done), total, len(remaining))
                ctx = TopicContext(cfg, cell["topic"], cell["form"], cell["inter_turns"], turns_data)
                correct = sum(r["correct"] for r in done.values())
                for task_id in tqdm(remaining, desc=f"{cell['method']}/inter{cell['inter_turns']}", leave=False):
                    options, correct_letter = shuffle_options(
                        ctx.mcq[task_id]["classification_task_options"], cfg["seed"], cell["topic"], task_id)
                    backend.calls = []
                    t0 = time.time()
                    try:
                        out = run_question(ctx, backend, cell["method"], task_id, options, settings)
                    except OllamaUnavailable:
                        raise
                    except Exception as e:
                        consecutive_failures += 1
                        log.exception("%s task %d failed (%d consecutive): %s", tag, task_id, consecutive_failures, e)
                        if consecutive_failures >= CONSECUTIVE_FAILURE_LIMIT:
                            raise RuntimeError("too many consecutive failures; aborting") from e
                        continue
                    consecutive_failures = 0
                    choice = extract_choice(out["response_to_q"])
                    rec = {
                        "task_id": task_id, "topic": cell["topic"], "form": cell["form"], "method": cell["method"],
                        "inter_turns": cell["inter_turns"], "model": cfg["model"], "backend_rev": BACKEND_REV,
                        "shuffled_options": options, "correct_idx": correct_letter,
                        **out,
                        "choice": choice, "choice_lenient": lenient_choice(out["response_to_q"]),
                        "correct": choice == correct_letter,
                        "seconds": round(time.time() - t0, 3), "calls": backend.calls,
                        "truncated": any(c["truncated"] for c in backend.calls),
                        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                    }
                    append_record(path, rec)
                    correct += rec["correct"]
                    n_done_run += 1
                    log.debug("%s task %d: choice=%s correct=%s raw=%r", tag, task_id, choice, correct_letter,
                              out["response_to_q"])
                log.info("%s finished: accuracy %.1f%% (%d/%d)", tag, 100 * correct / total, correct, total)
    except KeyboardInterrupt:
        status = "interrupted"
        log.warning("Interrupted. Everything up to the last finished question is saved; rerun the same command to resume.")
    except (OllamaUnavailable, RuntimeError) as e:
        status = "aborted"
        log.error("Aborted: %s. Rerun the same command to resume.", e)
    finally:
        write_manifest({"event": "end", "time": dt.datetime.now().isoformat(timespec="seconds"),
                        "config_file": opts.config, "status": status, "questions_this_run": n_done_run,
                        "elapsed_s": round(time.time() - t_start, 1)})
    log.info("Run %s: %d questions in %.1f min. Next: python -m phase1.aggregate", status, n_done_run,
             (time.time() - t_start) / 60)
    return {"completed": 0, "interrupted": 130}.get(status, 1)


def cmd_status(opts):
    cfg = load_experiment(opts.config)
    rows, remaining_by_method, sec_by_method = [], {}, {}
    for c in iter_cells(cfg):
        recs = read_done(cell_path(cfg["model"], c))
        total = n_questions(c["topic"], cfg["limit"])
        m = c["method"]
        remaining_by_method[m] = remaining_by_method.get(m, 0) + total - len(recs)
        sec_by_method.setdefault(m, []).extend(r["seconds"] for r in recs.values())
        acc = f"{100 * sum(r['correct'] for r in recs.values()) / len(recs):5.1f}%" if recs else "    -"
        rows.append((c, len(recs), total, acc))
    for c, done, total, acc in rows:
        mark = "done" if done == total else ("....") if done else ""
        print(f"{c['topic']:34s} {c['form']:17s} inter{c['inter_turns']:<3d} {c['method']:10s} "
              f"{done:4d}/{total:<4d} acc {acc} {mark}")
    eta = 0.0
    for m, rem in remaining_by_method.items():
        s = sec_by_method[m]
        per_q = sum(s) / len(s) if s else DEFAULT_SEC_PER_Q[m]
        eta += rem * per_q
    done_q = sum(r[1] for r in rows)
    total_q = sum(r[2] for r in rows)
    print(f"\n{done_q}/{total_q} questions done ({100 * done_q / max(total_q, 1):.1f}%). "
          f"Estimated time remaining: {eta / 3600:.1f} h (from measured per-method latency)")
    return 0


def cmd_show_prompt(opts):
    """Print the exact prompt the model would get (first model call for selfcritic)."""
    models = load_yaml(MODELS_YAML)
    cfg = {"model": opts.model, "model_cfg": models[opts.model], "rag_topk": 5}
    settings = repo_settings()
    ctx = TopicContext(cfg, opts.topic, opts.form, opts.inter_turns,
                       load_json(os.path.join(DATA, "filtered_inter_turns.json")))
    options, correct = shuffle_options(ctx.mcq[opts.task_id]["classification_task_options"], 41, opts.topic, opts.task_id)

    class Capture:  # fake backend that records prompts instead of calling the model
        calls = []

        def generate(self, prompt, max_tokens, purpose="answer"):
            print(f"\n{'=' * 30} PROMPT ({purpose}, max_tokens={max_tokens}) {'=' * 30}\n{prompt}")
            return "<PREF_RESPONSE>" if purpose == "pref_response" else "<choice>A</choice>"

        def upstream_generate_message(self, c, mid, mt, sp=None, messages=None, max_tokens=None, **kw):
            return self.generate(messages, max_tokens, "critic" if max_tokens > 50 else "answer")

    cap = Capture()
    upstream_baselines.generate_message = cap.upstream_generate_message
    run_question(ctx, cap, opts.method, opts.task_id, options, settings)
    print(f"\n(correct answer: {correct})")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run (or resume) an experiment config")
    r.add_argument("--config", required=True)
    r.add_argument("--limit", type=int, default=None, help="max questions per cell (overrides config)")
    r.add_argument("--dry-run", action="store_true", help="list cells and progress, call nothing")
    r.add_argument("--host", default=os.environ.get("OLLAMA_HOST_URL", "http://localhost:11434"))
    s = sub.add_parser("status", help="progress, per-cell accuracy and ETA")
    s.add_argument("--config", required=True)
    sp = sub.add_parser("show-prompt", help="print the exact prompt(s) for one question")
    sp.add_argument("--model", default="llama3-8b")
    sp.add_argument("--form", default="explicit", choices=FORMS)
    sp.add_argument("--method", default="zero-shot", choices=METHODS)
    sp.add_argument("--inter-turns", type=int, default=0)
    sp.add_argument("--topic", default="travel_restaurant", choices=ALL_TOPICS)
    sp.add_argument("--task-id", type=int, default=0)
    opts = p.parse_args(argv)
    if opts.cmd == "run":
        return cmd_run(opts)
    return {"status": cmd_status, "show-prompt": cmd_show_prompt}[opts.cmd](opts)


if __name__ == "__main__":
    sys.exit(main())
