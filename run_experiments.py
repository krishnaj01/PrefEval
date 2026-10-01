#!/usr/bin/env python3
"""
run_experiments.py
==================
Master runner for the PrefEval project grid.

Tracks every (model × topic × task × inter_turns × pref_form) experiment
in  benchmark_results/run_state.json  and can resume from any crash.

Usage
-----
# Fresh run:
python run_experiments.py

# Resume after a crash / interruption:
python run_experiments.py --resume

# Retry only failed experiments (leaves done ones untouched):
python run_experiments.py --resume --retry-failed

# Dry-run: print what would run without running anything:
python run_experiments.py --dry-run

# Show current progress only:
python run_experiments.py --status

# Override topics / tasks / turns inline (space-separated):
python run_experiments.py --resume \\
    --topics travel_restaurant lifestyle_dietary \\
    --tasks zero-shot remind \\
    --turns 2 10
"""

import argparse
import os
import subprocess
import sys
import textwrap

# ── locate repo root (this file lives in <repo_root>/) ────────────────────────
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

from utils.checkpoint import (
    init_state,
    is_done,
    load_state,
    make_exp_key,
    mark_done,
    mark_failed,
    mark_running,
    print_progress,
    save_state,
)

# ══════════════════════════════════════════════════════════════════════════════
# Experiment Grid  —  edit these to match your project scope
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_MODEL = "llama8b-local"

DEFAULT_TOPICS = [
    "travel_restaurant",
    "lifestyle_dietary",
    "entertain_shows",
    "shop_technology",
    "education_learning_styles",
    "lifestyle_health",
]

DEFAULT_TASKS = ["zero-shot", "remind", "cot", "selfcritic", "rag"]
# NOTE: "proposed" will be added once your pipeline module is ready.

DEFAULT_TURNS = [2, 10, 20]   # 70 removed: exceeds RTX 3060 KV-cache budget.
                               # Add 50 here if you want to attempt the stretch run.

DEFAULT_PREF_FORMS = ["explicit", "implicit"]
DEFAULT_PREF_TYPE  = "choice"   # for implicit form; persona-driven is optional

STATE_FILE = os.path.join(REPO_ROOT, "benchmark_results", "run_state.json")

# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def build_experiment_list(model, topics, tasks, turns_list, pref_forms, pref_type):
    """Return an ordered list of experiment dicts for the given grid."""
    experiments = []
    for topic in topics:
        for pref_form in pref_forms:
            for inter_turns in turns_list:
                for task in tasks:
                    pt = pref_type if pref_form == "implicit" else ""
                    experiments.append(
                        dict(
                            model=model,
                            topic=topic,
                            task=task,
                            inter_turns=inter_turns,
                            pref_form=pref_form,
                            pref_type=pt,
                            key=make_exp_key(model, topic, task, inter_turns, pref_form, pt),
                        )
                    )
    return experiments


def run_one(exp: dict, args) -> None:
    """
    Run one experiment: call benchmark_classification.py,
    then get_preference_following_accuracy.py.

    Raises subprocess.CalledProcessError on failure.
    """
    model       = exp["model"]
    topic       = exp["topic"]
    task        = exp["task"]
    inter_turns = exp["inter_turns"]
    pref_form   = exp["pref_form"]
    pref_type   = exp["pref_type"]

    base_cmd = [
        sys.executable,
        os.path.join(REPO_ROOT, "classification_task", "benchmark_classification.py"),
        f"--model={model}",
        f"--topic={topic}",
        f"--task={task}",
        f"--inter_turns={inter_turns}",
        f"--pref_form={pref_form}",
    ]
    if pref_form == "implicit":
        base_cmd.append(f"--pref_type={pref_type}")

    # --- Step 1: generate responses ---
    print(f"\n  [1/2] Generating: {exp['key']}")
    subprocess.run(base_cmd, check=True, cwd=os.path.join(REPO_ROOT, "example_scripts"))

    # --- Step 2: compute accuracy ---
    acc_task = "rag_5" if task == "rag" else task
    acc_cmd = [
        sys.executable,
        os.path.join(REPO_ROOT, "generation_task",
                     "get_preference_following_accuracy_generation_task.py"),
        f"--model={model}",
        f"--topic={topic}",
        f"--task={acc_task}",
        f"--inter_turn={inter_turns}",
        f"--pref_form={pref_form}",
    ]
    if pref_form == "implicit":
        acc_cmd.append(f"--pref_type={pref_type}")

    print(f"  [2/2] Scoring:    {exp['key']}")
    subprocess.run(acc_cmd, check=True, cwd=os.path.join(REPO_ROOT, "example_scripts"))


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="PrefEval master runner with checkpoint/resume support.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python run_experiments.py                         # fresh run
              python run_experiments.py --resume                # skip done experiments
              python run_experiments.py --resume --retry-failed # also retry failed ones
              python run_experiments.py --status                # show progress, exit
              python run_experiments.py --dry-run               # show plan, don't run
        """),
    )
    parser.add_argument("--resume",       action="store_true",
                        help="Skip experiments already marked 'done' in run_state.json.")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Re-run experiments marked 'failed' (implies --resume).")
    parser.add_argument("--dry-run",      action="store_true",
                        help="Print what would run without actually running anything.")
    parser.add_argument("--status",       action="store_true",
                        help="Print current progress from run_state.json and exit.")

    # Grid overrides
    parser.add_argument("--model",   default=DEFAULT_MODEL)
    parser.add_argument("--topics",  nargs="+", default=DEFAULT_TOPICS)
    parser.add_argument("--tasks",   nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--turns",   nargs="+", type=int, default=DEFAULT_TURNS)
    parser.add_argument("--pref-forms", nargs="+", default=DEFAULT_PREF_FORMS,
                        dest="pref_forms")
    parser.add_argument("--pref-type",  default=DEFAULT_PREF_TYPE, dest="pref_type")

    args = parser.parse_args()

    # ── status-only mode ──────────────────────────────────────────────────────
    if args.status:
        state = load_state(STATE_FILE)
        if not state["experiments"]:
            print("No run_state.json found (or it is empty). Nothing has been run yet.")
        else:
            print(f"\nState file: {STATE_FILE}")
            print_progress(state)
            # Show per-experiment table
            print("\n  Experiment status:")
            max_klen = max(len(k) for k in state["experiments"])
            for k, v in state["experiments"].items():
                icon = {"done": "✅", "pending": "⏳", "running": "🔄",
                        "failed": "❌"}.get(v["status"], "?")
                print(f"    {icon}  {k:<{max_klen}}   {v['status']}")
        return

    # ── build experiment list ─────────────────────────────────────────────────
    experiments = build_experiment_list(
        args.model, args.topics, args.tasks, args.turns,
        args.pref_forms, args.pref_type,
    )
    keys = [e["key"] for e in experiments]

    grid_config = dict(
        model=args.model, topics=args.topics, tasks=args.tasks,
        turns=args.turns, pref_forms=args.pref_forms, pref_type=args.pref_type,
    )

    # ── initialise / merge state ──────────────────────────────────────────────
    state = init_state(STATE_FILE, keys, grid_config)

    # If --retry-failed, reset failed experiments back to pending
    if args.retry_failed:
        for key in keys:
            if state["experiments"][key]["status"] == "failed":
                state["experiments"][key] = {
                    "status": "pending",
                    "started_at": None,
                    "finished_at": None,
                    "error": None,
                }
        save_state(STATE_FILE, state)
        print("  Reset failed experiments → pending.")

    # ── print plan ────────────────────────────────────────────────────────────
    total    = len(experiments)
    to_run   = [e for e in experiments
                if not ((args.resume or args.retry_failed) and is_done(state, e["key"]))]
    skipped  = total - len(to_run)

    print(f"\n{'='*60}")
    print(f"  PrefEval Master Runner")
    print(f"{'='*60}")
    print(f"  Total experiments : {total}")
    print(f"  Will skip (done)  : {skipped}")
    print(f"  Will run          : {len(to_run)}")
    print(f"  State file        : {STATE_FILE}")
    print(f"{'='*60}\n")
    print_progress(state)
    print()

    if args.dry_run:
        print("  [DRY RUN] Experiments that would run:")
        for e in to_run:
            print(f"    {e['key']}")
        return

    if not to_run:
        print("  Nothing to run — all experiments are already done!")
        print("  Use --retry-failed to re-run failed ones, or edit the grid.")
        return

    # ── run loop ──────────────────────────────────────────────────────────────
    done_count = skipped
    fail_count = sum(
        1 for e in experiments
        if state["experiments"][e["key"]]["status"] == "failed"
    )

    for i, exp in enumerate(to_run, 1):
        key = exp["key"]

        if (args.resume or args.retry_failed) and is_done(state, key):
            continue  # extra safety guard

        print(f"\n[{i}/{len(to_run)}] Starting: {key}")
        mark_running(STATE_FILE, state, key)

        try:
            run_one(exp, args)
            mark_done(STATE_FILE, state, key)
            done_count += 1
            print(f"  ✅ Done: {key}")
        except subprocess.CalledProcessError as exc:
            err = f"subprocess exited with code {exc.returncode}"
            mark_failed(STATE_FILE, state, key, err)
            fail_count += 1
            print(f"  ❌ Failed: {key}  ({err})")
            print(f"     To retry: python run_experiments.py --resume --retry-failed")
        except Exception as exc:
            err = str(exc)
            mark_failed(STATE_FILE, state, key, err)
            fail_count += 1
            print(f"  ❌ Failed: {key}  ({err})")

        # Print running progress after every experiment
        print()
        print_progress(state)

    # ── final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Run complete.")
    print_progress(state)
    print(f"  State saved to: {STATE_FILE}")
    if fail_count:
        print(f"\n  Re-run failed experiments with:")
        print(f"    python run_experiments.py --resume --retry-failed")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

