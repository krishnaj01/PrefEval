"""
utils/checkpoint.py
-------------------
Atomic checkpoint helpers for the PrefEval master runner.

Design: all state is kept in a single JSON file at <benchmark_results>/run_state.json.
Writes are atomic (write temp -> rename) so a crash mid-write never corrupts the file.

State file schema:
{
    "meta": {
        "created_at": "<ISO timestamp>",
        "last_updated": "<ISO timestamp>",
        "config": { ... }
    },
    "experiments": {
        "<exp_key>": {
            "status": "pending" | "running" | "done" | "failed",
            "started_at": "<ISO timestamp>" | null,
            "finished_at": "<ISO timestamp>" | null,
            "error": "<str>" | null
        }
    }
}
"""

import json
import os
import tempfile
from datetime import datetime, timezone


def make_exp_key(model, topic, task, inter_turns, pref_form, pref_type=""):
    """Return a deterministic string key for one experiment configuration."""
    parts = [model, topic, task, f"{inter_turns}turns", pref_form]
    if pref_type:
        parts.append(pref_type)
    return "__".join(parts)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _atomic_write(path, data):
    """Write data to path atomically via temp-file + rename."""
    dir_ = os.path.dirname(os.path.abspath(path))
    os.makedirs(dir_, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)   # atomic on POSIX
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_state(state_path):
    """Load state JSON. Returns empty scaffold if file does not exist."""
    if os.path.exists(state_path):
        with open(state_path, "r") as f:
            return json.load(f)
    return {
        "meta": {"created_at": _now(), "last_updated": _now(), "config": {}},
        "experiments": {},
    }


def save_state(state_path, state):
    """Atomically persist state to state_path."""
    state["meta"]["last_updated"] = _now()
    _atomic_write(state_path, state)


def init_state(state_path, experiment_keys, config):
    """
    Initialise state file with experiment_keys set to 'pending'.
    Existing entries are left untouched (safe for --resume).
    Returns the merged state dict.
    """
    state = load_state(state_path)
    state["meta"]["config"] = config
    for key in experiment_keys:
        if key not in state["experiments"]:
            state["experiments"][key] = {
                "status": "pending",
                "started_at": None,
                "finished_at": None,
                "error": None,
            }
    save_state(state_path, state)
    return state


def mark_running(state_path, state, key):
    state["experiments"][key]["status"] = "running"
    state["experiments"][key]["started_at"] = _now()
    save_state(state_path, state)


def mark_done(state_path, state, key):
    state["experiments"][key]["status"] = "done"
    state["experiments"][key]["finished_at"] = _now()
    save_state(state_path, state)


def mark_failed(state_path, state, key, error):
    state["experiments"][key]["status"] = "failed"
    state["experiments"][key]["finished_at"] = _now()
    state["experiments"][key]["error"] = error
    save_state(state_path, state)


def is_done(state, key):
    return state["experiments"].get(key, {}).get("status") == "done"


def print_progress(state):
    exps = state["experiments"]
    counts = {"done": 0, "running": 0, "failed": 0, "pending": 0}
    for v in exps.values():
        counts[v["status"]] = counts.get(v["status"], 0) + 1
    total = len(exps)
    print(
        f"  Progress: {counts['done']}/{total} done  |  "
        f"{counts['pending']} pending  |  "
        f"{counts['failed']} failed  |  "
        f"{counts['running']} running"
    )
    if counts["failed"]:
        failed = [k for k, v in exps.items() if v["status"] == "failed"]
        print(f"  Failed: {failed}")

