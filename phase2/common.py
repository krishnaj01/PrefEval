"""Shared helpers for Phase 2: paths, config loading, seeding, crash-safe JSONL I/O, run manifest."""

import datetime as dt
import hashlib
import json
import os
import platform
import random
import subprocess

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(REPO, "benchmark_dataset")
DATA_DIR = os.path.join(REPO, "data", "phase2")            # built datasets + pseudo-labels (committed)
PSEUDO_DIR = os.path.join(DATA_DIR, "pseudo")              # LLM label caches (committed)
# metrics, predictions, figures (committed). PHASE2_RESULTS_DIR overrides it (used for smoke tests: results/phase2_smoke)
RESULTS_DIR = os.environ.get("PHASE2_RESULTS_DIR") or os.path.join(REPO, "results", "phase2")
MODELS_DIR = os.path.join(REPO, "models", "phase2")        # checkpoints (git-ignored)
LOGS_DIR = os.path.join(REPO, "logs", "phase2")            # verbose logs (git-ignored)
CONFIG_DIR = os.path.join(REPO, "configs", "phase2")

PREFEVAL_TURNS = os.path.join(DATA_DIR, "prefeval_turns.jsonl")   # output of build_data
LMSYS_OOD = os.path.join(DATA_DIR, "lmsys_ood.jsonl")              # output of build_data (eval only)
OASST_RAW = os.path.join(DATA_DIR, "oasst_sample.jsonl")           # sampled OASST turns (unlabelled)
DATASET = os.path.join(DATA_DIR, "dataset.jsonl")                  # final labelled dataset with folds
SPLITS = os.path.join(DATA_DIR, "splits.json")
DATA_MANIFEST = os.path.join(DATA_DIR, "manifest.json")
AUDIT_CSV = os.path.join(DATA_DIR, "audit_sample.csv")
AUDIT_REPORT = os.path.join(DATA_DIR, "audit_report.json")
RUN_MANIFEST = os.path.join(RESULTS_DIR, "run_manifest.jsonl")

MODEL_NAMES = ["distilbert", "modernbert", "tinyllama", "qwen3"]   # fixed order used in every table/figure


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def data_config():
    return load_yaml(os.path.join(CONFIG_DIR, "data.yaml"))


def model_config(name_or_path):
    path = name_or_path if name_or_path.endswith(".yaml") else os.path.join(CONFIG_DIR, f"{name_or_path}.yaml")
    return load_yaml(path)


def set_seed(seed):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ----------------------------------------------------------------------------- JSONL
def read_jsonl(path):
    """Read a JSONL file, silently skipping a torn last line left by a crash."""
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def write_jsonl(path, rows):
    """Atomic full rewrite (write temp file, then rename)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def append_jsonl(path, row):
    """Append one record and fsync it: the resume unit for long-running LLM labelling."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


# ----------------------------------------------------------------------------- provenance
def git_state():
    try:
        commit = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
        branch = subprocess.check_output(["git", "-C", REPO, "branch", "--show-current"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "-C", REPO, "status", "--porcelain"], text=True).strip())
        return {"commit": commit, "branch": branch, "dirty": dirty}
    except Exception as e:
        return {"error": str(e)}


def env_versions():
    out = {"python": platform.python_version()}
    for mod in ("torch", "transformers", "peft", "datasets", "sklearn", "accelerate"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            pass
    try:
        import torch
        if torch.cuda.is_available():
            out["gpu"] = torch.cuda.get_device_name(0)
            out["cuda"] = torch.version.cuda
    except Exception:
        pass
    return out


def now():
    return dt.datetime.now().isoformat(timespec="seconds")


def write_run_manifest(entry, path=None):
    entry = {"time": now(), "git": git_state(), "env": env_versions(), **entry}
    append_jsonl(path or RUN_MANIFEST, entry)


def update_data_manifest(key, value):
    """data/phase2/manifest.json: provenance of every data artefact (hashes, revisions, LLM digests)."""
    os.makedirs(DATA_DIR, exist_ok=True)
    m = {}
    if os.path.exists(DATA_MANIFEST):
        with open(DATA_MANIFEST) as f:
            m = json.load(f)
    m[key] = {"updated": now(), **value}
    with open(DATA_MANIFEST, "w") as f:
        json.dump(m, f, indent=1)
