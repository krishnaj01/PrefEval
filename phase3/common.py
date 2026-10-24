"""Shared helpers for Phase 3: paths and config loading. JSONL / hashing / manifest helpers are reused from Phase 2."""

import os

from phase2.common import (append_jsonl, env_versions, git_state, load_yaml, now, read_jsonl,  # noqa: F401
                           set_seed, sha256_text, write_jsonl)
from phase2.common import write_run_manifest as _write_run_manifest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(REPO, "benchmark_dataset")
CONFIG_DIR = os.path.join(REPO, "configs", "phase3")
EMB_DIR = os.path.join(REPO, "data", "phase3", "emb")            # embedding caches (git-ignored, reproducible)
# rankings, metrics, figures (committed). PHASE3_RESULTS_DIR overrides it (smoke tests: results/phase3_smoke)
RESULTS_DIR = os.environ.get("PHASE3_RESULTS_DIR") or os.path.join(REPO, "results", "phase3")
RETRIEVAL_DIR = os.path.join(RESULTS_DIR, "retrieval")
# Inputs of the MCQ runs (memories, EmbeddingGemma-2 rankings) are always read from the real results, also in smoke runs
MAIN_RESULTS_DIR = os.path.join(REPO, "results", "phase3")
PHASE1_RESULTS = os.path.join(REPO, "results", "phase1")
LOGS_DIR = os.path.join(REPO, "logs", "phase3")

SIMCSE = "simcse_paper"                  # the paper's retriever, scored from the stored upstream similarities


def write_run_manifest(entry):
    """Append to results/phase3/run_manifest.jsonl (time, git state, library versions + entry)."""
    _write_run_manifest(entry, os.path.join(RESULTS_DIR, "run_manifest.jsonl"))


def encoders_config():
    return load_yaml(os.path.join(CONFIG_DIR, "encoders.yaml"))


def retrieval_config():
    return load_yaml(os.path.join(CONFIG_DIR, "retrieval.yaml"))


def encoder_names():
    """Encoders in config order (the fixed order used in every table and figure)."""
    return list(encoders_config())
