# phase2/: preference-detection classifier (see docs/PHASE2.md)

- Env: `conda activate prefeval-p2` (`requirements-phase2.txt`). It is separate from Phase 1's `prefeval` env. transformers is **v5**:
  use `dtype=` (not `torch_dtype`) and `warmup_steps` (no `warmup_ratio`), and do not pass `save_safetensors`.
- **Model input is `text` only.** All other row fields are metadata for splitting and analysis. Never feed them to a model.
- Leakage rules (asserted in `splits.py`): topic-level folds; LMSYS distractor turns (`lmsys_ood`) are evaluation-only,
  because Phase 3 replays them. Fold k: test = k, val = k+1, train = the rest (`splits.fold_roles`).
- LLM calls (pseudo-labels, baseline) reuse `phase1.backend.OllamaBackend` and `phase1.runner.check_ollama_env`.
  The labeller is llama3-8b under its default Ollama settings. Caches in `data/phase2/pseudo/*.jsonl` are append-only and never relabelled.
  The prompt text is hashed into every row: if you change a prompt, use a new cache file.
- Training artefacts: `results/phase2/<model>/fold<k>/` (committed) and `models/phase2/` (git-ignored). A fold with `metrics.json`
  is skipped unless `--force`. Smoke runs (`--smoke`) write to `results/phase2_smoke/` and `models/phase2_smoke/`. Analyse them with
  `PHASE2_RESULTS_DIR=results/phase2_smoke python -m phase2.evaluate` (and the same for ensemble / report).
- Thresholds are tuned on validation only. Ensembles fit everything (thresholds, stacking LR) on validation and score on test.
- Long steps (pseudo-labelling, training, LLM baseline) are handed to the user as tmux commands (`scripts/phase2_tmux.sh "<cmd>"`).
