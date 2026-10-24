# phase3/: preference memory + retrieval (see docs/PHASE3.md)

- Env: `prefeval-p2` + `requirements-phase3.txt` (sentence-transformers 6.x on transformers v5).
- Phase 1/2 code and results are **read-only** inputs: replies come from `results/phase1/<llm>/explicit/zero-shot/inter0`,
  and the SimCSE files are loaded via `phase1.runner.rag_paths` / `rag_unavailable_reason` / `rag_available_count`. Never edit them.
- Encoders run locally via sentence-transformers (not Ollama). `embed.local_snapshot` loads the **pinned commit** from the HF
  cache and downloads it only if missing. Change a model or prompt in `configs/phase3/encoders.yaml`, and then delete
  `data/phase3/emb/<encoder>/`, because the cache is keyed by text hash only, not by prompt.
- `conversations.pool()` is the single definition of a candidate pool. Its order must equal the SimCSE score index order
  (explicit: pref, reply, LMSYS...; implicit: 4 dialogue messages, then LMSYS). `phase3.checks` asserts the SimCSE
  top-5 equals upstream's RAG selection. Rerun it after touching `conversations.py` or `retrieve.py`.
- Only message content is embedded (no role prefix); the query is the MCQ question.
- Partial runs (`--topics` / `--inter-turns`) require `PHASE3_RESULTS_DIR=results/phase3_smoke` (asserted), so the
  committed `results/phase3/` is only ever written by full runs.
- Ties in ranking are broken by conversation order (stable sort), the same as upstream.
- MCQ inputs (memories `results/phase3/memory/`, EmbeddingGemma-2 rankings) are always read from `MAIN_RESULTS_DIR`. Only
  outputs follow `PHASE3_RESULTS_DIR`. `--limit/--topics` runs require the smoke dir (asserted).
- Classifier leakage rule: topic t uses the Phase 2 fold model whose **test** fold contains t (`memory.topic_fold`).
- Every Phase 3 method must stay "one change versus a Phase 1 baseline". `phase3.checks.check_prompts` asserts that the reductions
  are byte-identical. Rerun it after touching `prompts.py`.
- Upstream explicit `msg_index` files are shifted by one LMSYS exchange in 13 topics (docs/PHASE3.md). Never decode SimCSE indices
  with them in Phase 3 code. Use `conversations.pool` (real conversation order).
- Long tiers are handed to the user via `scripts/phase3_tmux.sh "python -m phase3.runner run --config ..."`.
  Each tier must run with its model's Ollama settings (guard exit code 3).
