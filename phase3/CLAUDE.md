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
