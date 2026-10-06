# phase1/: local PrefEval classification baselines

- `upstream/` holds **verbatim** copies of upstream prompt code from commit 5079505. Only import lines are edited
  (marked `[phase1 edit]`). Never change prompt text or formatting here. If a prompt needs normalising, do it in
  `backend.py` and bump `BACKEND_REV`.
- `backend.py` contains `OllamaBackend`, which calls `/api/generate` with `raw=true`. It strips the leading literal `<|begin_of_text|>`
  (Ollama adds BOS itself) and trims trailing spaces and tabs. Without the trim, upstream's Llama self-critic templates end in whitespace
  and the model returns empty output. In Ollama 0.34, `prompt_eval_count` already includes cached tokens.
- `BACKEND_REV` is stored in every result record. `runner.read_done` and `aggregate` ignore other revisions.
  **Bump it whenever decoding or prompt normalisation changes**, otherwise old and new results get mixed.
- `runner.py` runs cells in the order topic → form → inter_turns → method. The resume key is `task_id` within a cell file.
  The self-critic handlers call the module-global `upstream.baselines_handling_classification.generate_message`, which
  `cmd_run` monkey-patches to the backend adapter.
- RAG at inter_turns=0 is skipped, because the upstream assert needs at least 5 exchanges. RAG cells whose retrieval file doesn't parse
  are also skipped (`rag_unavailable_reason`). The known case is upstream's truncated explicit `entertain_games` file.
  RAG files are loaded only for RAG cells (`need_rag`), so a broken file never blocks the other methods.
  Upstream implicit RAG files end early in 7 topics. `rag_available_count` / `n_questions(..., form, method)` limit RAG
  cells to questions that have retrieval data (973/1000 for implicit). Use these helpers, never `len(mcq)`, for RAG cell sizes.
- Option shuffle: `random.Random(f"{seed}:{topic}:{task_id}")`, where `options[0]` is the correct answer in the dataset.
- Only `run` may rewrite (repair) result files. `status` must stay read-only, because it runs alongside live jobs.
- The metric is upstream `extract_choice` (strict). `choice_lenient` is a diagnostic only.
- Verify changes with `python -m phase1.runner show-prompt ...` and `configs/phase1/smoke.yaml` (about 2 min).
- **Ollama settings guard:** `models.yaml: ollama_env` lists the numerics-relevant server settings (flash attention, KV-cache type)
  each model's baselines were produced with: Llama3-8B = defaults, Mistral-7B = FA + q8_0 KV. `run` exits with code 3 on a mismatch.
  Never relax this for Phase 3 comparisons. Switch the server setting instead.
