# Phase 1: Local Baseline Reproduction (PrefEval classification task)

**Project:** *Improving Long-Context Preference Following in LLMs via a Trained Preference-Detection
Classifier and Enhanced Retrieval* (Krishna Jhanwar, 12341250). Base paper: Zhao et al., ICLR 2025 (PrefEval).

| Phase | Goal | Status |
|---|---|---|
| **1. Setup & baseline verification** | Reproduce the paper's classification-task baselines with a **local** model | **this document** |
| 2. Preference-detection classifier | Train DistilBERT/ModernBERT-scale "does this turn state a preference?" classifier | not started |
| 3. Full pipeline | Classifier → preference memory + modern retriever (bge/e5), compared against the Phase 1 baselines | not started |

Phase 1 results are what Phase 3 is compared against, so they live in their own tree (`results/phase1/`)
and nothing written in Phase 2/3 should write there.

---

## 1. What Phase 1 measures

PrefEval places a user preference at the start of a conversation, inserts *N* unrelated LMSYS turns
(`inter_turns`), then asks a question with 4 options, only one of which respects the preference.
**Accuracy = fraction of questions where the model picks that option.**

| Paper row (Fig. 6) | `inter_turns` | ≈ prompt tokens (Llama3) |
|---|---|---|
| 0.2k | 0 | ~300 |
| 1k | 3 | ~1.4k |
| 3k | 8 | ~3.3k |
| 10k / 16k / 23k | 28 / 48 / 68 | (Mistral only, optional Tier C) |

The paper's "N turns" equals `inter_turns + 2` (the preference turn and the query turn). I checked this against the distractor data.

Methods (paper §3.1): **Zero-shot**, **Reminder** (one appended sentence), **CoT** (5 few-shot examples),
**Self-Critic** (answer → critique → revise; 3 model calls), **RAG** (top-5 exchanges retrieved by
SimCSE; the authors' retrieval scores are precomputed in `benchmark_dataset/rag_retrieval/`, so no
embedding model is needed). RAG at `inter_turns=0` is skipped: it is undefined upstream and the paper reports no value there.

Model: **Llama-3-8B-Instruct** (`llama3:8b-instruct-q8_0` via Ollama). This is one of the six models in the paper,
so its column of Figure 6 is the direct baseline:

| Llama3-8B, paper Fig. 6 (%) | 0.2k | 1k | 3k |
|---|---|---|---|
| Zero-shot | 85 | 55 | 42 |
| Reminder | 94 | 80 | 59 |
| Self-Critic | 79 | 73 | 62 |
| CoT | 83 | 67 | 50 |
| RAG (top-5) | – | 87 | 80 |

## 2. What is (and isn't) possible without Amazon Bedrock

| Task | Local? | Comparable to paper? |
|---|---|---|
| Classification, explicit, Llama3-8B, 0.2k–3k, all 5 methods | ✅ | ✅ direct (Fig. 6) |
| Classification, Mistral-7B-v0.2, up to 70 turns (~23k tokens) | ✅ (32k context, slower) | ✅ direct (Fig. 6) |
| Classification, implicit choice / persona | ✅ | ⚠ paper only reports these for *generation*, so these become **your own** baselines |
| Llama3-8B beyond ~10 turns | ❌ 8k context limit | – |
| Generation task + 4 error types | ⚠ only with a local judge (e.g. `qwen3:14b`) | ❌ paper's judge is Claude 3 Sonnet. Use only as a Phase 3 sanity check |
| SFT of Mistral-7B (paper §3.7) | ❌ not on 12 GB | – |
| Phase 2 classifier training | ✅ easily | n/a |

## 3. How the pipeline works

```
configs/phase1/<tier>.yaml
        │  (model, forms, methods, inter_turns, topics)
        ▼
phase1/runner.py ── loops cells: topic → form → inter_turns → method ── skips finished questions
        │
        │  for each question:
        │   1. shuffle the 4 options (seeded per question → same shuffle for every method/length)
        │   2. explicit only: model replies to the stated preference (5 tokens, exactly as upstream)
        │   3. build the prompt with the VERBATIM upstream code  (phase1/upstream/)
        ▼
phase1/backend.py ── Ollama /api/generate, raw=true, temperature 0, seed 41, num_ctx 8192
        │
        ▼
upstream extract_choice("<choice>B</choice>")  →  correct?   (+ a lenient parse kept as a diagnostic)
        │
        ▼
results/phase1/<model>/<form>/<method>/inter<N>/<topic>.jsonl   ← 1 line per question, fsync'd
        │
        ▼
phase1/aggregate.py → results/phase1/summary/{vs_paper.md, summary.csv, per_topic.csv, *.png}
```

**Why your earlier attempt kept erroring.** These are the root causes found in `prefeval.zip/logs/`:

| Error | Cause | How it's avoided now |
|---|---|---|
| `cannot import name 'genai' from 'google'` | upstream imports the Gemini SDK at top level | vendored copy makes it optional |
| `name 'random' is not defined` | upstream bug in `benchmark_classification.py` | new runner doesn't use that script |
| `Invalid model_type: local_vllm` | upstream has 3 copies of `generate_message`; the patch fixed only some | one backend; upstream prompt builders get Ollama through a single adapter |
| `UnboundLocalError: model_id` | duplicate `get_model_info` copies lacked the new model | runner reads `configs/phase1/models.yaml` |
| circular import in `utils/checkpoint.py` | – | not used |
| generation-task runs / "turns" 2, 10, 20 | generation needs a Bedrock Claude judge; 2/10/20 aren't paper columns | classification only; `inter_turns` 0/3/8 |

**Two subtle problems found and fixed while building this.** Both are recorded as `backend_rev: 2` in every result.
1. *Double BOS.* Upstream Llama prompts start with a literal `<|begin_of_text|>`, and Ollama adds another,
   so the model saw two (verified by token counts). The literal is stripped, so the model gets exactly one BOS, as on Bedrock.
2. *Empty self-critiques.* The upstream Llama self-critic templates are indented strings that end in
   `"<|end_header_id|>\n        "`. With that trailing whitespace Llama-3 ends its turn immediately, which produced empty
   critiques and about 0% accuracy. Trailing spaces/tabs are now trimmed. No other upstream prompt has trailing spaces, so nothing else changes.

### Deviations from the paper (state these in your report)
- **Quantization:** Q8_0 (llama.cpp) instead of bf16 on Bedrock. Expect a few points of difference.
- **Option shuffling:** a per-question seed instead of one global `random.seed(41)` stream. It is still random, but
  resume-safe and identical across methods, so method comparisons are paired.
- **Prompt hygiene:** the two whitespace/BOS fixes above. All prompt *text* is unchanged from upstream.
- **RAG on `entertain_games` (explicit) is skipped:** upstream's precomputed retrieval file
  `simcse_explicit_pref/entertain_games_overall300_topk_history.json` is truncated (broken since the
  authors' first commit; only 4 of 51 questions parse). RAG is therefore averaged over 19 topics. Zero-shot,
  Reminder, CoT and Self-Critic still use all 20. The file can't be rebuilt exactly, because index 1 is an assistant reply the
  authors never published. An approximate SimCSE rebuild is possible later if needed.
- **Topics:** the paper's Fig. 6 is averaged over 20 topics. Pilot numbers on 3 topics are indicative only.
- **Context window:** Ollama `num_ctx` is set to 8192 explicitly. Ollama's default would silently cut the *start*
  of long prompts, which is where the preference is. Every call logs its prompt token count, and any truncation is flagged.

## 4. Experiment plan and time budget

Measured on this machine (RTX 3060, Llama3-8B Q8): zero-shot, Reminder and CoT take **0.3–1.1 s/question**, RAG **1.3–2.6 s**,
Self-Critic **3–7 s** (3 calls). Questions per topic: travel_restaurant 56, lifestyle_dietary 57, entertain_shows 62.

| Step | Config | What | Questions | Expected time |
|---|---|---|---|---|
| 0 ✅ | `smoke` | all methods, explicit + implicit-choice, 5 q/cell | 90 | 2 min (done) |
| **1** | `tierA_pilot` | Zero-shot, Reminder, RAG; 3 topics | 1,400 | **~20 min** (allow 30) |
| **2** | `tierB_pilot` | CoT, Self-Critic; 3 topics | 1,050 | **~45 min** (allow 75) |
| ⏸ | – | **Review pilot results together** before scaling | | |
| **3** | `tierA` | same as step 1, all 20 topics (pilot cells skipped) | +6,600 | **~1.5 h** (allow 2.5) |
| **4** | `tierB` | same as step 2, all 20 topics | +4,950 | **~4 h** (allow 6) |
| opt. | `tierD_implicit` | implicit choice + persona baselines for Phase 3 | ~2,600 (3 topics) | ~45 min |
| opt. | `mistral_*` | Mistral-7B-v0.2, see §9 | | |

Steps 1–4 complete **Phase 1**: the full Llama3-8B column of Fig. 6 over 20 topics.

## 5. Commands

Everything runs from the repo root in the `prefeval` env:

```bash
cd ~/Documents/ML_Project_Krishna/PrefEval
conda activate prefeval
```

**First-time setup on a new machine.** This is already done on this machine.
```bash
conda create -y -n prefeval python=3.11 && conda activate prefeval
pip install -r requirements-phase1.txt
ollama pull llama3:8b-instruct-q8_0        # 8.5 GB
curl -s localhost:11434/api/version        # Ollama must be running (systemctl status ollama)
```

**Look before running** (no model calls):
```bash
python -m phase1.runner run --config configs/phase1/tierA_pilot.yaml --dry-run    # list cells + progress
python -m phase1.runner show-prompt --method remind --inter-turns 3               # exact prompt text
```

**Run.** Use tmux for anything over a few minutes:
```bash
scripts/phase1_tmux.sh tierA_pilot      # starts in tmux session "prefeval" (windows: run, gpu)
#   Ctrl-b d   detach (the run keeps going)  |  tmux attach -t prefeval   re-attach
#   Ctrl-b n   switch to the nvidia-smi window
```
Without tmux: `python -m phase1.runner run --config configs/phase1/tierA_pilot.yaml`

**Monitor, from any other terminal, while a run is going:**
```bash
python -m phase1.runner status --config configs/phase1/tierA_pilot.yaml   # per-cell progress, accuracy, ETA
tail -f logs/phase1/tierA_pilot_*.log                                      # per-question detail
```

**Resume after Ctrl-C, a crash, a reboot or a power cut:** run **the same command again**. Every finished
question is already on disk (one fsync'd line each), so at most the single in-flight question is redone.
Only one run at a time: the tmux script refuses to start a second session.

**Summarise:**
```bash
python -m phase1.aggregate                      # complete topic-cells only
python -m phase1.aggregate --include-partial    # include unfinished cells
# → results/phase1/summary/vs_paper.md, summary.csv, per_topic.csv, acc_vs_context_*.png
```

**Scale to all 20 topics after reviewing the pilot:**
```bash
scripts/phase1_tmux.sh tierA    # finished pilot cells are skipped automatically
scripts/phase1_tmux.sh tierB
```

## 6. Where everything is saved

```
results/phase1/                                  ← COMMIT THIS (small JSONL text)
├── run_manifest.jsonl          start/end of every run: config, git commit, Ollama version,
│                               model digest, backend_rev, status (completed/interrupted/aborted)
├── llama3-8b/<form>/<method>/inter<N>/<topic>.jsonl
│     one line per question: options (shuffled), correct letter, model reply to the preference,
│     raw answer, parsed choice (strict + lenient), correct?, critique (self-critic),
│     per-call latency / prompt tokens / truncation flag, timestamp
└── summary/                    regenerated by phase1.aggregate
logs/phase1/<config>_<timestamp>.log            ← git-ignored; DEBUG detail per question
```

## 7. When to push to GitHub

| Moment | Why |
|---|---|
| **Now** (code + smoke results) | the pipeline is verified, so the code is backed up |
| After each tier finishes (`tierA_pilot`, `tierB_pilot`, `tierA`, `tierB`) | results are the expensive part |
| Every ~1–2 h during long tiers (tierB) | the files are append-only, so committing mid-run is safe. You'd lose at most 1–2 h |
| Before editing anything in `phase1/` | so result-producing code versions are recoverable (the manifest stores the commit hash) |

```bash
# one-time: move to a branch and stop tracking the 60 MB zip
git switch -c phase1-local-ollama
git rm --cached prefeval.zip

# each checkpoint
python -m phase1.aggregate --include-partial
git add phase1 configs/phase1 scripts docs requirements-phase1.txt .gitignore CLAUDE.md results/phase1
git commit -m "Phase 1: <what finished>"
git push -u origin phase1-local-ollama
```

The old attempt's files (`run_experiments.py`, `*.patch`, `utils/checkpoint.py`, the edited `utils/`
and `classification_task/`) are **not used** by this pipeline and were left untouched. Whether to keep them is up to you.

## 8. Run history notes (corrections to `results/phase1/run_manifest.jsonl`)

- Model for all Phase 1 runs: `llama3:8b-instruct-q8_0`, Ollama digest
  `1b8e49cece7fb21deb006f8ced45ce18c54010941dbdf25fb37ecb3cc40dc798` (8.54 GB). Manifest entries before
  2026-10-02 22:00 show `digest: null`, because Ollama 0.34's `/api/show` doesn't return it. Fixed; it is now read from `/api/tags`.
- The `tierA` run ending 2026-10-02 18:55:40 is logged as `completed` but actually **crashed** on the truncated upstream
  `entertain_games` RAG file (see §2 deviations). The runner now logs such cases as `crashed`.
  All 3,263 questions from that run are valid, and the run resumed at 19:44 and completed at 21:35.
- The `tierA` start at 19:40:43 has no end entry: that session was killed before it started work. A killed
  tmux session gives the runner no chance to write an end entry.
- Tier A totals: 7,898 questions, about 197 min of compute across both runs, 0 truncated prompts.

## 9. Second model: Mistral-7B-Instruct-v0.2 (`mistral:7b-instruct-v0.2-q8_0`)

This is the paper's "Mistral 7b" (Bedrock `mistral.mistral-7b-instruct-v0:2`). Fig. 6 gives baselines for all 5 methods from 0.2k up to
23k tokens. Its 32k context gives the proposal's **long regime**, which Llama3-8B's 8k context can't reach.

Setup findings (smoke test, 2026-10-03):
- **Same double-BOS issue as Llama.** Upstream Mistral prompts start with a literal `<s>`, and Ollama adds BOS too
  ("hello" → 3 tokens, "<s>hello" → 4). The literal is stripped, which is the same `backend_rev 2` rule as Llama.
- 0 parse failures and proper self-critiques. The upstream Mistral templates aren't indented, so the empty-revision issue doesn't occur.
- **Long-context size:** 68 inter turns (paper's 23k row) = 26.9k–27.7k Mistral tokens, so it fits in a 32k context without truncation.
- **Measured speed with the setting below (100% GPU, ~2,000 tokens/s):** 4k-token prompt 2.6 s, 12k 5.3 s, 27k 13.9 s,
  RAG at 27.7k 21 s per call.
- **GPU memory:** at 32k context with the default fp16 KV cache the model needs about 12 GB and spills 10% onto the CPU,
  giving 11–695 s per question. Fix: flash attention + 8-bit KV cache (`OLLAMA_FLASH_ATTENTION=1`,
  `OLLAMA_KV_CACHE_TYPE=q8_0`), which roughly halves context memory. **All Mistral tiers run with this setting**
  (it is recorded in the manifest as `ollama_env`), so short and long Mistral results are consistent. The Llama runs used Ollama defaults.
  **Guard:** `configs/phase1/models.yaml` stores each model's required setting, and the runner refuses to start on a
  mismatch and prints the switch command. This applies to Phase 3 too, so baseline and new method always run under identical settings.

| Tier | Config | What | Questions | Est. time |
|---|---|---|---|---|
| M-A | `mistral_tierA` | Zero-shot, Reminder, RAG; 0.2k/1k/3k; 20 topics | ~7,900 | ~4 h |
| M-B | `mistral_tierB` | CoT, Self-Critic; 0.2k/1k/3k; 20 topics | ~6,000 | ~7 h |
| M-C pilot | `mistral_tierC_long_pilot` | Zero-shot, Reminder, RAG; 10k/16k/23k; 3 topics | ~1,575 | ~4 h (allow 6) |
| M-C full | `mistral_tierC_long` | same, 20 topics (pilot cells skipped) | ~9,000 | ~22 h, across several nights |

Mixtral-8x7B (also in the paper) was ruled out: Q4 is 28.4 GB, which is more than 12 GB VRAM plus ~10 GB free RAM. Q2/Q3 would fit
only with heavy CPU offload and degraded quality, so the results wouldn't be comparable to the paper.

## 10. Implicit-choice baselines (own baselines for Phase 3)

The paper reports implicit preferences (choice-based, persona-driven) **only for the generation task** (Fig. 5,
Tables 8–13), judged by Claude 3 Sonnet, which isn't reproducible locally. Its classification results (Fig. 6) are explicit-only.
The proposal scopes Phase 3 to explicit **and implicit-choice**, so Phase 1 also produces implicit-choice
classification baselines with the same validated pipeline. They are not compared against the paper; they are the reference for Phase 3.

| Config | Model | What | Questions | Est. time | Ollama setting |
|---|---|---|---|---|---|
| `mistral_implicitA` | Mistral-7B | implicit-choice; Zero-shot, Reminder, RAG; 0.2k/1k/3k; 20 topics | 8,000 | ~4 h | FA + q8 KV (Mistral's) |
| `llama_implicitA` | Llama3-8B | same | 8,000 | ~3–4 h | Ollama defaults (Llama's) |

All 20 implicit-choice RAG files parse, but **in 7 topics they stop a few questions early** (upstream data issue,
found 2026-10-06): travel_restaurant 53/56, entertain_shows 60/62, pet_ownership 35/43, lifestyle_health 45/49,
education_learning_styles 26/31, shop_technology 32/35, travel_hotel 52/54. The implicit-persona files and the
question/inter-turn similarity file are short in the same way. The entries that exist are correctly aligned with the first questions (checked).
So implicit RAG is evaluated on **973 of 1,000 questions**, and the other methods on all 1,000. `vs_paper.md` notes this with ‡.
(The first Mistral run logged 54 `IndexError`s for these questions. The runner now counts only questions with retrieval data.) The old `tierD_implicit.yaml` (3 hard pilot topics + persona) was
replaced by these two configs. Implicit-persona, CoT/Self-Critic and long context for implicit can be added later if Phase 3 needs them.
