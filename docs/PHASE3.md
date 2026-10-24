# Phase 3: preference memory + modern retrieval

Goal: use the Phase 2 classifier to build a preference memory, retrieve it with a modern embedding model, put it in the
prompt, and compare with the Phase 1 baselines. Phase 3 is built step by step:

1. **Retrieval benchmark** (this document; no LLM calls), to choose the encoder.
2. Prompt format and memory design (to be decided after step 1).
3. End-to-end MCQ runs against the Phase 1 baselines.

Environment: `conda activate prefeval-p2` plus `pip install -r requirements-phase3.txt`. Phase 1, Phase 2 and their results are
only *read*, never modified.

## Step 1: retrieval benchmark

**Question:** if we search the conversation with the final question, does the message that states the preference come back?

| | Definition |
|---|---|
| Query | the final MCQ question (`mcq_options/<topic>.json`), exactly as in Phase 1 |
| Candidates | every message *before* the question, one message = one retrieval unit (as in the paper) |
| Explicit pool | [preference, the LLM's own reply to it] + the first 2N LMSYS messages, so 2 + 2N messages |
| Implicit-choice pool | [query, assistant_options, user_selection, assistant_acknowledgment] + 2N LMSYS messages, so 4 + 2N |
| Gold | the explicit preference sentence, or `user_selection` (the turn where the user reveals the choice) |
| N (inter_turns) | 3, 8, 28, 48, 68, which are the paper's 1k, 3k, 10k, 16k and 23k token rows |
| Reply | taken from Phase 1 (`response_to_pref`, the same in every method and length). Explicit pools are scored per LLM: `explicit@llama3-8b`, `explicit@mistral7b` |

**Retrievers**

| Name | Model | Params | Query prompt / document prompt |
|---|---|---|---|
| `simcse_paper` | princeton-nlp/sup-simcse-roberta-large (the paper's) | 355M | scores stored upstream in `benchmark_dataset/rag_retrieval/` |
| `bge_base` | BAAI/bge-base-en-v1.5 | 110M | bge retrieval instruction / none |
| `e5_base` | intfloat/e5-base-v2 | 110M | `query: ` / `passage: ` |
| `bge_large` | BAAI/bge-large-en-v1.5 | 335M | bge retrieval instruction / none |
| `mxbai_large` | mixedbread-ai/mxbai-embed-large-v1 | 335M | its ST `query` prompt / none |
| `embeddinggemma2` | google/embeddinggemma-2 | ~700M | `task: search result \| query: ` / `title: none \| text: ` |

Exact commits and prompts are in `configs/phase3/encoders.yaml`.

**How the encoders run:**
- They run **locally**, in the Python process, on the GPU, through PyTorch and sentence-transformers. **Ollama is not involved**: it serves only the chat LLMs.
- Each model is downloaded once, at its pinned commit, into `~/.cache/huggingface/hub/`. Later runs load that folder with no network access.

**Metrics**
- recall@1/3/5/10 of the gold message
- MRR and mean rank
- `distractor_share@5`: the share of the top 5 that are LMSYS messages. This is what a top-5 RAG prompt is filled with instead of the preference.
- cost: parameters, and ms to embed one message (batch 1)

**Comparison subset:** `common` covers the questions that also have stored SimCSE scores: 949 explicit (upstream's
`entertain_games` explicit file is truncated) and 973 implicit. This is the same coverage as Phase 1 RAG. The encoders are also reported on all 1,000 (`all`).

**Is pre-computing fair?** Yes. A bi-encoder embeds every message on its own, so a vector computed in advance is identical
to one computed when the message arrives. Causality holds by construction, because only earlier messages are candidates. The real-time
cost is reported as per-message latency.

**Known caveat of the paper's SimCSE scores:** the explicit *reply* score was computed by the authors on their own reply text,
while the reply placed in the prompt is the live 5-token LLM reply. For the encoders we embed the exact reply each LLM produced.

### Verification (done)
- `python -m phase3.conversations`: pool sizes, exactly one gold per pool, and alignment with the stored SimCSE scores (asserted).
- `python -m phase3.checks`: our SimCSE top-5 equals the top-5 that upstream's `handle_rag_task_mcq` /
  `handle_rag_task_implicit_mcq` put in the prompt, for all 9,610 (question, length) pairs.
- `python -m phase3.embed --encoder <e> --check`: re-encoding matches the cache (max diff ~2e-7, float noise).

### Commands (repo root)
```bash
conda activate prefeval-p2
python -m phase3.embed --encoder all       # downloads ~4-5 GB once, then ~1-2 min per encoder; resumable
python -m phase3.retrieve --encoder all    # SimCSE + 5 encoders, about 1 min
python -m phase3.retrieval_eval            # summary.csv / summary.md / figures
```
Or all three in tmux (session `phase3`, log in `logs/phase3/`):
`scripts/phase3_tmux.sh "python -m phase3.embed --encoder all && python -m phase3.retrieve --encoder all && python -m phase3.retrieval_eval"`
Smoke or partial runs must go to the smoke folder:
`PHASE3_RESULTS_DIR=results/phase3_smoke python -m phase3.retrieve --encoder bge_base --topics travel_hotel --inter-turns 3,68`.

### Outputs
| Path | Content | Git |
|---|---|---|
| `data/phase3/emb/<encoder>/` | `shard_*.npz` (sha256(text), unit vector) + `meta.json` (commit, prompts, dim, latency) | ignored (reproducible) |
| `results/phase3/retrieval/<encoder>/<pool>/inter<N>.jsonl` | per question: gold id and rank, top-10 ids, sources and scores | committed |
| `results/phase3/retrieval/summary.{csv,md}`, `figures/` | metrics tables and plots | committed |
| `results/phase3/run_manifest.jsonl` | time, git state, versions, per step | committed |

The stored top-10 rankings are also the input for the later RAG and memory prompt steps.
