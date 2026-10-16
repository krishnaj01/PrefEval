# Phase 2: Preference-detection classifier

**Question the classifier answers:** *does this user message state a preference?*
In Phase 3 it decides which user turns are written to the structured preference memory. Phase 2 trains and compares
**four models** (DistilBERT, ModernBERT, TinyLlama-1.1B, Qwen3-1.7B) and **every ensemble of them**, with a fully
reproducible pipeline. Branch: `phase2-classifier`. Environment: conda env `prefeval-p2` (`requirements-phase2.txt`).

---

## 1. Task definition

| | |
|---|---|
| **Unit** | one **user message** (a single turn written by the user) |
| **Model input (the only feature)** | the raw text of that message. No assistant turns, no history, no topic, no metadata. Tokenised with each model's own tokenizer and truncated to 256 tokens; PrefEval messages are ≤ ~130 tokens |
| **Model output** | `p = P(label = 1 \| text)` in [0, 1]; prediction = `1 if p ≥ τ` |
| **Threshold τ** | tuned per fold on the **validation** set to maximise F1, stored in `metrics.json` and next to the saved model |
| **Label 1** | the message reveals a personal **preference, dislike, restriction or constraint** of the user that should shape future recommendations, stated directly *or* implied by a choice or passing remark |
| **Label 0** | anything else: plain requests, questions, instructions, greetings, thanks, small talk |

**Why text only?** In Phase 3 the classifier sees each user turn as it arrives, so training uses exactly that input.

## 2. Data

### 2.1 Sources and how each row gets its label
| Source (`source`) | Where it comes from | Label | Label origin | Example |
|---|---|---|---|---|
| `explicit_pref` | PrefEval explicit preference sentence | 1 | gold | "I avoid hotels with a bustling nightlife atmosphere." |
| `persona_pref` | PrefEval persona `preference` field (mostly identical to explicit, so deduplicated) | 1 | gold | |
| `choice_selection` | user turn in a choice-based dialogue where the user picks an option | 1 | gold | "I'll go with The Westin… the other options seem a bit too lively for my taste." |
| `choice_query` | first user turn of a choice-based dialogue | 0 | gold | "What are some good hotels to stay at in Las Vegas?" |
| `question` | PrefEval final question | 0 | gold | "Can you suggest some great hotels for my upcoming trip to Las Vegas?" |
| `persona_turn` | each user turn of a persona dialogue | 1 for the turn revealing the dialogue's known preference, 0 for the others | **pseudo (LLM)** | "…As someone who follows a strict halal diet, I bring my own food…" → 1 |
| `oasst` | 5,000 English user messages from OpenAssistant `oasst1` (pinned revision) | YES/NO from the LLM | **pseudo (LLM)** | "What are the seven deadly sins?" → 0 |
| `lmsys_ood` | the 316 LMSYS distractor user turns | YES/NO from the LLM | **evaluation only, never trained on** | |

- **Pseudo-labeller:** Llama-3-8B-Instruct (Q8) via Ollama, temperature 0, under its Phase 1 server settings (enforced by the settings guard).
  It is a **one-time offline annotator**. The classifiers learn from its labels, and Phase 3 never calls an LLM for detection.
- **Persona cross-check:** the dataset doesn't mark which persona turn holds the preference. Besides the LLM's pick, an LLM-free check
  records the turn with the most word overlap with the known preference.
  - Their agreement rate is reported.
  - Disagreements form their own stratum in the manual audit.
  - *Finding:* PrefEval's authors put the preference in user turn 2 for about 65% of dialogues (turn 3 for about 23%). The classifier never sees turn positions, so this cannot become a shortcut.
- **Manual audit:** 200 stratified rows (40 per stratum: persona +, persona −, persona disagreements, OASST +, OASST −).
  - The reviewer fills in `your_label`.
  - Agreement and Cohen's κ are reported.
  - Audited labels override pseudo-labels (`label_origin = audited`).

### 2.2 Row schema (`data/phase2/*.jsonl`)
| Field | Meaning |
|---|---|
| `uid` | stable id, e.g. `prefeval/travel_hotel/12/persona_turn/3`, `oasst/<message_id>` |
| `text` | **model input** |
| `label` | **target** (0/1) |
| `label_origin` | `gold` / `pseudo` / `audited` |
| `source`, `form` | for per-source and per-form analysis (`explicit`, `implicit-choice`, `implicit-persona`, `chat`, `none`) |
| `topic`, `task_id`, `turn_idx` | traceability to the original PrefEval item (`topic = null` for OASST / LMSYS) |
| `group`, `fold` | splitting unit (topic or OASST conversation tree) and its cross-validation fold |

**Metadata is never a model input.** It exists for splitting, Phase 3 routing and error analysis.

### 2.3 Splits: topic-level 5-fold cross-validation
- The 20 topics are shuffled with seed 42 into 5 folds of 4 topics each. OASST rows go to `sha256(conversation tree id) % 5`.
- In round *k*: **test = fold k, validation = fold k+1, train = the other 3**.
- **Why by topic?** PrefEval contains many near-identical sentences within a topic. A random split would put near-copies of
  training sentences in the test set and inflate scores. Testing on unseen topics measures real generalisation.
- Phase 3 uses each round's model only on that round's test topics, so all 20 topics are covered with no leakage.
- **Asserted in code:** no topic or group is shared between train, validation and test in any round, and no LMSYS row is ever in the training data.
- Topic-bias check: the per-topic positive rate is reported in the data card (0.52–0.61 on gold data), and so is the per-topic F1 spread.

## 3. Models

| Name | Hugging Face model | Type | Training | Key hyperparameters |
|---|---|---|---|---|
| `distilbert` | `distilbert/distilbert-base-uncased` (66M) | encoder | full fine-tune, fp32 | lr 3e-5, 4 epochs, batch 32 |
| `modernbert` | `answerdotai/ModernBERT-base` (149M) | encoder | full fine-tune, bf16 | lr 5e-5, 4 epochs, batch 32 |
| `tinyllama` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | decoder | LoRA r=16 (all projections) + classification head, bf16 | lr 2e-4, 2 epochs, batch 16×2 |
| `qwen3` | `Qwen/Qwen3-1.7B` | decoder | same | same |

- Decoders classify from the hidden state of the **last** non-padding token.
- The validation set is checked 4× per epoch; early stopping uses validation F1 (patience 4) and keeps the best checkpoint.
- The exact Hugging Face commit of every model is resolved and stored.
- Smoke-test footprint: Qwen3 LoRA uses 3.7 GB of GPU memory and trains 17M of 1.7B parameters.

## 4. Pipeline and commands

Run from the repo root (`conda activate prefeval-p2`). Every step is resumable; finished work is skipped.

| Step | Command | What it does | Time |
|---|---|---|---|
| 1 | `python -m phase2.build_data` | PrefEval → `prefeval_turns.jsonl`, `lmsys_ood.jsonl` (deduplicated) | seconds |
| 2 | `python -m phase2.pseudolabel --what persona` | LLM picks each persona dialogue's preference turn, plus the overlap cross-check | ~5–10 min |
| 3 | `python -m phase2.pseudolabel --what oasst` | samples 5,000 OASST messages (pinned) + YES/NO labels | ~20–40 min |
| 3b | `python -m phase2.pseudolabel --what lmsys` | YES/NO labels for the LMSYS evaluation set | ~2 min |
| 4 | `python -m phase2.pseudolabel --audit` → fill in `data/phase2/audit_sample.csv` → `--audit-report` | manual label-quality check | your time |
| 5 | `python -m phase2.splits` | final `dataset.jsonl` + folds + data card | seconds |
| 6 | `python -m phase2.train --config configs/phase2/<model>.yaml --fold all` | 5 folds per model | encoders ~15–40 min, decoders ~1–2.5 h |
| 7 | `python -m phase2.llm_baseline` | zero-shot Llama-3-8B on 2,000 PrefEval rows (accuracy + latency) | ~15 min |
| 8 | `python -m phase2.evaluate` | all metrics, breakdowns and error analysis | seconds |
| 9 | `python -m phase2.ensemble` | 15 subsets × {soft, hard, stacking} + diversity | seconds |
| 10 | `python -m phase2.report` | all figures and tables | seconds |

- `python -m phase2.pipeline --stage {data|pseudo|splits|train|baseline|analyze|all}` chains the steps.
- `scripts/phase2_tmux.sh "<command>"` runs any of them inside tmux.
- **Ollama settings:** steps 2–3 and 7 need Ollama with Llama-3's default settings (see the switch command in the guard message). Training needs the GPU free: the `train` stage unloads Ollama models automatically.

## 5. Outputs

```
data/phase2/                     (committed: small text files)
  prefeval_turns.jsonl, lmsys_ood.jsonl, oasst_sample.jsonl, pseudo/*.jsonl (LLM label caches)
  dataset.jsonl, splits.json, data_card.json, audit_sample.csv, audit_report.json
  manifest.json                  sha256 + dataset revision + labeller model digest for every artefact
results/phase2/                  (committed)
  <model>/fold<k>/metrics.json, log_history.csv, predictions_{val,test,ood}.jsonl
  llm_baseline/                  predictions + metrics
  eval/                          summary, per_fold, per_source, per_topic, llm_comparison, curves.json, errors/<model>.md
  ensemble/                      ensembles.csv, diversity.csv
  figures/, tables/              everything for the report
  run_manifest.jsonl             every train/evaluate run: git commit, package versions, GPU, model revision, timings
models/phase2/<model>/fold<k>/best   saved model / LoRA adapter + threshold (git-ignored, several GB)
```

### Metrics reported
- Pooled over the 5 test folds, plus mean ± std across folds: accuracy, precision, recall, F1, macro-F1, ROC-AUC, PR-AUC, confusion matrix, expected calibration error.
- Breakdowns: recall per positive source and false-positive rate per negative source; F1 per topic.
- LMSYS out-of-domain flag rate.
- Latency (ms per message, batch 1, GPU), parameters (total and trainable), training time, peak GPU memory.
- Loss and validation-F1 curves.
- The most confident errors, for error analysis.
- The same-sample comparison with the LLM-prompted detector.
- Ensembles: all 15 subsets with three combination rules, pairwise error correlation, and accuracy vs cost.

## 6. Reproducibility
- Global seed 42 (Python, NumPy, torch, HF `set_seed`, `data_seed`).
  - Verified: two identical runs give **identical** probabilities (max difference 0.0).
- Pinned: the model and dataset Hugging Face revisions, the labeller model digest, prompt hashes in every pseudo-label row, and exact package versions (bottom of `requirements-phase2.txt`).
- Data files and splits are hashed in `data/phase2/manifest.json`.
- Every reported number is computed from the saved predictions by `evaluate` / `ensemble` / `report`.

## 7. Notes from implementation
- `transformers` 5.x API changes handled: `warmup_ratio` → computed `warmup_steps`; `torch_dtype` → `dtype`; ModernBERT no longer takes `reference_compile`.
- Smoke test (gold labels only): models trained without persona / OASST data flagged 24–92% of real LMSYS chat turns as preferences.
  This is the motivation for the pseudo-labelled realistic negatives. The out-of-domain flag rate tracks it in the final runs.
