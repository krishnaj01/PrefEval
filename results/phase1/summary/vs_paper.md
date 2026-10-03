# Phase 1: our local reproduction vs. paper (Figure 6, classification task)

Accuracy in %, macro-averaged over topics (as in the paper). Paper numbers are averaged over **all 20 topics**, so with fewer topics the comparison is only indicative.

Backend revision: 2. Paper: Zhao et al., ICLR 2025, Fig. 6.

## llama3-8b | explicit

| Method | Context (inter_turns) | Ours | Paper | Δ (ours − paper) | Topics | Questions | Parse-fail % | Truncated |
|---|---|---|---|---|---|---|---|---|
| Zero-shot | 0.2k (0) | 94.6 | 85 | +9.6 | 8/20 ⚠ partial | 411 | 0.0 | 0 |
| Zero-shot | 1k (3) | 50.2 | 55 | -4.8 | 8/20 ⚠ partial | 411 | 0.0 | 0 |
| Zero-shot | 3k (8) | 39.6 | 42 | -2.4 | 8/20 ⚠ partial | 411 | 0.0 | 0 |
| Reminder | 0.2k (0) | 97.2 | 94 | +3.2 | 8/20 ⚠ partial | 411 | 0.0 | 0 |
| Reminder | 1k (3) | 75.4 | 80 | -4.6 | 8/20 ⚠ partial | 411 | 0.0 | 0 |
| Reminder | 3k (8) | 57.1 | 59 | -1.9 | 8/20 ⚠ partial | 411 | 0.0 | 0 |
| CoT | 0.2k (0) | 80.0 | 83 | -3.0 | 1/20 ⚠ partial | 5 | 0.0 | 0 |
| CoT | 3k (8) | 60.0 | 50 | +10.0 | 1/20 ⚠ partial | 5 | 0.0 | 0 |
| Self-Critic | 0.2k (0) | 80.0 | 79 | +1.0 | 1/20 ⚠ partial | 5 | 0.0 | 0 |
| Self-Critic | 3k (8) | 60.0 | 62 | -2.0 | 1/20 ⚠ partial | 5 | 0.0 | 0 |
| RAG (top-5) | 1k (3) | 88.5 | 87 | +1.5 | 8/19 ⚠ partial† | 411 | 6.6 | 0 |
| RAG (top-5) | 3k (8) | 77.6 | 80 | -2.4 | 8/19 ⚠ partial† | 411 | 16.5 | 0 |
† RAG excludes entertain_games: the upstream precomputed SimCSE retrieval file is truncated (only 4/51 questions parse), so RAG is averaged over the remaining topics.

## llama3-8b | implicit-choice

| Method | Context (inter_turns) | Ours | Paper | Δ (ours − paper) | Topics | Questions | Parse-fail % | Truncated |
|---|---|---|---|---|---|---|---|---|
| Zero-shot | 0.2k (0) | 60.0 | – | – | 1/20 ⚠ partial | 5 | 0.0 | 0 |
| Zero-shot | 3k (8) | 40.0 | – | – | 1/20 ⚠ partial | 5 | 0.0 | 0 |
| Reminder | 0.2k (0) | 80.0 | – | – | 1/20 ⚠ partial | 5 | 0.0 | 0 |
| Reminder | 3k (8) | 40.0 | – | – | 1/20 ⚠ partial | 5 | 0.0 | 0 |
| CoT | 0.2k (0) | 60.0 | – | – | 1/20 ⚠ partial | 5 | 0.0 | 0 |
| CoT | 3k (8) | 20.0 | – | – | 1/20 ⚠ partial | 5 | 0.0 | 0 |
| Self-Critic | 0.2k (0) | 80.0 | – | – | 1/20 ⚠ partial | 5 | 0.0 | 0 |
| Self-Critic | 3k (8) | 20.0 | – | – | 1/20 ⚠ partial | 5 | 0.0 | 0 |
| RAG (top-5) | 3k (8) | 60.0 | – | – | 1/20 ⚠ partial | 5 | 0.0 | 0 |

