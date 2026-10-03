# Phase 1: our local reproduction vs. paper (Figure 6, classification task)

Accuracy in %, macro-averaged over topics (as in the paper). Paper numbers are averaged over **all 20 topics**, so with fewer topics the comparison is only indicative.

Backend revision: 2. Paper: Zhao et al., ICLR 2025, Fig. 6.

## llama3-8b | explicit

| Method | Context (inter_turns) | Ours | Paper | Δ (ours − paper) | Topics | Questions | Parse-fail % | Truncated |
|---|---|---|---|---|---|---|---|---|
| Zero-shot | 0.2k (0) | 100.0 | 85 | +15.0 | 1/20 ⚠ partial | 5 | 0.0 | 0 |
| Zero-shot | 3k (8) | 40.0 | 42 | -2.0 | 1/20 ⚠ partial | 5 | 0.0 | 0 |
| Reminder | 0.2k (0) | 100.0 | 94 | +6.0 | 1/20 ⚠ partial | 5 | 0.0 | 0 |
| Reminder | 3k (8) | 60.0 | 59 | +1.0 | 1/20 ⚠ partial | 5 | 0.0 | 0 |
| CoT | 0.2k (0) | 80.0 | 83 | -3.0 | 1/20 ⚠ partial | 5 | 0.0 | 0 |
| CoT | 3k (8) | 60.0 | 50 | +10.0 | 1/20 ⚠ partial | 5 | 0.0 | 0 |
| Self-Critic | 0.2k (0) | 80.0 | 79 | +1.0 | 1/20 ⚠ partial | 5 | 0.0 | 0 |
| Self-Critic | 3k (8) | 60.0 | 62 | -2.0 | 1/20 ⚠ partial | 5 | 0.0 | 0 |
| RAG (top-5) | 3k (8) | 20.0 | 80 | -60.0 | 1/20 ⚠ partial | 5 | 80.0 | 0 |

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

