# Phase 1: our local reproduction vs. paper (Figure 6, classification task)

Accuracy in %, macro-averaged over topics (as in the paper). Paper numbers are averaged over **all 20 topics**, so with fewer topics the comparison is only indicative.

Backend revision: 2. Paper: Zhao et al., ICLR 2025, Fig. 6.

## llama3-8b | explicit

| Method | Context (inter_turns) | Ours | Paper | Δ (ours − paper) | Topics | Questions | Parse-fail % | Truncated |
|---|---|---|---|---|---|---|---|---|
| Zero-shot | 0.2k (0) | 94.2 | 85 | +9.2 | 20/20 | 1000 | 0.0 | 0 |
| Zero-shot | 1k (3) | 52.7 | 55 | -2.3 | 20/20 | 1000 | 0.0 | 0 |
| Zero-shot | 3k (8) | 43.3 | 42 | +1.3 | 20/20 | 1000 | 0.1 | 0 |
| Reminder | 0.2k (0) | 97.4 | 94 | +3.4 | 20/20 | 1000 | 0.0 | 0 |
| Reminder | 1k (3) | 78.0 | 80 | -2.0 | 20/20 | 1000 | 0.0 | 0 |
| Reminder | 3k (8) | 59.4 | 59 | +0.4 | 20/20 | 1000 | 0.1 | 0 |
| RAG (top-5) | 1k (3) | 88.6 | 87 | +1.6 | 19/19† | 949 | 5.9 | 0 |
| RAG (top-5) | 3k (8) | 74.8 | 80 | -5.2 | 19/19† | 949 | 19.1 | 0 |
† RAG excludes entertain_games: the upstream precomputed SimCSE retrieval file is truncated (only 4/51 questions parse), so RAG is averaged over the remaining topics.

