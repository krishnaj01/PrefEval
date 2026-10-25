# Phase 3 vs Phase 1 baselines (classification accuracy %, macro over topics)

`all` = each method's full coverage (Phase 1 RAG: questions with stored SimCSE scores); RAG rows at the bottom of each table are on identical questions (`rag_set`).

## llama3-8b | explicit

| Method | 0.2k | 1k | 3k |
|---|---|---|---|
| Zero-shot | 94.2 | 52.7 | 43.3 |
| Reminder | 97.4 | 78.0 | 59.4 |
| CoT | 93.7 | 78.0 | 63.9 |
| Self-Critic | 78.8 | 50.1 | 46.7 |
| RAG (SimCSE, paper) | – | 88.6 | 74.8 |
| Memory (ours) | 97.7 | 96.0 | 96.5 |
| RAG (EmbeddingGemma-2, ours) | – | 94.7 | 93.9 |
| Memory + RAG (ours) | – | 93.4 | 92.9 |
| **same questions (rag_set)** |  |  |  |
| RAG (SimCSE, paper) | – | 88.6 | 74.8 |
| RAG (EmbeddingGemma-2, ours) | – | 94.6 | 93.9 |
| Memory + RAG (ours) | – | 93.2 | 92.7 |

Phase 3 diagnostics:

| method     |   inter_turns |       n |   topics |   parse_fail_pct |   acc_lenient |   truncated |   sec_per_q |
|:-----------|--------------:|--------:|---------:|-----------------:|--------------:|------------:|------------:|
| memory     |             0 | 1000.00 |    20.00 |             0.00 |         97.71 |        0.00 |        0.35 |
| memory     |             3 | 1000.00 |    20.00 |             0.00 |         96.02 |        0.00 |        0.89 |
| memory     |             8 | 1000.00 |    20.00 |             0.00 |         96.48 |        0.00 |        1.88 |
| memory_rag |             3 | 1000.00 |    20.00 |             3.30 |         93.42 |        0.00 |        1.08 |
| memory_rag |             8 | 1000.00 |    20.00 |             4.10 |         92.86 |        0.00 |        2.04 |
| rag_gemma  |             3 | 1000.00 |    20.00 |             0.30 |         94.68 |        0.00 |        1.03 |
| rag_gemma  |             8 | 1000.00 |    20.00 |             1.10 |         93.85 |        0.00 |        2.01 |

## llama3-8b | implicit-choice

| Method | 0.2k | 1k | 3k |
|---|---|---|---|
| Zero-shot | 77.0 | 53.3 | 46.0 |
| Reminder | 91.0 | 77.0 | 60.3 |
| CoT | 84.4 | 70.3 | 61.7 |
| Self-Critic | 66.1 | 50.1 | 38.7 |
| RAG (SimCSE, paper) | – | 82.8 | 73.7 |
| Memory (ours) | 90.0 | 86.3 | 83.5 |
| RAG (EmbeddingGemma-2, ours) | – | 83.6 | 79.1 |
| Memory + RAG (ours) | – | 84.2 | 75.4 |
| **same questions (rag_set)** |  |  |  |
| RAG (SimCSE, paper) | – | 82.8 | 73.7 |
| RAG (EmbeddingGemma-2, ours) | – | 84.1 | 79.6 |
| Memory + RAG (ours) | – | 84.5 | 75.8 |

Phase 3 diagnostics:

| method     |   inter_turns |       n |   topics |   parse_fail_pct |   acc_lenient |   truncated |   sec_per_q |
|:-----------|--------------:|--------:|---------:|-----------------:|--------------:|------------:|------------:|
| memory     |             0 | 1000.00 |    20.00 |             0.00 |         90.01 |        0.00 |        0.41 |
| memory     |             3 | 1000.00 |    20.00 |             0.00 |         86.32 |        0.00 |        0.97 |
| memory     |             8 | 1000.00 |    20.00 |             0.10 |         83.51 |        0.00 |        2.01 |
| memory_rag |             3 | 1000.00 |    20.00 |             0.40 |         84.19 |        0.00 |        1.13 |
| memory_rag |             8 | 1000.00 |    20.00 |             7.90 |         75.37 |        0.00 |        2.18 |
| rag_gemma  |             3 | 1000.00 |    20.00 |             0.10 |         83.63 |        0.00 |        1.07 |
| rag_gemma  |             8 | 1000.00 |    20.00 |             0.30 |         79.08 |        0.00 |        2.09 |
