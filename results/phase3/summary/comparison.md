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

## mistral7b | explicit

| Method | 0.2k | 1k | 3k | 10k | 16k | 23k |
|---|---|---|---|---|---|---|
| Zero-shot | 95.5 | 61.5 | 55.5 | 42.5 | 37.1 | 37.6 |
| Reminder | 96.2 | 78.9 | 71.3 | 56.7 | 43.9 | 46.9 |
| CoT | 88.0 | 74.9 | 67.0 | – | – | – |
| Self-Critic | 79.8 | 67.0 | 61.0 | – | – | – |
| RAG (SimCSE, paper) | – | 81.0 | 75.7 | 64.6 | 59.0 | 53.7 |
| Memory (ours) | 97.1 | 94.7 | 94.9 | 93.7 | 92.4 | 88.6 |
| RAG (EmbeddingGemma-2, ours) | – | 85.3 | 86.1 | – | – | – |
| Memory + RAG (ours) | – | 94.2 | 94.4 | 94.1 | 90.6 | 87.5 |
| **same questions (rag_set)** |  |  |  |  |  |  |
| RAG (SimCSE, paper) | – | 81.0 | 75.7 | 64.6 | 59.0 | 53.7 |
| RAG (EmbeddingGemma-2, ours) | – | 85.5 | 86.3 | – | – | – |
| Memory + RAG (ours) | – | 94.2 | 94.5 | 94.3 | 90.8 | 88.3 |

Phase 3 diagnostics:

| method     |   inter_turns |       n |   topics |   parse_fail_pct |   acc_lenient |   truncated |   sec_per_q |
|:-----------|--------------:|--------:|---------:|-----------------:|--------------:|------------:|------------:|
| memory     |             0 | 1000.00 |    20.00 |             0.00 |         97.15 |        0.00 |        0.41 |
| memory     |             3 | 1000.00 |    20.00 |             0.00 |         94.70 |        0.00 |        1.03 |
| memory     |             8 | 1000.00 |    20.00 |             0.00 |         94.90 |        0.00 |        2.14 |
| memory     |            28 | 1000.00 |    20.00 |             0.00 |         93.70 |        0.00 |        7.68 |
| memory     |            48 | 1000.00 |    20.00 |             0.00 |         92.42 |        0.00 |       13.56 |
| memory     |            68 | 1000.00 |    20.00 |             0.00 |         88.35 |        0.00 |       20.99 |
| memory_rag |             3 | 1000.00 |    20.00 |             0.00 |         94.22 |        0.00 |        0.95 |
| memory_rag |             8 | 1000.00 |    20.00 |             0.00 |         94.38 |        0.00 |        2.40 |
| memory_rag |            28 | 1000.00 |    20.00 |             0.00 |         94.09 |        0.00 |        8.33 |
| memory_rag |            48 | 1000.00 |    20.00 |             0.00 |         90.64 |        0.00 |       13.82 |
| memory_rag |            68 | 1000.00 |    20.00 |             0.00 |         87.33 |        0.00 |       21.26 |
| rag_gemma  |             3 | 1000.00 |    20.00 |             0.00 |         85.26 |        0.00 |        0.67 |
| rag_gemma  |             8 | 1000.00 |    20.00 |             0.00 |         86.09 |        0.00 |        2.36 |

## mistral7b | implicit-choice

| Method | 0.2k | 1k | 3k |
|---|---|---|---|
| Zero-shot | 81.0 | 50.8 | 47.5 |
| Reminder | 88.4 | 71.9 | 63.1 |
| CoT | 74.0 | 60.2 | 59.5 |
| Self-Critic | 67.2 | 51.8 | 50.9 |
| RAG (SimCSE, paper) | – | 69.0 | 64.0 |
| Memory (ours) | 81.9 | 81.4 | 79.8 |
| RAG (EmbeddingGemma-2, ours) | – | 67.2 | 65.0 |
| Memory + RAG (ours) | – | 79.3 | 78.3 |
| **same questions (rag_set)** |  |  |  |
| RAG (SimCSE, paper) | – | 69.0 | 64.0 |
| RAG (EmbeddingGemma-2, ours) | – | 67.6 | 65.6 |
| Memory + RAG (ours) | – | 79.3 | 78.2 |

Phase 3 diagnostics:

| method     |   inter_turns |       n |   topics |   parse_fail_pct |   acc_lenient |   truncated |   sec_per_q |
|:-----------|--------------:|--------:|---------:|-----------------:|--------------:|------------:|------------:|
| memory     |             0 | 1000.00 |    20.00 |             0.10 |         81.48 |        0.00 |        0.42 |
| memory     |             3 | 1000.00 |    20.00 |             0.00 |         81.16 |        0.00 |        1.09 |
| memory     |             8 | 1000.00 |    20.00 |             0.10 |         79.69 |        0.00 |        2.36 |
| memory_rag |             3 | 1000.00 |    20.00 |             0.00 |         79.26 |        0.00 |        0.91 |
| memory_rag |             8 | 1000.00 |    20.00 |             0.10 |         78.28 |        0.00 |        2.55 |
| rag_gemma  |             3 | 1000.00 |    20.00 |             0.00 |         67.18 |        0.00 |        1.45 |
| rag_gemma  |             8 | 1000.00 |    20.00 |             0.00 |         65.02 |        0.00 |        2.44 |
