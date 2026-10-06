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
| CoT | 0.2k (0) | 93.7 | 83 | +10.7 | 20/20 | 1000 | 0.0 | 0 |
| CoT | 1k (3) | 78.0 | 67 | +11.0 | 20/20 | 1000 | 0.2 | 0 |
| CoT | 3k (8) | 63.9 | 50 | +13.9 | 20/20 | 1000 | 0.1 | 0 |
| Self-Critic | 0.2k (0) | 78.8 | 79 | -0.2 | 20/20 | 1000 | 0.0 | 0 |
| Self-Critic | 1k (3) | 50.1 | 73 | -22.9 | 20/20 | 1000 | 25.2 | 0 |
| Self-Critic | 3k (8) | 46.7 | 62 | -15.3 | 20/20 | 1000 | 2.5 | 0 |
| RAG (top-5) | 1k (3) | 88.6 | 87 | +1.6 | 19/19† | 949 | 5.9 | 0 |
| RAG (top-5) | 3k (8) | 74.8 | 80 | -5.2 | 19/19† | 949 | 19.1 | 0 |
† RAG excludes entertain_games: the upstream precomputed SimCSE retrieval file is truncated (only 4/51 questions parse), so RAG is averaged over the remaining topics.

## mistral7b | explicit

| Method | Context (inter_turns) | Ours | Paper | Δ (ours − paper) | Topics | Questions | Parse-fail % | Truncated |
|---|---|---|---|---|---|---|---|---|
| Zero-shot | 0.2k (0) | 95.5 | 87 | +8.5 | 20/20 | 1000 | 0.0 | 0 |
| Zero-shot | 1k (3) | 61.5 | 66 | -4.5 | 20/20 | 1000 | 0.0 | 0 |
| Zero-shot | 3k (8) | 55.5 | 57 | -1.5 | 20/20 | 1000 | 0.0 | 0 |
| Zero-shot | 10k (28) | 42.5 | 50 | -7.5 | 20/20 | 1000 | 0.0 | 0 |
| Zero-shot | 16k (48) | 37.1 | 42 | -4.9 | 20/20 | 1000 | 0.0 | 0 |
| Zero-shot | 23k (68) | 37.6 | 47 | -9.4 | 20/20 | 1000 | 0.0 | 0 |
| Reminder | 0.2k (0) | 96.2 | 95 | +1.2 | 20/20 | 1000 | 0.0 | 0 |
| Reminder | 1k (3) | 78.9 | 85 | -6.1 | 20/20 | 1000 | 0.0 | 0 |
| Reminder | 3k (8) | 71.3 | 75 | -3.7 | 20/20 | 1000 | 0.0 | 0 |
| Reminder | 10k (28) | 56.7 | 63 | -6.3 | 20/20 | 1000 | 0.0 | 0 |
| Reminder | 16k (48) | 43.9 | 56 | -12.1 | 20/20 | 1000 | 0.0 | 0 |
| Reminder | 23k (68) | 46.9 | 56 | -9.1 | 20/20 | 1000 | 0.0 | 0 |
| CoT | 0.2k (0) | 88.0 | 90 | -2.0 | 20/20 | 1000 | 0.1 | 0 |
| CoT | 1k (3) | 74.9 | 79 | -4.1 | 20/20 | 1000 | 0.0 | 0 |
| CoT | 3k (8) | 67.0 | 73 | -6.0 | 20/20 | 1000 | 0.0 | 0 |
| Self-Critic | 0.2k (0) | 79.8 | 75 | +4.8 | 20/20 | 1000 | 0.2 | 0 |
| Self-Critic | 1k (3) | 67.0 | 65 | +2.0 | 20/20 | 1000 | 0.0 | 0 |
| Self-Critic | 3k (8) | 61.0 | 58 | +3.0 | 20/20 | 1000 | 0.0 | 0 |
| RAG (top-5) | 1k (3) | 81.0 | 87 | -6.0 | 19/19† | 949 | 0.0 | 0 |
| RAG (top-5) | 3k (8) | 75.7 | 86 | -10.3 | 19/19† | 949 | 0.6 | 0 |
| RAG (top-5) | 10k (28) | 64.6 | 81 | -16.4 | 19/19† | 949 | 0.0 | 0 |
| RAG (top-5) | 16k (48) | 59.0 | 80 | -21.0 | 19/19† | 949 | 0.0 | 0 |
| RAG (top-5) | 23k (68) | 53.7 | 75 | -21.3 | 19/19† | 949 | 0.2 | 0 |
† RAG excludes entertain_games: the upstream precomputed SimCSE retrieval file is truncated (only 4/51 questions parse), so RAG is averaged over the remaining topics.

## mistral7b | implicit-choice

| Method | Context (inter_turns) | Ours | Paper | Δ (ours − paper) | Topics | Questions | Parse-fail % | Truncated |
|---|---|---|---|---|---|---|---|---|
| Zero-shot | 0.2k (0) | 81.0 | – | – | 20/20 | 1000 | 0.0 | 0 |
| Zero-shot | 1k (3) | 50.8 | – | – | 20/20 | 1000 | 0.0 | 0 |
| Zero-shot | 3k (8) | 47.5 | – | – | 20/20 | 1000 | 0.0 | 0 |
| Reminder | 0.2k (0) | 88.4 | – | – | 20/20 | 1000 | 0.0 | 0 |
| Reminder | 1k (3) | 71.9 | – | – | 20/20 | 1000 | 0.0 | 0 |
| Reminder | 3k (8) | 63.1 | – | – | 20/20 | 1000 | 0.0 | 0 |
| RAG (top-5) | 1k (3) | 69.0 | – | – | 20/20 | 973 | 0.0 | 0 |
| RAG (top-5) | 3k (8) | 64.0 | – | – | 20/20 | 973 | 0.0 | 0 |
‡ RAG covers 973 of 1000 questions: the upstream retrieval files end early for travel_restaurant (53/56), entertain_shows (60/62), pet_ownership (35/43), lifestyle_health (45/49), education_learning_styles (26/31), shop_technology (32/35), travel_hotel (52/54).

