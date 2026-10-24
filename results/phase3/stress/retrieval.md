# Stress test, level 1: is the true preference among the entries put in the prompt?

Memory at 3k tokens (inter_turns 8) + D preferences from other topics; top5 = EmbeddingGemma-2 retrieval over the memory.

| form            |   distractors | policy   |    n |   entries_in_prompt |   gold_in_prompt |   gold_in_prompt_given_in_memory |   mean_gold_position |
|:----------------|--------------:|:---------|-----:|--------------------:|-----------------:|---------------------------------:|---------------------:|
| explicit        |             5 | all      | 1000 |               5.986 |            0.986 |                            1.000 |                3.501 |
| explicit        |             5 | top5     | 1000 |               5.000 |            0.986 |                            1.000 |                2.973 |
| explicit        |            10 | all      | 1000 |              10.986 |            0.986 |                            1.000 |                6.044 |
| explicit        |            10 | top5     | 1000 |               5.000 |            0.977 |                            0.991 |                3.055 |
| explicit        |            20 | all      | 1000 |              20.986 |            0.986 |                            1.000 |               10.859 |
| explicit        |            20 | top5     | 1000 |               5.000 |            0.960 |                            0.974 |                2.995 |
| implicit-choice |             5 | all      | 1000 |               5.999 |            0.999 |                            1.000 |                3.466 |
| implicit-choice |             5 | top5     | 1000 |               5.000 |            0.999 |                            1.000 |                2.974 |
| implicit-choice |            10 | all      | 1000 |              10.999 |            0.999 |                            1.000 |                6.084 |
| implicit-choice |            10 | top5     | 1000 |               5.000 |            0.998 |                            0.999 |                3.052 |
| implicit-choice |            20 | all      | 1000 |              20.999 |            0.999 |                            1.000 |               10.993 |
| implicit-choice |            20 | top5     | 1000 |               5.000 |            0.984 |                            0.985 |                3.003 |
