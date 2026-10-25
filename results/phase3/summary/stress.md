# Stress test, level 2: accuracy with D extra preferences in memory (memory-only prompt)

## llama3-8b | explicit | 3k tokens (D = 0: tier A memory run)

|   distractors |   all |   top5 |
|--------------:|------:|-------:|
|             0 |  96.5 |   96.5 |
|             5 |  94.3 |   94.2 |
|            10 |  92.8 |   93.8 |
|            20 |  90.5 |   93.9 |

## llama3-8b | implicit-choice | 3k tokens (D = 0: tier A memory run)

|   distractors |   all |   top5 |
|--------------:|------:|-------:|
|             0 |  83.5 |   83.5 |
|             5 |  77.1 |   78.0 |
|            10 |  73.7 |   77.7 |
|            20 |  71.1 |   77.0 |
