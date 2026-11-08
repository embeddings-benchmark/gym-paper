Mean absolute rank error (1-28) over every task and model; CIs bootstrap placements. Models with no published MTEB score on any of the other 27 tasks: 1 (colbert-ir/colbertv2.0).

| method | placements | mean_rank_error | ci_low | ci_high | median_rank_error |
|---|---|---|---|---|---|
| JEV-27B pointwise original | 784 | 3.903 | 3.614 | 4.212 | 2.000 |
| JEV-27B pointwise synthetic | 784 | 4.702 | 4.381 | 5.033 | 3.000 |
| JEV-27B pairwise original | 784 | 3.707 | 3.425 | 4.006 | 2.000 |
| JEV-27B pairwise synthetic | 784 | 4.696 | 4.379 | 5.036 | 3.000 |
| history (labels) | 784 | 4.823 | 4.517 | 5.147 | 4.000 |
| history (published) | 784 | 4.726 | 4.413 | 5.047 | 3.000 |
| no information (rank 14.5) | 784 | 7.000 | nan | nan | 7.000 |
