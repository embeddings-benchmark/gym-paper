Regret in nDCG@10 points against the labels on the same 100 sampled queries (data/labels). Methods as in 3_top_weighted.py.

| method | tasks | mean_regret | median_regret | tasks_under_1_point | max_regret |
|---|---|---|---|---|---|
| JEV-27B pointwise original | 28 | 6.546 | 4.022 | 8 | 29.906 |
| JEV-27B pointwise synthetic | 28 | 7.856 | 4.354 | 5 | 37.835 |
| JEV-27B pairwise original | 28 | 5.249 | 2.424 | 11 | 29.906 |
| JEV-27B pairwise synthetic | 28 | 6.604 | 3.603 | 8 | 37.835 |
| history (labels) | 28 | 6.330 | 4.022 | 6 | 29.906 |
| history (published) | 28 | 6.330 | 4.022 | 6 | 29.906 |
| median model (reference) | 28 | 16.948 | 13.862 | 0 | 80.991 |
| all-MiniLM-L6-v2 (reference) | 28 | 26.099 | 24.216 | 0 | 89.868 |
