RBO at p = 0.9 (Webber et al., 2010) against the labels' ranking; top-1 and top-3 counted out of 28 tasks. History baselines: see history.py.

| method | tasks | mean_rbo | ci_low | ci_high | median_rbo | top1 | top_pick_in_label_top3 |
|---|---|---|---|---|---|---|---|
| JEV-27B pointwise original | 28 | 0.645 | 0.584 | 0.705 | 0.685 | 7 | 17 |
| JEV-27B pointwise synthetic | 28 | 0.576 | 0.519 | 0.632 | 0.597 | 4 | 13 |
| JEV-27B pairwise original | 28 | 0.666 | 0.601 | 0.726 | 0.694 | 10 | 20 |
| JEV-27B pairwise synthetic | 28 | 0.589 | 0.526 | 0.652 | 0.590 | 7 | 19 |
| history (labels) | 28 | 0.589 | 0.542 | 0.636 | 0.572 | 4 | 16 |
| history (published) | 28 | 0.584 | 0.536 | 0.630 | 0.584 | 4 | 16 |
