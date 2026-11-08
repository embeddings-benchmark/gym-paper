Records: gym-results, judge JEV-27B; synthetic queries from DeepSeek-V4-Flash. Labels: data/labels (the three BRIGHT tasks with excluded documents drop them). 95% CI: 10,000 bootstrap resamples of tasks, seed 0.

| judging | queries | tasks | mean_rho | ci_low | ci_high | median_rho | tasks_at_70 | mean_tau | median_tau |
|---|---|---|---|---|---|---|---|---|---|
| pointwise | original | 28 | 0.742 | 0.638 | 0.828 | 0.827 | 21 | 0.608 | 0.669 |
| pointwise | synthetic | 28 | 0.665 | 0.546 | 0.761 | 0.753 | 19 | 0.526 | 0.606 |
| pairwise | original | 28 | 0.764 | 0.665 | 0.845 | 0.836 | 22 | 0.631 | 0.683 |
| pairwise | synthetic | 28 | 0.665 | 0.553 | 0.757 | 0.772 | 17 | 0.528 | 0.593 |
