Each row against JEV-27B on the same tasks: judges on the same queries, generators against JEV-27B on DeepSeek-V4-Flash's queries. Rule fixed before results: replace JEV-27B only for a gain >= .03 with the 95% CI excluding 0. Low-effort grades: data/low_effort_*.tsv (temperature 0, no cap otherwise).

| kind | judge | judging | queries | tasks | mean_rho | median_rho | diff_vs_jev27b | ci_low | ci_high | passes_rule | note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| judge | AutoTrust-JEV-27B-v1.1 | pointwise | DeepSeek-V4-Flash | 28 | 0.680 | 0.738 | 0.015 | -0.009 | 0.041 | False |  |
| judge | AutoTrust-JEV-27B-v1.1 | pointwise | original | 28 | 0.743 | 0.830 | 0.001 | -0.026 | 0.028 | False |  |
| judge | DeepSeek-V4-Flash | pointwise | DeepSeek-V4-Flash | 28 | 0.641 | 0.736 | -0.024 | -0.048 | -0.000 | False | 3 grades at low reasoning effort |
| judge | DeepSeek-V4-Flash | pointwise | original | 28 | 0.759 | 0.844 | 0.017 | -0.005 | 0.039 | False | 3 grades at low reasoning effort |
| judge | GEV-26B-Decide | pointwise | DeepSeek-V4-Flash | 28 | 0.662 | 0.768 | -0.003 | -0.023 | 0.014 | False |  |
| judge | GEV-26B-Decide | pointwise | original | 28 | 0.720 | 0.846 | -0.022 | -0.052 | 0.004 | False |  |
| generator | JEV-27B | pointwise | MiniMax-M2.7 | 28 | 0.672 | 0.744 | 0.007 | -0.011 | 0.026 | False |  |
| generator | JEV-27B | pointwise | Qwen3.5-4B | 28 | 0.653 | 0.700 | -0.012 | -0.038 | 0.013 | False |  |
| generator | JEV-27B | pointwise | Qwen3.8-27B | 28 | 0.662 | 0.732 | -0.003 | -0.023 | 0.016 | False |  |
| judge | JEV-35B | pointwise | DeepSeek-V4-Flash | 28 | 0.666 | 0.757 | 0.001 | -0.011 | 0.015 | False |  |
| judge | JEV-35B | pointwise | original | 28 | 0.753 | 0.856 | 0.011 | -0.018 | 0.052 | False |  |
| judge | JEV-9B | pointwise | DeepSeek-V4-Flash | 28 | 0.662 | 0.784 | -0.003 | -0.019 | 0.011 | False |  |
| judge | JEV-9B | pointwise | original | 28 | 0.709 | 0.794 | -0.033 | -0.053 | -0.015 | False |  |
| judge | Qwen3.8-27B | pointwise | DeepSeek-V4-Flash | 28 | 0.668 | 0.778 | 0.003 | -0.012 | 0.019 | False | 71 grades at low reasoning effort |
| judge | Qwen3.8-27B | pointwise | original | 28 | 0.785 | 0.831 | 0.043 | 0.023 | 0.068 | True | 71 grades at low reasoning effort |
| judge | decider-4b | pointwise | DeepSeek-V4-Flash | 28 | 0.687 | 0.782 | 0.022 | -0.013 | 0.058 | False |  |
| judge | decider-4b | pointwise | original | 28 | 0.768 | 0.839 | 0.025 | -0.040 | 0.105 | False |  |
