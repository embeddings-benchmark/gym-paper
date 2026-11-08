"""1. Agreement with labels per judging mode and query source (JEV-27B, DeepSeek-V4-Flash queries): Spearman and Kendall
over the 28 models per task, then mean, median, tasks at >= .70 and a 95% bootstrap CI over tasks."""

import statistics as st

from common import MAIN_GENERATOR, MAIN_JUDGE, TASKS, agree, boot_mean, labels, ratings, records, write

R = records()
rows, per_task = [], []
for judging in ("pointwise", "pairwise"):
    for queries in ("original", MAIN_GENERATOR):
        recs = R[(MAIN_JUDGE, judging, queries)]
        assert sorted(recs) == sorted(TASKS), (judging, queries, len(recs))
        rho, tau = zip(*(agree(ratings(recs[t]), labels(t)) for t in TASKS))
        m, lo, hi = boot_mean(rho)
        rows.append(dict(judging=judging, queries="original" if queries == "original" else "synthetic", tasks=len(rho),
                         mean_rho=m, ci_low=lo, ci_high=hi, median_rho=st.median(rho), tasks_at_70=sum(r >= 0.7 for r in rho),
                         mean_tau=st.mean(tau), median_tau=st.median(tau)))
        per_task += [dict(task=t, judging=judging, queries=rows[-1]["queries"], rho=r, tau=k, record=recs[t]["_path"])
                     for t, r, k in zip(TASKS, rho, tau)]
write("1_main_table", rows, "Records: gym-results, judge JEV-27B; synthetic queries from DeepSeek-V4-Flash. "
      "Labels: data/labels (the three BRIGHT tasks with excluded documents drop them). 95% CI: 10,000 bootstrap "
      "resamples of tasks, seed 0.")
write("1_main_table_per_task", per_task)
for r in rows:
    print(r)
