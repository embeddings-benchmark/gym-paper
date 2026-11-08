"""5. Leaderboard history as a baseline: the same agreement with labels as in 1, for rankings made from the other 27
tasks only (history.py), and each Gym method's paired difference from it, with 95% bootstrap CIs over tasks."""

import statistics as st
from importlib import import_module

from common import TASKS, agree, boot_mean, labels, write

methods = import_module("3_top_weighted").methods()


def rho(scores, t):
    s = {m: v for m, v in scores.items() if v == v}
    return agree(s, labels(t))[0]


R = {name: [rho(s[t], t) for t in TASKS] for name, s in methods.items()}
rows = []
for name, v in R.items():
    m, lo, hi = boot_mean(v)
    rows.append(dict(method=name, mean_rho=m, ci_low=lo, ci_high=hi, median_rho=st.median(v), tasks_at_70=sum(x >= 0.7 for x in v)))
diffs = []
for base in ("history (labels)", "history (published)"):
    for name, v in R.items():
        if name.startswith("history"):
            continue
        d, lo, hi = boot_mean([a - b for a, b in zip(v, R[base])])
        diffs.append(dict(method=name, baseline=base, diff=d, ci_low=lo, ci_high=hi,
                          tasks_method_better=sum(a > b for a, b in zip(v, R[base]))))
write("5_leaderboard_baseline", rows, "Spearman against labels per task; history baselines in history.py.")
write("5_leaderboard_baseline_diff", diffs, "Paired over the 28 tasks: method minus baseline.")
for x in rows + diffs:
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in x.items()})
