"""6. Placing one model among the other 27 on a task, with mean absolute error between its predicted rank and its
labelled rank (1-28). The Gym methods judge every model on the task, so a model needs no history. The history
baselines place a model by its ranks on the other tasks; a model with no history at all can only be put in the middle
(rank 14.5). Earlier numbers (2026-11-06, inline): 3.9 original, 4.7 synthetic, 4.82 history (labels), 7.0 none."""

import statistics as st
from importlib import import_module

import numpy as np

from common import TASKS, boot_mean, labels, rank_of, write
from history import published

methods = import_module("3_top_weighted").methods()
pub = published()
rows, no_history = [], []
for name, s in methods.items():
    err = []
    for t in TASKS:
        y = labels(t)
        models = sorted(y)
        scores = {m: (s[t][m] if s[t][m] == s[t][m] else -np.inf) for m in models}  # no history: last, see below
        ry, rm = rank_of(y, models), rank_of(scores, models)
        for i, m in enumerate(models):
            err.append(abs(14.5 - ry[i]) if scores[m] == -np.inf else abs(rm[i] - ry[i]))
    e, lo, hi = boot_mean(err)
    rows.append(dict(method=name, placements=len(err), mean_rank_error=e, ci_low=lo, ci_high=hi, median_rank_error=st.median(err)))
err = [abs(14.5 - r) for t in TASKS for r in rank_of(labels(t), sorted(labels(t)))]
rows.append(dict(method="no information (rank 14.5)", placements=len(err), mean_rank_error=float(np.mean(err)),
                 ci_low=float("nan"), ci_high=float("nan"), median_rank_error=st.median(err)))
# models with no published score on any other of the 28 tasks, task by task
for t in TASKS:
    for m in sorted(labels(t)):
        if not any(m in pub[o] for o in TASKS if o != t):
            no_history.append(dict(task=t, model=m))
models_without = sorted({x["model"] for x in no_history})
write("6_unseen_models", rows, "Mean absolute rank error (1-28) over every task and model; CIs bootstrap placements. "
      f"Models with no published MTEB score on any of the other 27 tasks: {len(models_without)} "
      f"({', '.join(models_without) or 'none'}).")
for x in rows:
    print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in x.items()})
print("models with no published history on the other tasks:", models_without)
cov = {m: sum(m in pub[t] for t in TASKS) for m in sorted(labels(TASKS[0]))}
print("published scores per model (of 28 tasks):", sorted(cov.values()))
