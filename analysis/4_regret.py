"""4. Selection regret: the labelled nDCG@10 of the model a method ranks first, minus the best model's, in nDCG points
(x 100), per task. Also the median model and all-MiniLM-L6-v2 as reference picks."""

import statistics as st

from common import TASKS, labels, write
from importlib import import_module

methods = import_module("3_top_weighted").methods()
rows, per_task = [], []
picks = {name: {t: max((m for m, v in s[t].items() if v == v), key=s[t].get) for t in TASKS} for name, s in methods.items()}
picks["median model (reference)"] = {t: sorted(labels(t), key=labels(t).get)[len(labels(t)) // 2] for t in TASKS}
picks["all-MiniLM-L6-v2 (reference)"] = {t: "sentence-transformers/all-MiniLM-L6-v2" for t in TASKS}
for name, pick in picks.items():
    regret = []
    for t in TASKS:
        y = labels(t)
        regret.append(100 * (max(y.values()) - y[pick[t]]))
        per_task.append(dict(method=name, task=t, pick=pick[t], regret=regret[-1]))
    rows.append(dict(method=name, tasks=len(regret), mean_regret=st.mean(regret), median_regret=st.median(regret),
                     tasks_under_1_point=sum(r < 1 for r in regret), max_regret=max(regret)))
write("4_regret", rows, "Regret in nDCG@10 points against the labels on the same 100 sampled queries (data/labels). "
      "Methods as in 3_top_weighted.py.")
write("4_regret_per_task", per_task)
for x in rows:
    print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in x.items()})
