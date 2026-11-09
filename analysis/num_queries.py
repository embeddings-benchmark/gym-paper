"""How many queries are needed: agreement with the labels when the gym uses only 10, 25 or 50 of the 100 queries.

JEV-27B pointwise. For each task, 20 random draws of queries (seeds 0-19), and the models are ranked by their mean nDCG
on the drawn queries. A task with fewer queries than a count uses all it has."""

import numpy as np

from common import GENERATOR, JUDGE, TASKS, agreement, judged, ndcg, overall_agreement, records, save_csv, write

R = records()
COUNTS, DRAWS = (10, 25, 50, 100), 20

result, per_task, overall = {}, [], []
for queries in ("original", GENERATOR):
    found = {n: [] for n in COUNTS}
    draws_scores = {}  # (n, seed) -> {task: {model: score}}
    for task in TASKS:
        models, qids, top, grades = judged(R[(JUDGE, "pointwise", queries)][task])
        table = np.array([[ndcg(top[m][q], grades[q], grades[q].values(), 10) for m in models] for q in qids])
        for n in COUNTS:
            draws = []
            for seed in range(DRAWS):
                pick = np.random.default_rng(seed).choice(len(qids), min(n, len(qids)), replace=False)
                scores = dict(zip(models, table[pick].mean(0)))
                draws_scores.setdefault((n, seed), {})[task] = scores
                draws.append(agreement(scores, task))
            found[n] += draws
            per_task.append({"queries": queries, "task": task, "n": n, "agreement": float(np.nanmean(draws))})
        print(queries, task, flush=True)
    result[queries] = {n: float(np.nanmean(v)) for n, v in found.items()}
    for n in COUNTS:
        values = [overall_agreement(draws_scores[(n, seed)]) for seed in range(DRAWS)]
        overall.append({"queries": queries, "n": n, "agreement": float(np.mean(values)),
                        "low": float(np.percentile(values, 2.5)), "high": float(np.percentile(values, 97.5))})

rows = [{"Queries per task": n, "Original queries": result["original"][n], "Generated queries": result[GENERATOR][n]}
        for n in COUNTS]
save_csv("num_queries_per_task", per_task)
save_csv("num_queries_overall", overall)
write("num_queries", rows, "Agreement with the labels by number of queries, averaged over 28 tasks and 20 draws.")
