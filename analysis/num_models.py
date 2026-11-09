"""How the number of models compared changes agreement: the gym ranks 5, 10, 15 or 20 of the 28 models.

JEV-27B pointwise, from the published grades. For each task, 20 random sets of models (seeds 0-19). Each query's ideal
list comes only from the documents the chosen models returned, as if the gym had been run on those models alone."""

import numpy as np

from common import GENERATOR, JUDGE, TASKS, agreement, judged, ndcg, overall_agreement, records, save_csv, write

R = records()
SIZES, DRAWS = (5, 10, 15, 20, 28), 20


def ranking(chosen, qids, top, grades):
    scores = {m: [] for m in chosen}
    for q in qids:
        pool = [grades[q][d] for d in {d for m in chosen for d in top[m][q]} if d in grades[q]]
        for m in chosen:
            scores[m].append(ndcg(top[m][q], grades[q], pool, 10))
    return {m: np.mean(v) for m, v in scores.items()}


result, per_task, overall = {}, [], []
for queries in ("original", GENERATOR):
    found = {n: [] for n in SIZES}
    draws_scores = {}  # (n, seed) -> {task: {model: score}}; a seed picks the same models on every task
    for task in TASKS:
        models, qids, top, grades, _ = judged(R[(JUDGE, "pointwise", queries)][task])
        for n in SIZES:
            draws = []
            for seed in range(DRAWS if n < len(models) else 1):
                chosen = list(np.random.default_rng(seed).choice(models, n, replace=False))
                scores = ranking(chosen, qids, top, grades)
                draws_scores.setdefault((n, seed), {})[task] = scores
                draws.append(agreement(scores, task))
            found[n] += draws
            per_task.append({"queries": queries, "task": task, "n": n, "agreement": float(np.nanmean(draws))})
        print(queries, task, flush=True)
    result[queries] = {n: float(np.nanmean(v)) for n, v in found.items()}
    for n in SIZES:
        values = [overall_agreement(draws_scores[key]) for key in draws_scores if key[0] == n]
        overall.append({"queries": queries, "n": n, "agreement": float(np.nanmean(values)),
                        "low": float(np.nanpercentile(values, 2.5)), "high": float(np.nanpercentile(values, 97.5))})

rows = [{"Models": n, "Original queries": result["original"][n], "Generated queries": result[GENERATOR][n]}
        for n in SIZES]
save_csv("num_models_per_task", per_task)
save_csv("num_models_overall", overall)
write("num_models", rows, "Agreement with the labels by number of models ranked, averaged over 28 tasks and 20 draws.")
