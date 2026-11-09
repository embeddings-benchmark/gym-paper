"""How the number of models compared changes agreement: the gym ranks 5, 10, 15 or 20 of the 28 models.

JEV-27B pointwise, from the published grades. For each task, 20 random sets of models (seeds 0-19). Each query's ideal
list comes only from the documents the chosen models returned, as if the gym had been run on those models alone."""

import numpy as np

from common import GENERATOR, JUDGE, TASKS, agreement, judged, ndcg, records, write

R = records()
SIZES, DRAWS = (5, 10, 15, 20, 28), 20


def ranking(chosen, qids, top, grades):
    scores = {m: [] for m in chosen}
    for q in qids:
        pool = [grades[q][d] for d in {d for m in chosen for d in top[m][q]} if d in grades[q]]
        for m in chosen:
            scores[m].append(ndcg(top[m][q], grades[q], pool, 10))
    return {m: np.mean(v) for m, v in scores.items()}


result = {}
for queries in ("original", GENERATOR):
    found = {n: [] for n in SIZES}
    for task in TASKS:
        models, qids, top, grades = judged(R[(JUDGE, "pointwise", queries)][task])
        for n in SIZES:
            for seed in range(DRAWS if n < len(models) else 1):
                chosen = list(np.random.default_rng(seed).choice(models, n, replace=False))
                found[n].append(agreement(ranking(chosen, qids, top, grades), task))
        print(queries, task, flush=True)
    result[queries] = {n: float(np.nanmean(v)) for n, v in found.items()}

rows = [{"Models": n, "Original queries": result["original"][n], "Generated queries": result[GENERATOR][n]}
        for n in SIZES]
write("num_models", rows, "Agreement with the labels by number of models ranked, averaged over 28 tasks and 20 draws.")
