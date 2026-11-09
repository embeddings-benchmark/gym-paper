"""How many documents per query the judge needs to grade: each model's top 1, 3, 5 or 10.

JEV-27B pointwise, from the published grades. With the top k, each model is scored by nDCG@k, and each query's ideal
list comes from the documents in any model's top k."""

import numpy as np

from common import GENERATOR, JUDGE, TASKS, agreement, judged, ndcg, records, write

R = records()
DEPTHS = (1, 3, 5, 10)

result = {}
for queries in ("original", GENERATOR):
    found = {k: [] for k in DEPTHS}
    for task in TASKS:
        models, qids, top, grades = judged(R[(JUDGE, "pointwise", queries)][task])
        for k in DEPTHS:
            scores = {m: [] for m in models}
            for q in qids:
                pool = [grades[q][d] for d in {d for m in models for d in top[m][q][:k]} if d in grades[q]]
                for m in models:
                    scores[m].append(ndcg(top[m][q], grades[q], pool, k))
            found[k].append(agreement({m: np.mean(v) for m, v in scores.items()}, task))
        print(queries, task, flush=True)
    result[queries] = {k: float(np.nanmean(v)) for k, v in found.items()}

rows = [{"Documents per query": k, "Original queries": result["original"][k], "Generated queries": result[GENERATOR][k]}
        for k in DEPTHS]
write("num_documents", rows, "Agreement with the labels by number of documents graded per model and query, "
      "averaged over 28 tasks.")
