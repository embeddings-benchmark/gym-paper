"""7. How many queries are enough: JEV-27B pointwise, original and synthetic (DeepSeek-V4-Flash) queries. Per-query
nDCG@10 is rebuilt from the grades and predictions the records came from, as published in mteb/gym-runs (checked against
each record's ratings),
then 10, 25, 50 and all queries are drawn 20 times per task (seeds 0-19). For each draw: Spearman against the labels
on the 100 sampled queries, and, for original queries, against the labels on the same drawn queries too; and the
regret of the top pick against the 100-query labels. A task with fewer queries than a count uses all it has."""

import glob
import json
import logging
import statistics as st
import warnings

import numpy as np
from scipy.stats import spearmanr

from huggingface_hub import hf_hub_download, snapshot_download

from common import GYM_RUNS, MAIN_JUDGE, TASKS, labels, ratings, records, write
from mteb_gym import corpus as corpus_mod
from mteb_gym.rank import ndcg_per_query

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
COUNTS, SEEDS = (10, 25, 50, 100), range(20)


def label_ndcg(ids: list[str], rels: dict[str, int], k: int = 10) -> float:
    """trec_eval's nDCG@k with the qrels' graded gains, as mteb scores retrieval."""
    ideal = sorted(rels.values(), reverse=True)[:k]
    idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal))
    dcg = sum(rels.get(d, 0) / np.log2(i + 2) for i, d in enumerate(ids[:k]))
    return dcg / idcg if idcg > 0 else 0.0


def tables(rec: dict, corp) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    """(queries, models, judge nDCG per query and model, label nDCG per query and model)."""
    c = rec["config"]
    qset, task = c["query_set"], rec["task_name"]
    grades: dict[str, dict[str, float]] = {}
    folder = snapshot_download(GYM_RUNS, repo_type="dataset", allow_patterns=[f"grades/{task}/autotrust_JEV-27B-*.jsonl"])
    for f in glob.glob(f"{folder}/grades/{task}/autotrust_JEV-27B-*.jsonl"):
        for line in open(f):
            g = json.loads(line)
            if g["query_set"] == qset and g["parsed_ok"]:  # one JEV-27B pointwise grade file per query set
                grades.setdefault(g["qid"], {})[g["doc_id"]] = g["grade"]
    models = sorted(c["models"])
    top = {m: {} for m in models}
    # one row per model and query with its top 100 documents; the table names the query set by its queries file
    table_qset = qset if c["query_source"] == "original" else qset.rsplit("-", 1)[0]
    for line in open(hf_hub_download(GYM_RUNS, f"tables/predictions/{task}.jsonl", repo_type="dataset")):
        p = json.loads(line)
        if p["query_set"] == table_qset and p["model"] in top and p["revision"] == c["model_revisions"][p["model"]]:
            ids = [d for d in p["doc_ids"] if not (corp.ignore_identical_ids and d == p["qid"])]  # as mteb scores it
            top[p["model"]][p["qid"]] = ids[:10]
    if any(not top[m] for m in models):
        raise ValueError(f"no predictions for {[m for m in models if not top[m]]} on {task}")
    queries = sorted(q for q in grades if all(q in top[m] for m in models))
    judge = np.array([[ndcg_per_query(top[m][q], grades[q], list(grades[q].values()), 10) for m in models] for q in queries])
    lab = None
    if c["query_source"] == "original":
        lab = np.array([[label_ndcg(top[m][q], corp.qrels.get(q, {})) for m in models] for q in queries])
    return queries, models, judge, lab


R = records()
rows, check = [], []
for source, gen in (("original", "original"), ("synthetic", "DeepSeek-V4-Flash")):
    out = {n: {"rho_100": [], "rho_same": [], "regret": []} for n in COUNTS}
    for t in TASKS:
        rec = R[(MAIN_JUDGE, "pointwise", gen)][t]
        corp = corpus_mod.load(t)
        queries, models, judge, lab = tables(rec, corp)
        full = dict(zip(models, 100 * judge.mean(0)))
        rec_r = ratings(rec)
        check.append(dict(task=t, queries=source, n=len(queries), max_gap_to_record=max(abs(full[m] - rec_r[m]) for m in models),
                          label_gap_to_ref=max(abs(lab.mean(0)[i] - labels(t)[m]) for i, m in enumerate(models)) if lab is not None else float("nan")))
        y = labels(t)
        yv = [y[m] for m in models]
        for n in COUNTS:
            for s in SEEDS:
                idx = np.random.default_rng(s).choice(len(queries), min(n, len(queries)), replace=False)
                score = judge[idx].mean(0)
                out[n]["rho_100"].append(spearmanr(score, yv)[0])
                if lab is not None:
                    out[n]["rho_same"].append(spearmanr(score, lab[idx].mean(0))[0])
                out[n]["regret"].append(100 * (max(yv) - yv[int(np.argmax(score))]))
    for n in COUNTS:
        o = out[n]
        rows.append(dict(queries=source, n_queries=n, draws=len(o["rho_100"]), mean_rho_vs_100q_labels=float(np.nanmean(o["rho_100"])),
                         mean_rho_vs_same_queries=float(np.nanmean(o["rho_same"])) if o["rho_same"] else float("nan"),
                         mean_regret=float(np.mean(o["regret"])), median_regret=float(np.median(o["regret"]))))
        print(rows[-1], flush=True)
write("7_query_count", rows, "JEV-27B pointwise. Each row averages 28 tasks x 20 draws. Labels on the 100 sampled "
      "original queries (data/labels); 'same queries' uses labels on the drawn queries (original only).")
write("7_query_count_check", check, "Rebuilt per-query nDCG against each record's ratings (should be ~0), and rebuilt "
      "label nDCG against data/labels (should be ~0).")
print("largest gap to records", max(c["max_gap_to_record"] for c in check), "largest label gap", np.nanmax([c["label_gap_to_ref"] for c in check]))
