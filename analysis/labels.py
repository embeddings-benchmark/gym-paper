"""The label references: nDCG@10 of each model against the task's own relevance labels, on the 100 original queries
the gym sampled (seed 0), scored with mteb's own metric code. Rebuilt from the published predictions table in
mteb/gym-runs (top 100 documents per query, enough for nDCG@10) and the task's qrels as mteb loads them.

The three BRIGHT tasks with excluded documents drop each query's excluded documents from every list before scoring,
as mteb does (excluded_ids in mteb/BRIGHT's examples).

    python labels.py          # check data/labels against a rebuild; exits 1 on any gap above 1e-9
    python labels.py --write  # rewrite data/labels from the rebuild
"""

import json
import logging
import sys
import warnings

from huggingface_hub import hf_hub_download

from common import DATA, GYM_RUNS, TASKS
from mteb_gym import corpus as corpus_mod

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

BRIGHT_REVISION = "c26703e6600d97c579ee2985f16cf307db13ed85"
BRIGHT_DOMAIN = {"BrightAopsRetrieval": "aops", "BrightLeetcodeRetrieval": "leetcode",
                 "BrightTheoremQAQuestionsRetrieval": "theoremqa_questions"}


def excluded(task: str) -> dict[str, set[str]]:
    if task not in BRIGHT_DOMAIN:
        return {}
    import datasets

    ex = datasets.load_dataset("mteb/BRIGHT", "examples", split=BRIGHT_DOMAIN[task], revision=BRIGHT_REVISION)
    return {x["id"]: set(x["excluded_ids"]) - {"N/A"} for x in ex}


def rebuild(task: str) -> dict[str, float]:
    from mteb._evaluators.retrieval_metrics import calculate_retrieval_scores

    corp = corpus_mod.load(task)
    skip = excluded(task)
    runs: dict[str, dict[str, dict[str, float]]] = {}
    for line in open(hf_hub_download(GYM_RUNS, f"tables/predictions/{task}.jsonl", repo_type="dataset")):
        p = json.loads(line)
        if p["generator"] != "original":
            continue
        drop = skip.get(p["qid"], set()) | ({p["qid"]} if corp.ignore_identical_ids else set())
        runs.setdefault(p["model"], {})[p["qid"]] = {d: s for d, s in zip(p["doc_ids"], p["scores"]) if d not in drop}
    out = {}
    for m, h in sorted(runs.items()):
        qrels = {q: corp.qrels[q] for q in h if q in corp.qrels}
        out[m] = float(calculate_retrieval_scores(h, qrels, [10]).ndcg["NDCG@10"])
    return out


if __name__ == "__main__":
    write, worst = "--write" in sys.argv, 0.0
    for t in TASKS:
        new = rebuild(t)
        path = DATA / "labels" / f"{t}.json"
        if write:
            path.write_text(json.dumps(new, indent=1))
            print(t, len(new), "models written", flush=True)
            continue
        old = json.loads(path.read_text())
        gap = max(abs(new[m] - old[m]) for m in old) if set(new) >= set(old) else float("inf")
        worst = max(worst, gap)
        print(f"{t:36s} models {len(new):3d}  largest gap {gap:.2e}", flush=True)
    if not write:
        print("largest gap over all tasks", worst)
        sys.exit(0 if worst <= 1e-9 else 1)
