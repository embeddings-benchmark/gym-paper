"""Code shared by the scripts: loading records, labels and grades, agreement, and writing tables."""

import csv
import glob
import json
import os
import pickle
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download, snapshot_download
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
TABLES = ROOT / "tables"
TASKS = (DATA / "tasks.txt").read_text().split()
GYM_RUNS = "mteb/gym-runs"
JUDGE = "JEV-27B"
GENERATOR = "DeepSeek-V4-Flash"


def short(name):
    return "original" if not name else name.split("+")[0].split("/")[-1]


def runs_folder(pattern):
    """A folder holding mteb/gym-runs files that match pattern: a local copy if GYM_RUNS_LOCAL is set, else a download."""
    if os.environ.get("GYM_RUNS_LOCAL"):
        return os.environ["GYM_RUNS_LOCAL"]
    return snapshot_download(GYM_RUNS, repo_type="dataset", allow_patterns=[pattern])


def records():
    """All records as {(judge, judging, queries): {task: record}}. queries is "original" or the generator's name.

    Records come from gym-results, or from a local folder set in GYM_RESULTS."""
    if os.environ.get("GYM_RESULTS"):
        root = Path(os.environ["GYM_RESULTS"])
    else:
        from mteb_gym.results import download_results

        root = download_results(download_latest=False) / "results"
    out = {}
    for path in sorted(glob.glob(str(root / "*" / "*.json"))):
        r = json.loads(Path(path).read_text())
        c = r["config"]
        queries = "original" if c["query_source"] == "original" else short(c.get("generator_model"))
        key = (short(c["judge_model"]), c["judging"], queries)
        assert r["task_name"] not in out.get(key, {}), f"two records for {key} on {r['task_name']}"
        out.setdefault(key, {})[r["task_name"]] = r
    return out


def ratings(record):
    return {x["model"]: x["rating"] for x in record["ratings"]}


def labels(task):
    """Each model's nDCG@10 against the task's own labels, on the 100 original queries (made by labels.py)."""
    return json.loads((DATA / "labels" / f"{task}.json").read_text())


def agreement(scores, task):
    """Spearman correlation between a ranking of the models and the ranking from the task's labels."""
    truth = labels(task)
    models = sorted(set(scores) & set(truth))
    return float(spearmanr([scores[m] for m in models], [truth[m] for m in models])[0])


def overall_agreement(per_task):
    """Spearman correlation between the models' mean gym score over tasks and their mean nDCG@10 on the labels.

    per_task is {task: {model: score}}."""
    models = sorted(set.intersection(*(set(s) for s in per_task.values())))
    gym = [np.mean([per_task[t][m] for t in per_task]) for m in models]
    truth = [np.mean([labels(t)[m] for t in per_task]) for m in models]
    return float(spearmanr(gym, truth)[0])


def mean_ci(values, seed=0):
    """Mean with a 95% interval from 10,000 bootstrap resamples."""
    v = np.asarray(values, dtype=float)
    means = v[np.random.default_rng(seed).integers(0, len(v), (10_000, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def ndcg(doc_ids, grades, pool, k):
    """nDCG@k of one ranked list, with the judge's grades as gains and the ideal list from the k best grades in pool."""
    ideal = sorted(pool, reverse=True)[:k]
    idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal))
    dcg = sum(grades.get(d, 0.0) / np.log2(i + 2) for i, d in enumerate(doc_ids[:k]))
    return dcg / idcg if idcg > 0 else 0.0


CACHE = Path(os.environ.get("GYM_PAPER_CACHE", Path.home() / ".cache" / "gym-paper"))


def judged(record):
    """What the judge saw for one JEV-27B pointwise record: (models, queries, top, grades, qrels).

    top[model][query] is the model's top 10 documents, grades[query][doc] is the judge's grade, qrels the task's own
    relevance labels. Cached on disk (GYM_PAPER_CACHE) after the first load."""
    path = CACHE / f"{record['task_name']}__{record['config']['query_set']}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    out = _judged(record)
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(out))
    return out


def _judged(record):
    from mteb_gym import corpus

    c, task = record["config"], record["task_name"]
    pattern = f"grades/{task}/autotrust_JEV-27B-*.jsonl"
    folder = snapshot_download(GYM_RUNS, repo_type="dataset", allow_patterns=[pattern])
    grades = {}
    for path in glob.glob(f"{folder}/{pattern}"):
        for line in open(path):
            g = json.loads(line)
            if g["query_set"] == c["query_set"] and g["parsed_ok"]:
                grades.setdefault(g["qid"], {})[g["doc_id"]] = g["grade"]

    # the predictions table names a generated query set without its last part
    query_set = c["query_set"] if c["query_source"] == "original" else c["query_set"].rsplit("-", 1)[0]
    corp = corpus.load(task)
    skip_self = corp.ignore_identical_ids
    models = sorted(c["models"])
    top = {m: {} for m in models}
    for line in open(hf_hub_download(GYM_RUNS, f"tables/predictions/{task}.jsonl", repo_type="dataset")):
        p = json.loads(line)
        if p["query_set"] == query_set and p["model"] in top and p["revision"] == c["model_revisions"][p["model"]]:
            top[p["model"]][p["qid"]] = [d for d in p["doc_ids"] if not (skip_self and d == p["qid"])][:10]
    queries = sorted(q for q in grades if all(q in top[m] for m in models))
    # keep only documents some model returned in its top 10
    grades = {q: {d: g for d, g in grades[q].items() if any(d in top[m][q] for m in models)} for q in queries}

    # the published ratings must come out of these files exactly
    rebuilt = {m: 100 * np.mean([ndcg(top[m][q], grades[q], grades[q].values(), 10) for q in queries]) for m in models}
    gap = max(abs(rebuilt[m] - ratings(record)[m]) for m in models)
    assert gap < 1e-6, f"{task}: rebuilt ratings differ from the record by {gap}"
    return models, queries, top, grades, {q: dict(corp.qrels.get(q, {})) for q in queries}


def arena_battles():
    """MTEB Arena's retrieval battles (mteb/arena-results): the query, the top document each side showed, and the vote."""
    votes = {"leftvote": "A", "rightvote": "B", "tievote": "tie", "bothbadvote": "both_bad"}
    folder = snapshot_download("mteb/arena-results", repo_type="dataset", allow_patterns=["data/retrieval_battle-*.jsonl"])
    for path in glob.glob(f"{folder}/data/retrieval_battle-*.jsonl"):
        for line in open(path, errors="replace"):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("type") not in votes:
                continue
            a = [str(x) for hit in r.get("0_output") or [] for x in hit[1:]]
            b = [str(x) for hit in r.get("1_output") or [] for x in hit[1:]]
            if a and b:
                yield {"id": f"{r.get('0_conv_id')}__{r.get('1_conv_id')}__{r['type']}", "query": str(r["0_prompt"]),
                       "a": a[0], "b": b[0], "ma": r["0_model_name"], "mb": r["1_model_name"],
                       "vote": votes[r["type"]], "corpus": str(r.get("0_corpus"))}


def num(x):
    """.742 for a correlation, 12.3 for anything larger."""
    return f"{x:.3f}".replace("0.", ".", 1) if abs(x) < 1 else f"{x:.1f}"


def interval(low, high):
    return f"[{num(low)}, {num(high)}]"


def save_csv(name, rows):
    TABLES.mkdir(exist_ok=True)
    with open(TABLES / f"{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def write(name, rows, caption="", wide=False, bold=(), rules=()):
    """Write a table to tables/<name>.csv and tables/<name>.tex (booktabs). A wide table spans both columns; rows
    whose index is in bold are set in bold; a rule is drawn before each row index in rules."""
    save_csv(name, rows)
    cols = list(rows[0])

    def cell(x, strong=False):
        text = num(x) if isinstance(x, float) else str(x).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")
        return rf"\textbf{{{text}}}" if strong and text else text

    env = "table*" if wide else "table"
    align = "".join("l" if isinstance(rows[0][c], str) else "r" for c in cols)
    tex = [rf"\begin{{{env}}}[t]", r"\centering\small", r"\setlength{\tabcolsep}{5pt}", rf"\begin{{tabular}}{{{align}}}",
           r"\toprule", " & ".join(rf"\textbf{{{cell(c)}}}" if c else "" for c in cols) + r" \\", r"\midrule"]
    for i, r in enumerate(rows):
        if i in rules:
            tex.append(r"\midrule")
        tex.append(" & ".join(cell(r[c], i in bold) for c in cols) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}", rf"\caption{{{caption}}}", rf"\label{{tab:{name}}}", rf"\end{{{env}}}"]
    (TABLES / f"{name}.tex").write_text("\n".join(tex) + "\n")
    for r in rows:
        print("  ".join(cell(r[c]) for c in cols))
