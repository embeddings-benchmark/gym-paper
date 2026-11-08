"""Shared loading for the paper analyses: records by task and settings, label references, and table output.

Records come from the published results repository (embeddings-benchmark/gym-results), fetched and cached the way
`mteb_gym.load_results` fetches it; set GYM_RESULTS to a local results folder to use that instead."""

from __future__ import annotations

import csv
import glob
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, rankdata, spearmanr

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "out"
TASKS = (DATA / "tasks.txt").read_text().split()
EXCLUDED = {"BrightAopsRetrieval", "BrightLeetcodeRetrieval", "BrightTheoremQAQuestionsRetrieval"}
GYM_RUNS = "mteb/gym-runs"  # the queries, predictions, grades and verdicts behind each record
MAIN_JUDGE = "JEV-27B"
MAIN_GENERATOR = "DeepSeek-V4-Flash"
BOOT = 10_000


def short(name: str | None) -> str:
    return "original" if not name else name.split("+")[0].split("/")[-1]


def labels(task: str) -> dict[str, float]:
    """nDCG@10 of each model against the task's labels on the 100 sampled original queries, from data/labels (rebuilt
    and checked by labels.py). The three BRIGHT tasks with excluded documents use the version that drops them, as mteb
    does."""
    return json.loads((DATA / "labels" / f"{task}.json").read_text())


def results_folder() -> Path:
    if os.environ.get("GYM_RESULTS"):
        return Path(os.environ["GYM_RESULTS"])
    from mteb_gym.results import download_results

    return download_results(download_latest=False) / "results"


def records() -> dict[tuple[str, str, str], dict[str, dict]]:
    """Every record as {(judge, judging, queries): {task: record}}, where queries is "original" or the generator."""
    out: dict = {}
    root = results_folder()
    for f in sorted(glob.glob(str(root / "*" / "*.json"))):
        r = json.loads(Path(f).read_text())
        c = r["config"]
        queries = "original" if c["query_source"] == "original" else short(c.get("generator_model"))
        key = (short(c["judge_model"]), c["judging"], queries)
        if r["task_name"] in out.get(key, {}):
            raise ValueError(f"two records for {key} on {r['task_name']}")
        r["_path"] = str(Path(f).relative_to(root))
        out.setdefault(key, {})[r["task_name"]] = r
    return out


def ratings(record: dict) -> dict[str, float]:
    return {x["model"]: x["rating"] for x in record["ratings"]}


def agree(scores: dict[str, float], truth: dict[str, float]) -> tuple[float, float]:
    models = sorted(set(scores) & set(truth))
    a, b = [scores[m] for m in models], [truth[m] for m in models]
    return float(spearmanr(a, b)[0]), float(kendalltau(a, b)[0])


def rank_of(scores: dict[str, float], models: list[str]) -> np.ndarray:
    """1 = best; ties share the average rank."""
    return rankdata([-scores[m] for m in models])


def boot_mean(values, seed: int = 0) -> tuple[float, float, float]:
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (BOOT, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def write(name: str, rows: list[dict], note: str = "") -> None:
    """A table as CSV and as Markdown, in out/."""
    OUT.mkdir(exist_ok=True)
    cols = list(rows[0])
    with open(OUT / f"{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows(rows)
    fmt = lambda x: f"{x:.3f}" if isinstance(x, float) else str(x)
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for r in rows]
    (OUT / f"{name}.md").write_text((note + "\n\n" if note else "") + "\n".join(md) + "\n")
