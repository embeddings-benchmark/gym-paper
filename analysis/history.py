"""Leaderboard history baselines, which rank models for a task without using that task's labels or any judging.

- "history (labels)": a model's mean rank over the other 27 tasks' labelled nDCG@10. This is how the earlier .67 and
  4.82 numbers were made (inline, in the session of 2026-11-06; there was no saved script).
- "history (published)": a model's mean rank over the other 27 tasks' published MTEB scores, from mteb's result cache
  (`mteb_gym.agreement.official_scores`), counting each task only for the models that have a score on it.
"""

from __future__ import annotations

import json
import logging

import numpy as np

from common import DATA, TASKS, labels, rank_of

PUBLISHED = DATA / "published_scores.json"


def published() -> dict[str, dict[str, float]]:
    """{task: {model: published main score}}, fetched once into data/ and read from there afterwards."""
    if not PUBLISHED.exists():
        import mteb

        from mteb_gym.agreement import official_scores

        logging.disable(logging.WARNING)
        models = sorted(labels(TASKS[0]))
        cache = mteb.ResultCache()
        data = {t: {m: float(s) for m, (s, _) in official_scores(cache, t, models).items()} for t in TASKS}
        PUBLISHED.write_text(json.dumps(data, indent=1))
    return json.loads(PUBLISHED.read_text())


def history(task: str, source: dict[str, dict[str, float]]) -> dict[str, float]:
    """Higher is better: minus the mean rank over the other tasks in `source` where the model has a score."""
    models = sorted(labels(task))
    ranks: dict[str, list[float]] = {m: [] for m in models}
    for other in TASKS:
        if other == task:
            continue
        have = [m for m in models if m in source.get(other, {})]
        if len(have) < 2:
            continue
        for m, r in zip(have, rank_of(source[other], have)):
            ranks[m].append(r / len(have))  # a rank as a share of the models ranked there
    return {m: -float(np.mean(v)) if v else float("nan") for m, v in ranks.items()}


def label_history(task: str) -> dict[str, float]:
    return history(task, {t: labels(t) for t in TASKS})


def published_history(task: str) -> dict[str, float]:
    return history(task, published())
