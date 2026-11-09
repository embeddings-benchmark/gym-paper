"""Figure: how well different ways of ranking the 28 models agree with the ranking from MTEB's labels.

Simple properties of the models from their mteb metadata (release date, max input length, embedding size, number of
parameters), and the gym. Left: the overall ranking (mean over the 28 tasks). Right: per task, averaged over tasks.
Models without a value for a property (BM25 and ColBERT have no embedding size) are left out of that row."""

import datetime
import math

import matplotlib.pyplot as plt
import mteb
import numpy as np
from scipy.stats import spearmanr

from common import GENERATOR, JUDGE, TASKS, labels, ratings, records
from plots import GRAY, INK, LIGHT, NAVY, save

R = records()
models = sorted(labels(TASKS[0]))
meta = {m: mteb.get_model_meta(m) for m in models}


def prop(m, key):
    v = getattr(meta[m], key)
    if key == "embed_dim" and isinstance(v, list):
        v = max(v)
    if key == "release_date":
        v = datetime.date.fromisoformat(str(v)).toordinal()
    if key == "max_tokens" and v is not None and math.isinf(v):
        v = 1e9  # no limit
    return v


def agreement_of(score):
    """score: {task: {model: value}} or {model: value} for all tasks. Returns (overall, mean per task)."""
    per_task = {t: score.get(t, score) for t in TASKS}
    ms = [m for m in models if all(per_task[t].get(m) is not None for t in TASKS)]
    overall = spearmanr([np.mean([per_task[t][m] for t in TASKS]) for m in ms],
                        [np.mean([labels(t)[m] for t in TASKS]) for m in ms])[0]
    each = np.mean([spearmanr([per_task[t][m] for m in ms], [labels(t)[m] for m in ms])[0] for t in TASKS])
    return overall, each


ROWS = [("Release date", {m: prop(m, "release_date") for m in models}, False),
        ("Max input length", {m: prop(m, "max_tokens") for m in models}, False),
        ("Embedding size", {m: prop(m, "embed_dim") for m in models}, False),
        ("Model size", {m: prop(m, "n_parameters") for m in models}, False),
        ("MTEB Gym, tasks' own queries", {t: ratings(R[(JUDGE, "pointwise", "original")][t]) for t in TASKS}, True),
        ("MTEB Gym", {t: ratings(R[(JUDGE, "pointwise", GENERATOR)][t]) for t in TASKS}, True)]
values = [(name, *agreement_of(score), gym) for name, score, gym in ROWS]

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True, gridspec_kw={"wspace": 0.08})
for ax, col, title in ((axes[0], 1, "Overall ranking"), (axes[1], 2, "Per task")):
    for i, row in enumerate(values):
        name, v, gym = row[0], row[col], row[3]
        main = name == "MTEB Gym"
        ax.barh(i, v, height=0.68, color=NAVY if main else ("#8DA9CF" if gym else LIGHT), zorder=2)
        ax.text(v + 0.015, i, f"{v:.2f}", va="center", fontsize=16, color=NAVY if main else INK,
                fontweight="bold" if main else "normal")
    ax.set_xlim(0, 1.08)
    ax.set_xticks([0, 0.5, 1], ["0", ".5", "1"])
    ax.set_title(title, pad=10)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
axes[0].set_yticks(range(len(values)), [v[0] for v in values])
for label in axes[0].get_yticklabels():
    if label.get_text() == "MTEB Gym":
        label.set_fontweight("bold")
        label.set_color(NAVY)
fig.supxlabel("Spearman correlation with the ranking from MTEB's labels", fontsize=17, y=-0.03)
save(fig, "proxies")
