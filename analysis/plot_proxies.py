"""Figure: how well different ways of ranking the 28 models agree with the ranking from MTEB's labels (Spearman).

Simple properties of the models from their mteb metadata (release date, max input length, embedding size, number of
parameters), and the gym. Top row: the overall ranking (mean over the 28 tasks). Bottom row: per task, averaged over
tasks. Models without a value for a property (BM25 and ColBERT have no embedding size) are left out of that column."""

import datetime
import math

import matplotlib.pyplot as plt
import mteb
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import spearmanr

from common import GENERATOR, JUDGE, TASKS, labels, ratings, records
from plots import INK, NAVY, save

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
    """score: {task: {model: value}}, or {model: value} for every task. Returns (overall, mean per task)."""
    per_task = {t: score.get(t, score) for t in TASKS}
    ms = [m for m in models if all(per_task[t].get(m) is not None for t in TASKS)]
    overall = spearmanr([np.mean([per_task[t][m] for t in TASKS]) for m in ms],
                        [np.mean([labels(t)[m] for t in TASKS]) for m in ms])[0]
    each = np.mean([spearmanr([per_task[t][m] for m in ms], [labels(t)[m] for m in ms])[0] for t in TASKS])
    return overall, each


COLUMNS = [("Release\ndate", {m: prop(m, "release_date") for m in models}),
           ("Max input\nlength", {m: prop(m, "max_tokens") for m in models}),
           ("Embedding\nsize", {m: prop(m, "embed_dim") for m in models}),
           ("Model\nsize", {m: prop(m, "n_parameters") for m in models}),
           ("MTEB Gym", {t: ratings(R[(JUDGE, "pointwise", GENERATOR)][t]) for t in TASKS})]
values = np.array([agreement_of(score) for _, score in COLUMNS]).T  # rows: overall, per task

cmap = LinearSegmentedColormap.from_list("navy", ["#f2f5fa", "#9db6d8", NAVY])
fig, ax = plt.subplots(figsize=(10, 3.2))
ax.imshow(values, cmap=cmap, vmin=0.2, vmax=1.0, aspect="auto")
for (r, c), v in np.ndenumerate(values):
    ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=20, color="white" if v > 0.72 else INK,
            fontweight="bold" if c == len(COLUMNS) - 1 else "normal")
ax.axvline(len(COLUMNS) - 1.5, color="white", lw=6)  # set the gym apart
ax.set_xticks(range(len(COLUMNS)), [name for name, _ in COLUMNS])
ax.set_yticks([0, 1], ["Overall\nranking", "Per task"])
ax.tick_params(length=0)
for side in ax.spines.values():
    side.set_visible(False)
ax.xaxis.tick_top()
gym = ax.get_xticklabels()[-1]
gym.set_fontweight("bold")
gym.set_color(NAVY)
ax.set_title("Spearman correlation with the ranking from MTEB's labels", fontsize=17, pad=58)
save(fig, "proxies")
