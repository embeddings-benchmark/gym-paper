"""Figure: agreement with the labels' ranking on each of the 28 tasks, as a strip grouped by benchmark.

JEV-27B grading each document. Top row: generated queries (the gym); bottom row: the tasks' own queries."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from common import GENERATOR, JUDGE, TASKS, agreement, ratings, records
from plots import INK, NAMES, NAVY, save

R = records()
GROUPS = [("MTEB(eng, v2)", [t for t in TASKS if not t.startswith("Bright") and t not in
                             ("SciFact", "NFCorpus", "FinQARetrieval", "AppsRetrieval", "AILACasedocs", "CUREv1")]),
          ("BRIGHT", [t for t in TASKS if t.startswith("Bright")]),
          ("Other domains", ["SciFact", "NFCorpus", "FinQARetrieval", "AppsRetrieval", "AILACasedocs", "CUREv1"])]
rows = [("MTEB Gym", GENERATOR), ("Tasks' own\nqueries", "original")]
values = {q: {t: agreement(ratings(R[(JUDGE, "pointwise", q)][t]), t) for t in TASKS} for _, q in rows}

cmap = LinearSegmentedColormap.from_list("navy", ["#fbe3dc", "#f2f5fa", "#9db6d8", NAVY])
widths = [len(ts) for _, ts in GROUPS]
fig, axes = plt.subplots(1, 3, figsize=(18, 3.6), gridspec_kw={"width_ratios": widths, "wspace": 0.03})
for ax, (group, ts) in zip(axes, GROUPS):
    ts = sorted(ts, key=lambda t: -values[GENERATOR][t])
    grid = np.array([[values[q][t] for t in ts] for _, q in rows])
    ax.imshow(grid, cmap=cmap, vmin=-0.5, vmax=1, aspect="auto")
    for (r, c), v in np.ndenumerate(grid):
        ax.text(c, r, f"{v:.2f}".replace("0.", ".", 1), ha="center", va="center", fontsize=12.5,
                color="white" if v > 0.75 else INK, fontweight="bold" if r == 0 else "normal")
    ax.set_xticks(range(len(ts)), [NAMES[t] for t in ts], rotation=50, ha="right", fontsize=13)
    ax.set_yticks([])
    ax.tick_params(length=0)
    ax.set_title(group, fontsize=16, fontweight="bold", pad=8)
    for side in ax.spines.values():
        side.set_visible(False)
axes[0].set_yticks([0, 1], [name for name, _ in rows], fontsize=14)
axes[0].get_yticklabels()[0].set_fontweight("bold")
axes[0].get_yticklabels()[0].set_color(NAVY)
save(fig, "task_strip")
