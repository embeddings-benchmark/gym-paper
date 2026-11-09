"""Figure: agreement with the labels' ranking on each of the 28 tasks, one block per benchmark.

JEV-27B grading each document. Left column of each block: generated queries (the gym); right: the tasks' own queries.
Tasks sorted by the gym's agreement."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from common import GENERATOR, JUDGE, TASKS, agreement, ratings, records
from plots import BLUE, BLUE_PALE, BLUE_TINT, INK, NAMES, ORANGE_PALE, save

R = records()
OTHER = ["SciFact", "NFCorpus", "FinQARetrieval", "AppsRetrieval", "AILACasedocs", "CUREv1"]
GROUPS = [("MTEB(eng, v2)", [t for t in TASKS if not t.startswith("Bright") and t not in OTHER]),
          ("BRIGHT", [t for t in TASKS if t.startswith("Bright")]),
          ("Other domains", OTHER)]
COLS = [("MTEB\nGym", GENERATOR), ("Own\nqueries", "original")]
values = {q: {t: agreement(ratings(R[(JUDGE, "pointwise", q)][t]), t) for t in TASKS} for _, q in COLS}
cmap = LinearSegmentedColormap.from_list("blue", [ORANGE_PALE, BLUE_PALE, BLUE_TINT, BLUE])
rows = max(len(ts) for _, ts in GROUPS)

fig, axes = plt.subplots(1, 3, figsize=(15, 6.2), gridspec_kw={"wspace": 0.95})
for ax, (group, ts) in zip(axes, GROUPS):
    ts = sorted(ts, key=lambda t: -values[GENERATOR][t])
    grid = np.array([[values[q][t] for _, q in COLS] for t in ts])
    ax.imshow(grid, cmap=cmap, vmin=-0.5, vmax=1, aspect="auto", extent=(-0.5, 1.5, len(ts) - 0.5, -0.5))
    for (r, c), v in np.ndenumerate(grid):
        ax.text(c, r, f"{v:.2f}".replace("0.", ".", 1), ha="center", va="center", fontsize=14,
                color="white" if v > 0.72 else INK, fontweight="bold" if c == 0 else "normal")
    for r in range(len(ts) + 1):  # white lines between cells
        ax.axhline(r - 0.5, color="white", lw=2)
    ax.axvline(0.5, color="white", lw=2)
    ax.set_ylim(rows - 0.5, -0.5)
    ax.set_yticks(range(len(ts)), [NAMES[t] for t in ts], fontsize=14)
    ax.set_xticks([0, 1], [name for name, _ in COLS], fontsize=13)
    ax.xaxis.tick_top()
    ax.get_xticklabels()[0].set_fontweight("bold")
    ax.get_xticklabels()[0].set_color(BLUE)
    ax.tick_params(length=0)
    ax.set_title(group, fontsize=17, fontweight="bold", pad=52)
    for side in ax.spines.values():
        side.set_visible(False)
save(fig, "task_strip")
