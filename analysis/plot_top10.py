"""Figure: the top 10 models by MTEB's labels (left) and by the gym (right), averaged over the 28 tasks.

The gym uses generated queries, JEV-27B grading each document. A line joins each model's two places; a model in only one
top 10 shows its rank in the other list in gray."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

from common import GENERATOR, JUDGE, TASKS, labels, ratings, records
from plots import ORANGE, GRAY, INK, BLUE, save

R = records()
per_task = {t: ratings(R[(JUDGE, "pointwise", GENERATOR)][t]) for t in TASKS}
models = sorted(per_task[TASKS[0]])
by_gym = dict(zip(models, rankdata([-np.mean([per_task[t][m] for t in TASKS]) for m in models])))
by_label = dict(zip(models, rankdata([-np.mean([labels(t)[m] for t in TASKS]) for m in models])))
TOP = 10
name = lambda m: m.split("/")[-1].replace("baseline-bm25s", "BM25")

fig, ax = plt.subplots(figsize=(8, 6.2))
for m in models:
    a, b = by_label[m], by_gym[m]
    if a > TOP and b > TOP:
        continue
    same = abs(a - b) <= 1
    color = BLUE if same else ORANGE
    ya, yb = min(a, TOP + 3), min(b, TOP + 3)
    ax.plot([0, 1], [ya, yb], color=color, lw=3 if same else 2.5, alpha=0.9, solid_capstyle="round")
    if a <= TOP:
        ax.text(-0.04, a, f"{name(m)}  {int(a)}", ha="right", va="center", fontsize=14, color=INK)
    if b <= TOP:
        ax.text(1.04, b, f"{int(b)}  {name(m)}", ha="left", va="center", fontsize=14, color=INK)
    for x, r, y in ((0, a, ya), (1, b, yb)):
        if r > TOP:
            ax.text(x + (-0.04 if x == 0 else 0.04), y, f"{int(r)}", ha="right" if x == 0 else "left", va="center",
                    fontsize=13, color=GRAY)
ax.text(0, 0.0, "MTEB's labels", ha="center", fontsize=17, fontweight="bold", color=INK)
ax.text(1, 0.0, "MTEB Gym", ha="center", fontsize=17, fontweight="bold", color=BLUE)
ax.set_xlim(-0.9, 1.9)
ax.set_ylim(TOP + 3.5, -0.6)
ax.axis("off")
save(fig, "top10")
