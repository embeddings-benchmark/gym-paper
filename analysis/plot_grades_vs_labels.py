"""Appendix figure: the judge's grades against the tasks' own relevance labels.

Every document the judge graded for the original queries (JEV-27B, 28 tasks), by its relevance label: not relevant
(no label or 0), relevant (1), highly relevant (2 or more). Each row shows how that row's documents spread over the
judge's grade, rounded to 0-5; rows sum to 100%."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from common import JUDGE, TASKS, judged, records
from plots import BLUE, INK, save

R = records()
LEVELS = ["Not relevant", "Relevant", "Highly\nrelevant"]
counts = np.zeros((3, 6))
for t in TASKS:
    _, qids, _, grades, qrels = judged(R[(JUDGE, "pointwise", "original")][t])
    for q in qids:
        for d, g in grades[q].items():
            counts[min(qrels[q].get(d, 0), 2), int(round(g))] += 1
share = 100 * counts / counts.sum(1, keepdims=True)
print(counts.sum(1))

cmap = LinearSegmentedColormap.from_list("blue", ["#ffffff", BLUE])
fig, ax = plt.subplots(figsize=(9, 4))
ax.imshow(share, cmap=cmap, vmin=0, vmax=60, aspect="auto")
for (r, c), v in np.ndenumerate(share):
    ax.text(c, r, f"{v:.0f}%", ha="center", va="center", fontsize=16, color="white" if v > 33 else INK,
            fontweight="bold" if v == share[r].max() else "normal")
ax.set_xticks(range(6), [str(g) for g in range(6)])
ax.set_yticks(range(3), [f"{name}\n({int(n):,} docs)" for name, n in zip(LEVELS, counts.sum(1))], fontsize=14)
ax.set_xlabel("Judge's grade")
ax.set_ylabel("Human label")
ax.tick_params(length=0)
for side in ax.spines.values():
    side.set_visible(False)
save(fig, "grades_vs_labels")
