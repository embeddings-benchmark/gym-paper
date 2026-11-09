"""Figure: agreement on each task with the tasks' own queries against agreement with generated queries.

JEV-27B grades each document. Each point is one of the 28 tasks; on the diagonal, generated queries do as well as the
task's own. The tasks furthest from it, or with the lowest agreement, are named."""

import matplotlib.pyplot as plt

from common import GENERATOR, JUDGE, TASKS, agreement, ratings, records
from plots import CORAL, GRAY, INK, NAMES, NAVY, save

R = records()
own = {t: agreement(ratings(R[(JUDGE, "pointwise", "original")][t]), t) for t in TASKS}
gen = {t: agreement(ratings(R[(JUDGE, "pointwise", GENERATOR)][t]), t) for t in TASKS}

fig, ax = plt.subplots(figsize=(7, 6.3))
ax.plot([-0.6, 1.05], [-0.6, 1.05], color=GRAY, lw=2, ls=(0, (4, 3)), zorder=1)
ax.axhline(0, color=GRAY, lw=1)
ax.axvline(0, color=GRAY, lw=1)
ax.scatter([own[t] for t in TASKS], [gen[t] for t in TASKS], s=150, color=NAVY, edgecolor="white", lw=1, zorder=3)

# names for the tasks that stand out: label positions are set by hand so nothing overlaps
LABELS = {"HotpotQAHardNegatives": (-0.04, 0.0, "right"), "FEVERHardNegatives": (0.04, -0.04, "left"),
          "ClimateFEVERHardNegatives": (0.04, 0.0, "left"), "BrightPonyRetrieval": (0.04, 0.0, "left"),
          "BrightAopsRetrieval": (-0.04, 0.02, "right"), "BrightEconomicsRetrieval": (0.035, 0.0, "left")}
for t, (dx, dy, ha) in LABELS.items():
    ax.annotate(NAMES[t], (own[t], gen[t]), (own[t] + dx, gen[t] + dy), ha=ha, va="center", fontsize=16, color=INK)

ax.text(-0.55, 0.97, "Generated\nqueries\ndo better", color=GRAY, va="top", fontsize=16)
ax.text(0.98, -0.52, "Tasks' own queries\ndo better", color=GRAY, ha="right", va="bottom", fontsize=16)
ax.set_xlim(-0.6, 1.05)
ax.set_ylim(-0.6, 1.05)
ax.set_aspect("equal")
ticks = [-0.5, 0, 0.5, 1]
ax.set_xticks(ticks, ["−0.5", "0", "0.5", "1"])
ax.set_yticks(ticks, ["−0.5", "0", "0.5", "1"])
ax.set_xlabel("Agreement, tasks' own queries")
ax.set_ylabel("Agreement, generated queries")
save(fig, "per_task")
