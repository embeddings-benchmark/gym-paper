"""Appendix figure: agreement with the labels' ranking on each of the 28 tasks, grouped by benchmark.

JEV-27B grading each document; blue: generated queries (the gym), orange: the tasks' own queries. Within a benchmark,
tasks are sorted by the gym's agreement. Dashed: the gym's mean over the 28 tasks."""

import matplotlib.pyplot as plt
import numpy as np

from common import GENERATOR, JUDGE, TASKS, agreement, ratings, records
from plots import BLUE, GEN, GRAY, INK, LIGHT, NAMES, ORANGE, OWN, save

R = records()
OTHER = ["SciFact", "NFCorpus", "FinQARetrieval", "AppsRetrieval", "AILACasedocs", "CUREv1"]
GROUPS = [("MTEB(eng, v2)", [t for t in TASKS if not t.startswith("Bright") and t not in OTHER]),
          ("BRIGHT", [t for t in TASKS if t.startswith("Bright")]),
          ("Other domains", OTHER)]
gen = {t: agreement(ratings(R[(JUDGE, "pointwise", GENERATOR)][t]), t) for t in TASKS}
own = {t: agreement(ratings(R[(JUDGE, "pointwise", "original")][t]), t) for t in TASKS}

rows, headers, y = [], [], 0.0
for group, ts in GROUPS:
    headers.append((y, group))
    y += 1
    for t in sorted(ts, key=lambda t: -gen[t]):
        rows.append((y, t))
        y += 1
    y += 0.6

fig, ax = plt.subplots(figsize=(8, 11))
for yy, t in rows:
    ax.plot([gen[t], own[t]], [yy, yy], color=LIGHT, lw=4, solid_capstyle="round", zorder=1)
ax.scatter([own[t] for _, t in rows], [yy for yy, _ in rows], s=130, edgecolor="white", lw=1.2, zorder=3, **{
    k: v for k, v in OWN.items() if k != "label"})
ax.scatter([gen[t] for _, t in rows], [yy for yy, _ in rows], s=150, edgecolor="white", lw=1.2, zorder=4, **{
    k: v for k, v in GEN.items() if k != "label"})
ax.axvline(np.mean(list(gen.values())), color=BLUE, lw=1.5, ls=(0, (4, 3)), zorder=0)
ax.axvline(0, color=GRAY, lw=1, zorder=0)
for yy, group in headers:
    ax.text(-0.62, yy, group, fontsize=16, fontweight="bold", color=INK, va="center", ha="left")
ax.set_yticks([yy for yy, _ in rows], [NAMES[t] for _, t in rows], fontsize=14)
ax.set_ylim(y - 0.4, -0.8)
ax.set_xlim(-0.62, 1.02)
ax.set_xticks([-0.5, 0, 0.5, 1], ["−0.5", "0", ".5", "1"])
ax.set_xlabel("Agreement with the labels' ranking")
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
ax.text(-0.58, 3, "Generated queries", color=BLUE, fontsize=15, fontweight="bold")
ax.text(-0.58, 4, "Tasks' own queries", color=ORANGE, fontsize=15, fontweight="bold")
ax.text(-0.58, 5.2, f"dashed: gym mean {np.mean(list(gen.values())):.2f}".replace("0.", "."), color=BLUE,
        fontsize=13)
save(fig, "per_task")
