"""Figure: how much the gym needs to judge. Agreement with the labels as the number of queries, the number of models
ranked, and the number of documents graded per query change.

Reads the per-task values that num_queries.py, num_models.py and num_documents.py write; each point is the mean over the
28 tasks."""

import csv
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from common import GENERATOR, TABLES
from plots import BLUE, GEN, OWN, RED, save

PANELS = [("num_queries", "Queries per task", [10, 25, 50, 100]),
          ("num_models", "Models ranked", [5, 10, 15, 20, 28]),
          ("num_documents", "Documents graded per query", [1, 3, 5, 10])]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True, gridspec_kw={"wspace": 0.12})
for ax, (name, xlabel, xs) in zip(axes, PANELS):
    values = defaultdict(list)
    for r in csv.DictReader(open(TABLES / f"{name}_per_task.csv")):
        values[(r["queries"], int(r["n"]))].append(float(r["agreement"]))
    for queries, style in (("original", OWN), (GENERATOR, GEN)):
        ax.plot(xs, [np.nanmean(values[(queries, x)]) for x in xs], color=style["color"], marker=style["marker"],
                ms=14, mec="white", mew=1)
    ax.set_xlabel(xlabel)
    ax.set_xticks(xs)
axes[0].set_xscale("log")
axes[0].set_xticks([10, 25, 50, 100], ["10", "25", "50", "100"])
axes[0].minorticks_off()
axes[0].set_ylabel("Agreement")
axes[0].set_ylim(0.58, 0.8)
axes[0].text(100, 0.765, "Tasks' own queries", color=BLUE, ha="right", fontsize=17)
axes[0].text(100, 0.68, "Generated queries", color=RED, ha="right", fontsize=17)
save(fig, "how_much")
