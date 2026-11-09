"""Figure: the overall ranking needs few judgments. Agreement of the gym's overall ranking with the labels' as the
number of queries per task and the number of documents graded per model and query shrink.

Reads tables/num_queries_overall.csv and tables/num_documents_overall.csv (from num_queries.py and num_documents.py).
Queries: mean over 20 random draws, band from the 2.5th to the 97.5th percentile of the draws."""

import csv

import matplotlib.pyplot as plt

from common import GENERATOR, TABLES
from plots import CORAL, GEN, GRAY, INK, NAVY, OWN, save


def read(name):
    out = {}
    for r in csv.DictReader(open(TABLES / f"{name}_overall.csv")):
        out.setdefault(r["queries"], []).append({k: float(v) for k, v in r.items() if k != "queries"})
    return out


fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True, gridspec_kw={"wspace": 0.1})
for ax, name, xlabel in ((axes[0], "num_queries", "Queries per task"),
                         (axes[1], "num_documents", "Top-k documents judged")):
    data = read(name)
    for queries, style in (("original", OWN), (GENERATOR, GEN)):
        rows = data[queries]
        xs = [r["n"] for r in rows]
        if "low" in rows[0]:
            ax.fill_between(xs, [r["low"] for r in rows], [r["high"] for r in rows], color=style["color"], alpha=0.15,
                            lw=0)
        ax.plot(xs, [r["agreement"] for r in rows], color=style["color"], marker=style["marker"], ms=12, mec="white",
                mew=1.2, lw=3.5)
    ax.set_xscale("log")
    ax.set_xticks(xs, [str(int(x)) for x in xs])
    ax.minorticks_off()
    ax.set_xlabel(xlabel)
    ax.invert_xaxis()  # fewer judgments to the right
axes[0].set_ylim(0.85, 1.0)
axes[0].set_yticks([0.85, 0.9, 0.95, 1.0], [".85", ".90", ".95", "1"])
axes[0].set_ylabel("Agreement, overall ranking")
axes[0].text(100, 0.875, "Generated queries", color=NAVY, fontsize=16, fontweight="bold")
axes[0].text(100, 0.86, "Tasks' own queries", color=CORAL, fontsize=16, fontweight="bold")
for ax, x, y, xt in ((axes[0], 10, 0.947, 17), (axes[1], 1, 0.955, 1.9)):
    ax.annotate("10× fewer,\nsame agreement", xy=(x, y - 0.004), xytext=(xt, 0.905), fontsize=15, color=INK, ha="center",
                arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.5, connectionstyle="arc3,rad=-0.25"))
save(fig, "how_much")
