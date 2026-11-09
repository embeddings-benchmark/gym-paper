"""Figure 1: the gym's overall ranking of the 28 models against the ranking from MTEB's labels on the same 28 tasks.

A model's overall score is its mean over the tasks: the gym's score with generated queries (JEV-27B grading each
document), and nDCG@10 on the task's own labels. Each point is one model; rho is the Spearman correlation."""

import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

from common import GENERATOR, JUDGE, TASKS, labels, overall_agreement, ratings, records
from plots import CORAL, GRAY, INK, MAIN, save

R = records()
per_task = {t: ratings(R[(JUDGE, "pointwise", GENERATOR)][t]) for t in TASKS}
models = sorted(per_task[TASKS[0]])
gym = {m: np.mean([per_task[t][m] for t in TASKS]) for m in models}
truth = {m: np.mean([labels(t)[m] for t in TASKS]) for m in models}
by_gym = dict(zip(models, rankdata([-gym[m] for m in models])))
by_label = dict(zip(models, rankdata([-truth[m] for m in models])))
rho = overall_agreement(per_task)


def draw(ax, color):
    ax.plot([0, 29], [0, 29], color=GRAY, lw=1.5, ls=(0, (4, 3)), zorder=1)
    ax.scatter([by_label[m] for m in models], [by_gym[m] for m in models], s=150, color=color, edgecolor="white",
               lw=1.2, zorder=3)
    # the top three: ringed, and named in the empty lower right
    top = ["Qwen/Qwen3-Embedding-8B", "Qwen/Qwen3-Embedding-4B", "Linq-AI-Research/Linq-Embed-Mistral"]
    ax.scatter([by_label[m] for m in top], [by_gym[m] for m in top], s=420, facecolor="none", edgecolor=CORAL, lw=2.2,
               zorder=4)
    names = "\n".join(f"{i + 1}  {m.split('/')[-1]}" for i, m in enumerate(top))
    ax.text(0.96, 0.04, names, transform=ax.transAxes, ha="right", va="bottom", fontsize=14, color=INK, linespacing=1.6, multialignment="left",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor=CORAL, lw=1.8))
    ax.text(0.96, 0.27, "Top 3 by both", transform=ax.transAxes, ha="right", va="bottom", fontsize=14, color=CORAL,
            fontweight="bold")
    ax.text(27.5, 1.2, f"ρ = {rho:.2f}", fontsize=32, color=color, fontweight="bold", va="top")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_ticks([1, 7, 14, 21, 28])
    ax.set_xlim(29.2, -0.2)
    ax.set_ylim(29.2, -0.2)
    ax.set_aspect("equal")
    ax.set_xlabel("Rank by MTEB's labels")
    ax.set_ylabel("Rank by MTEB Gym")


if len(sys.argv) > 1:  # python plot_overall.py "#hex" "#hex" ...: the figure in several colors, side by side
    fig, axes = plt.subplots(1, len(sys.argv) - 1, figsize=(6.5 * (len(sys.argv) - 1), 6.5))
    for ax, color in zip(axes, sys.argv[1:]):
        draw(ax, color)
        ax.set_title(color, fontsize=14, color=GRAY)
    save(fig, "overall_colors")
else:
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    draw(ax, MAIN)
    save(fig, "overall")
