"""Figure 1: each model's rank by the gym against its rank by the task's own labels (nDCG@10), on three tasks.

JEV-27B grades each document. Each point is one of the 28 models; the diagonal is perfect agreement; rho is the Spearman
correlation."""

import matplotlib.pyplot as plt
from scipy.stats import rankdata

from common import GENERATOR, JUDGE, agreement, labels, ratings, records
from plots import GEN, GRAY, OWN, save

R = records()
TASKS = [("NFCorpus", "Medical\n(NFCorpus)"), ("FiQA2018", "Finance\n(FiQA-2018)"),
         ("BrightBiologyRetrieval", "Biology reasoning\n(BRIGHT)")]


def ranks(scores, models):
    """1 = best."""
    return dict(zip(models, rankdata([-scores[m] for m in models])))


fig, axes = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw={"wspace": 0.22})
for ax, (task, title) in zip(axes, TASKS):
    truth = labels(task)
    ax.plot([0, 29], [0, 29], color=GRAY, lw=2, ls=(0, (4, 3)), zorder=1)
    for queries, style, y in (("original", OWN, 0.97), (GENERATOR, GEN, 0.86)):
        gym = ratings(R[(JUDGE, "pointwise", queries)][task])
        models = sorted(gym)
        by_label, by_gym = ranks(truth, models), ranks(gym, models)
        ax.scatter([by_label[m] for m in models], [by_gym[m] for m in models], s=120, color=style["color"],
                   marker=style["marker"], edgecolor="white", lw=0.8, alpha=0.9, zorder=3)
        ax.text(0.03, y, f"ρ = {agreement(gym, task):.2f}", transform=ax.transAxes, va="top",
                color=style["color"])
    ax.set_title(title, pad=10)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_ticks([1, 10, 20, 28])
    ax.set_xlim(29.5, -0.5)
    ax.set_ylim(29.5, -0.5)
    ax.set_aspect("equal")
axes[0].set_ylabel("Rank by the gym")
axes[1].set_xlabel("Rank by the task's own labels")

handles = [plt.Line2D([], [], ls="", marker=st["marker"], color=st["color"], ms=12) for st in (OWN, GEN)]
leg = fig.legend(handles, [OWN["label"], GEN["label"]], loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
                 frameon=True, markerfirst=False, handletextpad=0.3, borderpad=0.4, columnspacing=1.5)
leg.get_frame().set(edgecolor="black", linewidth=1.5)
for text, st in zip(leg.get_texts(), (OWN, GEN)):
    text.set_color(st["color"])
save(fig, "models")
