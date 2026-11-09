"""Appendix figure: on each task, the Spearman correlation between the gym's ranking of the 28 models and MTEB's
ranking (nDCG@10 on the task's relevance labels).

JEV-27B grading each document. Blue bars: generated queries (the gym); outlined bars behind: the task's original queries. Pony is left
out: on its labels almost every model scores near zero (median nDCG@10 .04), so its reference ranking is mostly noise."""

import matplotlib.pyplot as plt

from common import GENERATOR, JUDGE, TASKS, agreement, ratings, records
from plots import BLUE, GRAY, INK, NAMES, ORANGE, ORANGE_PALE, save

R = records()
OTHER = ["SciFact", "NFCorpus", "FinQARetrieval", "AppsRetrieval", "AILACasedocs", "CUREv1"]
GROUPS = [("MTEB(eng, v2)", [t for t in TASKS if not t.startswith("Bright") and t not in OTHER]),
          ("BRIGHT", [t for t in TASKS if t.startswith("Bright") and t != "BrightPonyRetrieval"]),
          ("Other domains", OTHER)]
NOTES = {"FEVERHardNegatives": "claim verification", "ClimateFEVERHardNegatives": "claim verification",
         "HotpotQAHardNegatives": "multi-hop questions", "BrightAopsRetrieval": "labels near zero for all models"}
gen = {t: agreement(ratings(R[(JUDGE, "pointwise", GENERATOR)][t]), t) for t in TASKS}
real = {t: agreement(ratings(R[(JUDGE, "pointwise", "original")][t]), t) for t in TASKS}

fig, axes = plt.subplots(3, 1, figsize=(8, 10.5), sharex=True,
                         gridspec_kw={"height_ratios": [len(ts) for _, ts in GROUPS], "hspace": 0.32})
for ax, (group, ts) in zip(axes, GROUPS):
    ts = sorted(ts, key=lambda t: -gen[t])
    for i, t in enumerate(ts):
        ax.barh(i, max(real[t], 0), height=0.66, facecolor=ORANGE_PALE, edgecolor=ORANGE, lw=1.3, zorder=1)
        ax.barh(i, gen[t], height=0.66, color=BLUE, zorder=2)
        ax.barh(i, max(real[t], 0), height=0.66, facecolor="none", edgecolor=ORANGE, lw=1.3, zorder=2.5)
        ax.text(gen[t] - 0.015, i, f"{gen[t]:.2f}".lstrip("0"), va="center", ha="right", fontsize=12.5, color="white",
                fontweight="bold", zorder=3)
        if t in NOTES:
            ax.text(max(gen[t], real[t]) + 0.02, i, NOTES[t], va="center", fontsize=12, color=GRAY, style="italic")
    ax.set_yticks(range(len(ts)), [NAMES[t] for t in ts], fontsize=14)
    ax.set_ylim(len(ts) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_title(group, loc="left", fontsize=17, fontweight="bold", pad=6)
axes[-1].set_xlim(0, 1.08)
axes[-1].set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", ".25", ".5", ".75", "1"])
axes[-1].set_xlabel("Spearman ρ with MTEB's ranking on the task")
axes[0].text(1.08, -1.0, "with the original queries", color=ORANGE, fontsize=14, fontweight="bold", ha="right",
             va="bottom")
axes[0].text(0.66, -1.0, "MTEB Gym", color=BLUE, fontsize=14, fontweight="bold", ha="right", va="bottom")
save(fig, "per_task")
