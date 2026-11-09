"""Figure: a 4B judge and a 4B query writer give the same overall ranking as the large defaults.

Each point is one of the 28 models: its overall rank (mean score over the 28 tasks, generated queries) with the default
against its rank with the 4B model swapped in. Left: judges (JEV-27B vs decider-4b, on DeepSeek-V4-Flash's queries).
Right: query writers (DeepSeek-V4-Flash vs Qwen3.5-4B, JEV-27B judging). rho: Spearman correlation between the two
rankings; below it, each ranking's agreement with MTEB's labels."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata, spearmanr

from common import GENERATOR, JUDGE, TASKS, overall_agreement, ratings, records
from plots import BLUE, GRAY, INK, save

R = records()
PANELS = [("Judge", (JUDGE, "pointwise", GENERATOR), "JEV-27B", ("decider-4b", "pointwise", GENERATOR), "decider-4b"),
          ("Query writer", (JUDGE, "pointwise", GENERATOR), "DeepSeek-V4-Flash", (JUDGE, "pointwise", "Qwen3.5-4B"),
           "Qwen3.5-4B")]


def overall(key):
    per_task = {t: ratings(R[key][t]) for t in TASKS}
    models = sorted(per_task[TASKS[0]])
    return models, [np.mean([per_task[t][m] for t in TASKS]) for m in models], overall_agreement(per_task)


fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), gridspec_kw={"wspace": 0.3})
for ax, (role, big_key, big, small_key, small) in zip(axes, PANELS):
    models, big_scores, big_labels = overall(big_key)
    models_s, small_scores, small_labels = overall(small_key)
    assert models == models_s
    x, y = rankdata([-s for s in big_scores]), rankdata([-s for s in small_scores])
    ax.plot([0, 29], [0, 29], color=GRAY, lw=1.5, ls=(0, (4, 3)), zorder=1)
    ax.scatter(x, y, s=150, color=BLUE, edgecolor="white", lw=1.2, zorder=3)
    ax.text(0.04, 0.96, f"ρ = {spearmanr(big_scores, small_scores)[0]:.2f}", transform=ax.transAxes, va="top",
            fontsize=28, fontweight="bold", color=BLUE)
    ax.text(0.96, 0.04, f"agreement with MTEB's labels\n{big}: {big_labels:.2f}\n{small}: {small_labels:.2f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=13, color=GRAY, linespacing=1.4)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_ticks([1, 7, 14, 21, 28])
    ax.set_xlim(29.2, -0.2)
    ax.set_ylim(29.2, -0.2)
    ax.set_aspect("equal")
    ax.set_xlabel(f"Rank with {big} (default)")
    ax.set_ylabel(f"Rank with {small}")
    ax.set_title(f"{role}: 4B instead of the default", fontsize=17, fontweight="bold", pad=10)
save(fig, "small_models")
