"""Figure: judge size against agreement of the overall ranking with the labels' (generated queries, 28 tasks).

Sizes are parameter counts from the judges' model cards. DeepSeek-V4-Flash is left out: its size is not public."""

import matplotlib.pyplot as plt

from common import GENERATOR, overall_agreement, ratings, records
from plots import CORAL, GRAY, INK, NAVY, save

R = records()
SIZE = {"decider-4b": 4, "JEV-9B": 9, "GEV-26B-Decide": 26, "JEV-27B": 27, "AutoTrust-JEV-27B-v1.1": 27,
        "Qwen3.8-27B": 27, "JEV-35B": 35}
OFFSET = {"decider-4b": (0, 16, "center"), "JEV-9B": (0, -22, "center"), "GEV-26B-Decide": (-12, -6, "right"),
          "JEV-27B": (12, -14, "left"), "AutoTrust-JEV-27B-v1.1": (-12, 12, "right"), "Qwen3.8-27B": (0, 16, "center"),
          "JEV-35B": (12, 0, "left")}

fig, ax = plt.subplots(figsize=(8, 5))
for judge, size in SIZE.items():
    y = overall_agreement({t: ratings(r) for t, r in R[(judge, "pointwise", GENERATOR)].items()})
    small = size < 10
    ax.scatter([size], [y], s=260, color=CORAL if small else NAVY, edgecolor="white", lw=1.2, zorder=3)
    dx, dy, ha = OFFSET[judge]
    ax.annotate(judge + (" (default)" if judge == "JEV-27B" else ""), (size, y), xytext=(dx, dy),
                textcoords="offset points", ha=ha, va="center", fontsize=14, color=INK,
                fontweight="bold" if judge == "JEV-27B" else "normal")
ax.set_xscale("log")
ax.set_xticks([4, 9, 27, 35], ["4B", "9B", "27B", "35B"])
ax.minorticks_off()
ax.set_xlim(2.8, 60)
ax.set_ylim(0.90, 1.0)
ax.set_yticks([0.9, 0.95, 1.0], [".90", ".95", "1"])
ax.set_xlabel("Judge size (parameters)")
ax.set_ylabel("Agreement, overall ranking")
ax.axhline(0.95, color=GRAY, lw=1, ls=(0, (4, 3)), zorder=0)
save(fig, "judge_size")
