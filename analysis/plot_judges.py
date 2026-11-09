"""Figure: other judges and query writers. Left, each judge's agreement with the labels on the tasks' own queries and on
generated queries. Right, each query writer's agreement with JEV-27B judging. Means over the 28 tasks; the defaults are
the stars."""

import matplotlib.pyplot as plt
import numpy as np

from common import GENERATOR, JUDGE, TASKS, agreement, ratings, records
from plots import BLUE, GRAY, INK, RED, save

R = records()


def mean_agreement(key):
    return float(np.mean([agreement(ratings(R[key][t]), t) for t in TASKS]))


JUDGES = ["JEV-27B", "AutoTrust-JEV-27B-v1.1", "JEV-35B", "JEV-9B", "GEV-26B-Decide", "decider-4b", "Qwen3.8-27B",
          "DeepSeek-V4-Flash"]
WRITERS = ["DeepSeek-V4-Flash", "MiniMax-M2.7", "Qwen3.8-27B", "Qwen3.5-4B"]
# label offsets (points) set by hand so names don't overlap
OFFSET = {"JEV-27B": (6, -20, "left"), "AutoTrust-JEV-27B-v1.1": (-10, 6, "right"), "JEV-35B": (10, -4, "left"),
          "JEV-9B": (-10, 0, "right"), "GEV-26B-Decide": (0, 18, "center"), "decider-4b": (-10, 6, "right"),
          "Qwen3.8-27B": (0, 18, "center"), "DeepSeek-V4-Flash": (10, -4, "left")}

fig, (ax, bx) = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True, gridspec_kw={"width_ratios": [2.1, 1.4], "wspace": 0.08})
for j in JUDGES:
    x, y = mean_agreement((j, "pointwise", "original")), mean_agreement((j, "pointwise", GENERATOR))
    default = j == JUDGE
    ax.scatter([x], [y], s=520 if default else 200, marker="*" if default else "o", color=INK if default else BLUE,
               edgecolor="white", lw=1, zorder=3)
    dx, dy, ha = OFFSET[j]
    ax.annotate(j + (" (default)" if default else ""), (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha,
                va="center", fontsize=15, color=INK, fontweight="bold" if default else "normal")
ax.set_xlabel("Agreement, tasks' own queries")
ax.set_ylabel("Agreement, generated queries")
ax.set_title("Judges", pad=10)
ax.set_xlim(0.68, 0.81)
ax.set_xticks([0.70, 0.74, 0.78], [".70", ".74", ".78"])

for i, w in enumerate(WRITERS):
    default = w == GENERATOR
    y = mean_agreement((JUDGE, "pointwise", w))
    bx.scatter([i], [y], s=520 if default else 200, marker="*" if default else "v", color=INK if default else RED,
               edgecolor="white", lw=1, zorder=3)
bx.set_xticks(range(len(WRITERS)), ["DeepSeek\nV4-Flash", "MiniMax\nM2.7", "Qwen3.8\n27B", "Qwen3.5\n4B"],
              fontsize=13.5)
bx.set_xlim(-0.6, len(WRITERS) - 0.4)
bx.set_title("Query writers", pad=10)
bx.tick_params(axis="y", left=False)
bx.spines["left"].set_color(GRAY)
ax.set_ylim(0.632, 0.695)
ax.set_yticks([0.64, 0.66, 0.68], [".64", ".66", ".68"])
save(fig, "judges")
