"""Figure: the overall ranking with other judges and query writers, swapped in one at a time (generated queries).

Bars: agreement of the gym's overall ranking (mean over 28 tasks) with the labels'. The default is dark; 4B models are
orange. Query writers are scored with JEV-27B judging; judges on DeepSeek-V4-Flash's queries."""

import matplotlib.pyplot as plt

from common import GENERATOR, JUDGE, overall_agreement, ratings, records
from plots import BLUE, BLUE_TINT, INK, LIGHT, ORANGE, save

R = records()


def overall(key):
    return overall_agreement({t: ratings(r) for t, r in R[key].items()})


JUDGES = ["JEV-27B", "AutoTrust-JEV-27B-v1.1", "JEV-35B", "Qwen3.8-27B", "GEV-26B-Decide", "JEV-9B", "decider-4b",
          "DeepSeek-V4-Flash"]
WRITERS = ["DeepSeek-V4-Flash", "MiniMax-M2.7", "Qwen3.8-27B", "Qwen3.5-4B"]
SMALL = {"decider-4b", "Qwen3.5-4B"}
SHORT = {"AutoTrust-JEV-27B-v1.1": "AutoTrust\nJEV-27B-v1.1", "Qwen3.8-27B": "Qwen3.8\n27B", "GEV-26B-Decide": "GEV-26B\nDecide",
         "DeepSeek-V4-Flash": "DeepSeek\nV4-Flash", "MiniMax-M2.7": "MiniMax\nM2.7", "Qwen3.5-4B": "Qwen3.5\n4B"}
panels = [("Judges", [(j, overall((j, "pointwise", GENERATOR)), j == JUDGE) for j in JUDGES]),
          ("Query writers", [(w, overall((JUDGE, "pointwise", w)), w == GENERATOR) for w in WRITERS])]

fig, axes = plt.subplots(1, 2, figsize=(16, 4.8), sharey=True,
                         gridspec_kw={"width_ratios": [len(JUDGES), len(WRITERS)], "wspace": 0.05})
for ax, (title, bars) in zip(axes, panels):
    for i, (name, v, default) in enumerate(bars):
        color = BLUE if default else (ORANGE if name in SMALL else BLUE_TINT)
        ax.bar(i, v, width=0.7, color=color, zorder=2)
        ax.text(i, v + 0.015, f"{v:.2f}".lstrip("0"), ha="center", fontsize=15, color=INK,
                fontweight="bold" if default else "normal")
    labels = [SHORT.get(n, n) + ("\n(default)" if d else "") for n, _, d in bars]
    ax.set_xticks(range(len(bars)), labels, fontsize=12.5)
    ax.tick_params(axis="x", length=0)
    ax.set_title(title, fontsize=18, fontweight="bold", pad=10)
    ax.spines["bottom"].set_color(LIGHT)
axes[0].set_ylim(0, 1.08)
axes[0].set_yticks([0, 0.5, 1], ["0", ".5", "1"])
axes[0].set_ylabel("Agreement, overall ranking")
axes[1].tick_params(axis="y", left=False)
axes[1].spines["left"].set_visible(False)
axes[1].text(3.4, 1.04, "4B models", color=ORANGE, fontsize=14, fontweight="bold", ha="right")
save(fig, "ablations")
