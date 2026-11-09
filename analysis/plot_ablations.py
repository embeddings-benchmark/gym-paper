"""Figure: the overall ranking with other judges and query writers, swapped in one at a time (generated queries).

Bars: agreement of the gym's overall ranking (mean over 28 tasks) with the labels'. The default is dark blue and bold. Query writers are scored with JEV-27B judging; judges on DeepSeek-V4-Flash's queries."""

import matplotlib.pyplot as plt

from common import GENERATOR, JUDGE, overall_agreement, ratings, records
from plots import BLUE, BLUE_TINT, INK, save

R = records()


def overall(key):
    return overall_agreement({t: ratings(r) for t, r in R[key].items()})


JUDGES = ["JEV-27B", "AutoTrust-JEV-27B-v1.1", "JEV-35B", "Qwen3.8-27B", "GEV-26B-Decide", "JEV-9B", "decider-4b",
          "DeepSeek-V4-Flash"]
WRITERS = ["DeepSeek-V4-Flash", "MiniMax-M2.7", "Qwen3.8-27B", "Qwen3.5-4B"]
panels = [("Judges", [(j, overall((j, "pointwise", GENERATOR)), j == JUDGE) for j in JUDGES]),
          ("Query writers", [(w, overall((JUDGE, "pointwise", w)), w == GENERATOR) for w in WRITERS])]

fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                         gridspec_kw={"height_ratios": [len(JUDGES), len(WRITERS)], "hspace": 0.25})
for ax, (title, bars) in zip(axes, panels):
    for i, (name, v, default) in enumerate(bars):
        color = BLUE if default else BLUE_TINT
        ax.barh(i, v, height=0.68, color=color, zorder=2)
        ax.text(v + 0.012, i, f"{v:.2f}".lstrip("0"), va="center", fontsize=15, color=INK,
                fontweight="bold" if default else "normal")
    ax.set_yticks(range(len(bars)), [n + (" (default)" if d else "") for n, _, d in bars], fontsize=15)
    for label, (_, _, default) in zip(ax.get_yticklabels(), bars):
        if default:
            label.set_fontweight("bold")
    ax.set_ylim(len(bars) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_title(title, fontsize=17, fontweight="bold", loc="left", pad=6)
axes[1].set_xlim(0, 1.1)
axes[1].set_xticks([0, 0.5, 1], ["0", ".5", "1"])
axes[1].set_xlabel("Agreement of the overall ranking with MTEB's labels")
save(fig, "ablations")
