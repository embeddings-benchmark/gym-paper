"""Figure: other judges and query writers, swapped in one at a time (generated queries).

Agreement with MTEB's labels of the overall ranking (mean over 28 tasks; top row) and per task (bottom row). Judges
are scored on DeepSeek-V4-Flash's queries; query writers with JEV-27B judging. The defaults are in bold."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from common import GENERATOR, JUDGE, TASKS, agreement, overall_agreement, ratings, records
from plots import BLUE, BLUE_PALE, BLUE_TINT, GRAY, INK, save

R = records()


def both(key):
    per_task = {t: ratings(r) for t, r in R[key].items()}
    return overall_agreement(per_task), float(np.mean([agreement(per_task[t], t) for t in TASKS]))


COLUMNS = [("JEV-27B\n(default)", (JUDGE, "pointwise", GENERATOR)),
           ("AutoTrust\nJEV-27B", ("AutoTrust-JEV-27B-v1.1", "pointwise", GENERATOR)),
           ("JEV-35B", ("JEV-35B", "pointwise", GENERATOR)),
           ("Qwen3.8\n27B", ("Qwen3.8-27B", "pointwise", GENERATOR)),
           ("GEV-26B\nDecide", ("GEV-26B-Decide", "pointwise", GENERATOR)),
           ("JEV-9B", ("JEV-9B", "pointwise", GENERATOR)),
           ("decider\n4B", ("decider-4b", "pointwise", GENERATOR)),
           ("DeepSeek\nV4-Flash", ("DeepSeek-V4-Flash", "pointwise", GENERATOR)),
           ("DeepSeek\nV4-Flash\n(default)", (JUDGE, "pointwise", GENERATOR)),
           ("MiniMax\nM2.7", (JUDGE, "pointwise", "MiniMax-M2.7")),
           ("Qwen3.8\n27B", (JUDGE, "pointwise", "Qwen3.8-27B")),
           ("Qwen3.5\n4B", (JUDGE, "pointwise", "Qwen3.5-4B"))]
GROUPS = [((0, 7), "Judges"), ((8, 11), "Query writers")]
values = np.array([both(key) for _, key in COLUMNS]).T  # rows: overall, per task

cmap = LinearSegmentedColormap.from_list("blue", [BLUE_PALE, BLUE_TINT, BLUE])
fig, ax = plt.subplots(figsize=(16, 3.4))
ax.imshow(values, cmap=cmap, vmin=0.4, vmax=1.0, aspect="auto")
for (r, c), v in np.ndenumerate(values):
    default = "(default)" in COLUMNS[c][0]
    ax.text(c, r, f"{v:.2f}".lstrip("0"), ha="center", va="center", fontsize=19,
            color="white" if v > 0.75 else INK, fontweight="bold" if default else "normal")
ax.axvline(7.5, color="white", lw=10)
ax.axhline(0.5, color="white", lw=4)
ax.set_xticks(range(len(COLUMNS)), [name for name, _ in COLUMNS], fontsize=13.5)
for label, (name, _) in zip(ax.get_xticklabels(), COLUMNS):
    if "(default)" in name:
        label.set_fontweight("bold")
ax.set_yticks([0, 1], ["Overall\nranking", "Per task"], fontsize=15)
ax.tick_params(length=0)
for side in ax.spines.values():
    side.set_visible(False)
for (start, end), group in GROUPS:
    ax.annotate("", xy=(start - 0.4, -0.62), xytext=(end + 0.4, -0.62), annotation_clip=False,
                arrowprops=dict(arrowstyle="-", color=GRAY, lw=1.5))
    ax.text((start + end) / 2, -0.7, group, ha="center", va="bottom", fontsize=16, color=GRAY)
save(fig, "ablations")
