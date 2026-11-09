"""Figure: a bigger judge or query writer does not give a better ranking.

Agreement of the gym's overall ranking (mean over the 28 tasks) with MTEB's labels, by model size.
Left, judges: decider-4b, JEV-9B, JEV-27B, JEV-35B, on generated queries (DeepSeek-V4-Flash) and on the tasks' own
queries. Right, query writers with JEV-27B judging: Qwen3.5-4B and Qwen3.8-27B, and DeepSeek-V4-Flash and MiniMax-M2.7,
whose sizes are not public (shown apart, not joined)."""

import matplotlib.pyplot as plt

from common import GENERATOR, JUDGE, TASKS, overall_agreement, ratings, records
from plots import BLUE, GEN, GRAY, INK, ORANGE, OWN, save

R = records()


def overall(key):
    return overall_agreement({t: ratings(R[key][t]) for t in TASKS})


JUDGES = [("4B", "decider-4b"), ("9B", "JEV-9B"), ("27B", "JEV-27B"), ("35B", "JEV-35B")]
WRITERS = [("4B", "Qwen3.5-4B"), ("27B", "Qwen3.8-27B")]
UNKNOWN = ["DeepSeek-V4-Flash", "MiniMax-M2.7"]

fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True, gridspec_kw={"wspace": 0.1})
x = range(len(JUDGES))
for queries, style in (("original", OWN), (GENERATOR, GEN)):
    ax.plot(x, [overall((j, "pointwise", queries)) for _, j in JUDGES], color=style["color"], marker=style["marker"],
            ms=13, mec="white", mew=1.2, lw=3.5)
ax.set_xticks(x, [f"{size}\n{name}" for size, name in JUDGES], fontsize=14)
ax.set_xlabel("Judge size")
ax.set_xlim(-0.4, len(JUDGES) - 0.6)

bx.plot([0, 1], [overall((JUDGE, "pointwise", w)) for _, w in WRITERS], color=BLUE, marker="o", ms=13, mec="white",
        mew=1.2, lw=3.5)
bx.scatter([2.4, 3.4], [overall((JUDGE, "pointwise", w)) for w in UNKNOWN], s=170, color=BLUE, edgecolor="white",
           lw=1.2, zorder=3)
bx.axvline(1.7, color=GRAY, lw=1, ls=(0, (2, 3)))
bx.text(2.9, 0.86, "size not public", color=GRAY, fontsize=13, ha="center")
bx.set_xticks([0, 1, 2.4, 3.4], ["4B\nQwen3.5", "27B\nQwen3.8", "DeepSeek\nV4-Flash", "MiniMax\nM2.7"], fontsize=13)
bx.set_xlabel("Query writer size")
bx.set_xlim(-0.4, 3.8)

ax.set_ylim(0.85, 1.0)
ax.set_yticks([0.85, 0.9, 0.95, 1.0], [".85", ".90", ".95", "1"])
ax.set_ylabel("Agreement, overall ranking")
ax.text(0.03, 0.17, "Generated queries", transform=ax.transAxes, color=BLUE, fontsize=16, fontweight="bold")
ax.text(0.03, 0.06, "Tasks' own queries", transform=ax.transAxes, color=ORANGE, fontsize=16, fontweight="bold")
for a, y in ((ax, overall(("decider-4b", "pointwise", GENERATOR))), (bx, overall((JUDGE, "pointwise", "Qwen3.5-4B")))):
    a.annotate("4B is enough", xy=(0.05, y - 0.005), xytext=(0.9, 0.905), fontsize=15, color=INK, ha="center",
               arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.5, connectionstyle="arc3,rad=-0.25"))
save(fig, "small_models")
