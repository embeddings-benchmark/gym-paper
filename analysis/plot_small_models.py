"""Figure: a bigger judge does not give a better ranking, and every query writer does as well as the tasks' own queries.

Agreement of the gym's overall ranking (mean over the 28 tasks) with MTEB's labels.
Left, judges: decider-4b, JEV-9B, JEV-27B, JEV-35B, on generated queries (DeepSeek-V4-Flash) and on the tasks' own
queries. Right, the four query writers with JEV-27B judging, against the tasks' own queries (orange line)."""

import matplotlib.pyplot as plt

from common import GENERATOR, JUDGE, TASKS, overall_agreement, ratings, records
from plots import BLUE, GEN, GRAY, INK, ORANGE, OWN, save

R = records()


def overall(key):
    return overall_agreement({t: ratings(R[key][t]) for t in TASKS})


JUDGES = [("4B", "decider-4b"), ("9B", "JEV-9B"), ("27B", "JEV-27B"), ("35B", "JEV-35B")]
WRITERS = ["DeepSeek-V4-Flash", "MiniMax-M2.7", "Qwen3.8-27B", "Qwen3.5-4B"]

fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True, gridspec_kw={"wspace": 0.1})
x = range(len(JUDGES))
for queries, style in (("original", OWN), (GENERATOR, GEN)):
    ax.plot(x, [overall((j, "pointwise", queries)) for _, j in JUDGES], color=style["color"], marker=style["marker"],
            ms=13, mec="white", mew=1.2, lw=3.5)
ax.set_xticks(x, [f"{size}\n{name}" for size, name in JUDGES], fontsize=14)
ax.set_xlabel("Judge size")
ax.set_xlim(-0.4, len(JUDGES) - 0.6)

own = overall((JUDGE, "pointwise", "original"))
bx.axhline(own, color=ORANGE, lw=2.5, ls=(0, (5, 3)), zorder=1)
bx.text(3.45, own + 0.004, "tasks' own queries", color=ORANGE, fontsize=14, fontweight="bold", ha="right")
bx.scatter(range(len(WRITERS)), [overall((JUDGE, "pointwise", w)) for w in WRITERS], s=200, color=BLUE,
           edgecolor="white", lw=1.5, zorder=3)
bx.set_xticks(range(len(WRITERS)), ["DeepSeek\nV4-Flash", "MiniMax\nM2.7", "Qwen3.8\n27B", "Qwen3.5\n4B"], fontsize=14)
bx.set_xlabel("Query writer")
bx.set_xlim(-0.5, len(WRITERS) - 0.5)
bx.text(1.5, 0.905, "Every query writer is within .02\nof the tasks' own queries", fontsize=15, color=INK, ha="center")

ax.set_ylim(0.85, 1.0)
ax.set_yticks([0.85, 0.9, 0.95, 1.0], [".85", ".90", ".95", "1"])
ax.set_ylabel("Agreement, overall ranking")
ax.text(0.03, 0.17, "Generated queries", transform=ax.transAxes, color=BLUE, fontsize=16, fontweight="bold")
ax.text(0.03, 0.06, "Tasks' own queries", transform=ax.transAxes, color=ORANGE, fontsize=16, fontweight="bold")
for a, y in ((ax, overall(("decider-4b", "pointwise", GENERATOR))),):
    a.annotate("4B is enough", xy=(0.05, y - 0.005), xytext=(0.9, 0.905), fontsize=15, color=INK, ha="center",
               arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.5, connectionstyle="arc3,rad=-0.25"))
save(fig, "small_models")
