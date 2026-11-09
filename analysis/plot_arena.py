"""Appendix figures on MTEB Arena's retrieval battles, with JEV-27B as the judge.

arena_gap: on the votes for A or B where the two shown documents differ, how often the judge's higher-graded side is
the side the person picked, by the gap between the judge's two grades (five bins of equal size).
arena_votes: the judge's mean grade for the higher- and lower-graded side, by how the person voted."""

import json

import matplotlib.pyplot as plt
import numpy as np

from common import DATA
from plots import BLUE, GRAY, INK, LIGHT, ORANGE, save

battles = list({b["id"]: b for b in map(json.loads, open(DATA / "arena" / "JEV-27B.jsonl"))}.values())
decided = [b for b in battles if b["vote"] in ("A", "B") and not b["same"] and b["ga"] != b["gb"]]
gap = np.array([abs(b["ga"] - b["gb"]) for b in decided])
hit = np.array([(b["ga"] > b["gb"]) == (b["vote"] == "A") for b in decided])
edges = np.quantile(gap, np.linspace(0, 1, 6))
bins = np.clip(np.digitize(gap, edges[1:-1]), 0, 4)

# arena_gap
fig, ax = plt.subplots(figsize=(8.5, 4.6))
x = np.arange(5)
agree = [100 * hit[bins == i].mean() for i in x]
ax.plot(x, agree, color=BLUE, marker="o", ms=14, mec="white", mew=1.5, lw=4)
for i, v in zip(x, agree):
    ax.text(i, v + 3.5, f"{v:.0f}%", ha="center", fontsize=15, fontweight="bold", color=BLUE)
ax.axhline(50, color=GRAY, lw=1.5, ls=(0, (4, 3)))
ax.text(4.35, 51.5, "chance", color=GRAY, fontsize=14, ha="right")
ax.annotate("close calls", xy=(0.08, 52), xytext=(0.7, 63), fontsize=15, color=INK, ha="left",
            arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.5, connectionstyle="arc3,rad=0.25"))
ax.set_xticks(x, [f"{edges[i]:.1f}–{edges[i + 1]:.1f}" for i in x], fontsize=14)
ax.set_xlabel("How far apart the judge graded the two sides (0–5)")
ax.set_ylabel("Agrees with the human vote")
ax.set_ylim(40, 108)
ax.set_yticks([50, 75, 100], ["50", "75", "100%"])
ax.set_xlim(-0.4, 4.4)
save(fig, "arena_gap")

# arena_votes
fig, ax = plt.subplots(figsize=(7, 4.6))
votes = [("Both bad", [b for b in battles if b["vote"] == "both_bad" and not b["same"]]),
         ("A or B", decided),
         ("Tie", [b for b in battles if b["vote"] == "tie" and not b["same"]])]
for i, (name, bs) in enumerate(votes):
    hi = np.mean([max(b["ga"], b["gb"]) for b in bs])
    lo = np.mean([min(b["ga"], b["gb"]) for b in bs])
    ax.plot([i, i], [lo, hi], color=LIGHT, lw=12, solid_capstyle="round", zorder=1)
    ax.scatter([i], [hi], s=260, color=BLUE, edgecolor="white", lw=1.5, zorder=3)
    ax.scatter([i], [lo], s=260, marker="s", color=ORANGE, edgecolor="white", lw=1.5, zorder=3)
    ax.text(i + 0.13, (hi + lo) / 2, f"gap {hi - lo:.1f}", fontsize=14, color=INK, va="center")
ax.set_xticks(range(3), [n for n, _ in votes], fontsize=15)
ax.set_xlim(-0.5, 2.8)
ax.set_ylim(0, 5)
ax.set_xlabel("The human voted")
ax.set_ylabel("Judge's grade (0–5)")
ax.text(0.02, 0.97, "higher-graded side", transform=ax.transAxes, color=BLUE, fontsize=15, fontweight="bold", va="top")
ax.text(0.02, 0.87, "lower-graded side", transform=ax.transAxes, color=ORANGE, fontsize=15, fontweight="bold",
        va="top")
save(fig, "arena_votes")
