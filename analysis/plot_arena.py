"""Figures: the judge agrees with people's votes on MTEB Arena's retrieval battles (JEV-27B).

arena (main text): on votes for A or B where the two shown documents differ, how often the judge's higher-graded side is the side the
person picked, by the gap between the judge's two grades (five equal-sized bins). arena_votes (appendix): the judge's
mean grade for the two sides by how the person voted. Grades are the judge's expected grade, 0-5."""

import json

import matplotlib.pyplot as plt
import numpy as np

from common import DATA
from plots import BLUE, BLUE_TINT, GRAY, LIGHT, save

battles = list({b["id"]: b for b in map(json.loads, open(DATA / "arena" / "JEV-27B.jsonl"))}.values())
decided = [b for b in battles if b["vote"] in ("A", "B") and not b["same"] and b["ga"] != b["gb"]]
gap = np.array([abs(b["ga"] - b["gb"]) for b in decided])
hit = np.array([(b["ga"] > b["gb"]) == (b["vote"] == "A") for b in decided])
edges = np.quantile(gap, np.linspace(0, 1, 6))
bins = np.clip(np.digitize(gap, edges[1:-1]), 0, 4)

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(5)
agree = [100 * hit[bins == i].mean() for i in x]
ax.plot(x, agree, color=BLUE, marker="o", ms=14, mec="white", mew=1.5, lw=4)
for i, v in zip(x, agree):
    ax.text(i - (0.12 if i == 0 else 0), v + 4, f"{v:.0f}%", ha="right" if i == 0 else "center", fontsize=16, fontweight="bold", color=BLUE)
ax.axhline(50, color=GRAY, lw=1.5, ls=(0, (4, 3)))
ax.text(4.3, 52, "chance", color=GRAY, fontsize=14, ha="right")
ax.set_xticks(x, [f"{edges[i]:.1f}–{edges[i + 1]:.1f}" for i in x], fontsize=14)
ax.set_xlabel("Gap between the judge's grades for the two sides")
ax.set_ylabel("Judge agrees with the person")
ax.set_ylim(40, 108)
ax.set_yticks([50, 75, 100], ["50", "75", "100%"])
ax.set_xlim(-0.4, 4.4)
ax.text(0.02, 0.97, f"{100 * hit.mean():.0f}% overall", transform=ax.transAxes, fontsize=18, fontweight="bold",
        color=BLUE, va="top")
save(fig, "arena")

fig, bx = plt.subplots(figsize=(6.5, 5))

votes = [("Both bad", [b for b in battles if b["vote"] == "both_bad" and not b["same"]]),
         ("A or B", decided),
         ("Tie", [b for b in battles if b["vote"] == "tie" and not b["same"]])]
for i, (name, bs) in enumerate(votes):
    hi = np.mean([max(b["ga"], b["gb"]) for b in bs])
    lo = np.mean([min(b["ga"], b["gb"]) for b in bs])
    bx.plot([i, i], [lo, hi], color=LIGHT, lw=10, solid_capstyle="round", zorder=1)
    bx.scatter([i], [hi], s=220, color=BLUE, edgecolor="white", lw=1.5, zorder=3)
    bx.scatter([i], [lo], s=220, color=BLUE_TINT, edgecolor="white", lw=1.5, zorder=3)
    bx.text(i + 0.14, (hi + lo) / 2, f"gap\n{hi - lo:.1f}", fontsize=13, color=GRAY, va="center")
bx.set_xticks(range(3), [n for n, _ in votes])
bx.set_xlim(-0.5, 2.7)
bx.set_ylim(0, 5)
bx.set_yticks([0, 1, 2, 3, 4, 5])
bx.set_xlabel("The person voted")
bx.set_ylabel("Judge's grade (0–5)")
bx.text(-0.45, 4.75, "higher-graded side", color=BLUE, fontsize=14, fontweight="bold")
bx.text(-0.45, 4.35, "lower-graded side", color=BLUE_TINT, fontsize=14, fontweight="bold")
save(fig, "arena_votes")
