"""Figures: the judge agrees with people's votes on MTEB Arena's retrieval battles (JEV-27B).

arena (main text): the votes for A or B where the two shown documents differ, by the gap between the judge's two grades
(bins of 0.5), split into those where the judge's higher-graded side is the side the person picked and those where not. arena_votes (appendix): the judge's
mean grade for the two sides by how the person voted. Grades are the judge's expected grade, 0-5."""

import json

import matplotlib.pyplot as plt
import numpy as np

from common import DATA
from plots import BLUE, BLUE_TINT, GRAY, INK, LIGHT, ORANGE, save

battles = list({b["id"]: b for b in map(json.loads, open(DATA / "arena" / "JEV-27B.jsonl"))}.values())
decided = [b for b in battles if b["vote"] in ("A", "B") and not b["same"] and b["ga"] != b["gb"]]
gap = np.array([abs(b["ga"] - b["gb"]) for b in decided])
hit = np.array([(b["ga"] > b["gb"]) == (b["vote"] == "A") for b in decided])
edges = np.quantile(gap, np.linspace(0, 1, 6))
bins = np.clip(np.digitize(gap, edges[1:-1]), 0, 4)

fig, ax = plt.subplots(figsize=(10, 5.2))
width = 0.5
edges = np.arange(0, 5 + width, width)
bins = np.clip((gap // width).astype(int), 0, len(edges) - 2)
for i in range(len(edges) - 1):
    n_agree, n_all = int(hit[bins == i].sum()), int((bins == i).sum())
    if not n_all:
        continue
    x = edges[i] + width / 2
    ax.bar(x, n_agree, width=width * 0.86, color=BLUE, zorder=2)
    ax.bar(x, n_all - n_agree, bottom=n_agree, width=width * 0.86, color=ORANGE, zorder=2)
    ax.text(x, n_all + 12, f"{100 * n_agree / n_all:.0f}%", ha="center", fontsize=15, fontweight="bold", color=INK)
ax.text(0.98, 0.95, f"{100 * hit.mean():.0f}%", transform=ax.transAxes, ha="right", va="top", fontsize=46,
        fontweight="bold", color=BLUE)
ax.text(0.98, 0.69, f"of {len(hit):,} human votes,\nthe judge picks the same side", transform=ax.transAxes,
        ha="right", va="top", fontsize=16, color=INK, linespacing=1.3)
ax.text(0.98, 0.50, "agrees", transform=ax.transAxes, ha="right", fontsize=16, fontweight="bold", color=BLUE)
ax.text(0.98, 0.42, "disagrees", transform=ax.transAxes, ha="right", fontsize=16, fontweight="bold", color=ORANGE)
ax.set_xlim(0, 5)
ax.set_xticks(range(6))
ax.set_xlabel("How far apart the judge graded the two sides (0–5 scale)")
ax.set_ylabel("Human votes")
ax.spines["bottom"].set_color(LIGHT)
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
