"""Figure: how often the judge picks the side a person voted for, on MTEB Arena's retrieval battles.

Votes for A or B where the two shown documents differ (2,169 battles). The judge picks the side with the higher grade.
Chance is 50%. Bars: three judges; error bars: 95% bootstrap intervals over battles."""

import json

import matplotlib.pyplot as plt

from common import DATA, mean_ci
from plots import GRAY, INK, LIGHT, NAVY, save

JUDGES = ["JEV-27B", "JEV-35B", "AutoTrust-JEV-27B-v1.1"]

fig, ax = plt.subplots(figsize=(7, 4.6))
for i, judge in enumerate(JUDGES):
    battles = {b["id"]: b for b in map(json.loads, open(DATA / "arena" / f"{judge}.jsonl"))}.values()
    hits = [(b["ga"] > b["gb"]) == (b["vote"] == "A") for b in battles
            if b["vote"] in ("A", "B") and not b["same"] and b["ga"] != b["gb"]]
    mean, low, high = mean_ci(hits)
    ax.bar(i, 100 * mean, width=0.6, color=NAVY if judge == "JEV-27B" else "#8DA9CF", zorder=2)
    ax.errorbar(i, 100 * mean, yerr=[[100 * (mean - low)], [100 * (high - mean)]], color=INK, capsize=6, lw=1.5,
                zorder=3)
    ax.text(i, 100 * high + 2, f"{100 * mean:.0f}%", ha="center", fontsize=17, fontweight="bold", color=INK)
ax.axhline(50, color=GRAY, lw=1.5, ls=(0, (4, 3)))
ax.text(-0.42, 52, "chance", color=GRAY, fontsize=14, ha="left")
ax.set_xticks(range(len(JUDGES)), ["JEV-27B\n(default)", "JEV-35B", "AutoTrust-\nJEV-27B-v1.1"])
ax.set_ylim(0, 100)
ax.set_yticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"])
ax.set_ylabel("Agrees with the person's vote")
ax.tick_params(axis="x", length=0)
ax.spines["bottom"].set_color(LIGHT)
save(fig, "arena")
