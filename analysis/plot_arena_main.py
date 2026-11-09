"""Figure: how often different ways of picking a side agree with people's votes on MTEB Arena's retrieval battles.

The 2,169 votes for A or B where the two shown documents differ. A method picks the side it scores higher; an exact tie
counts as half right. Simple methods and embedding models as judges are from arena_baselines.py; the LLM judges grade
each side 0-5 (arena_judge.py)."""

import json

import matplotlib.pyplot as plt
import numpy as np

from common import DATA
from plots import BLUE, BLUE_TINT, GRAY, INK, LIGHT, save

base = {r["id"]: r for r in map(json.loads, open(DATA / "arena" / "baselines.jsonl"))}
decided = [i for i, r in base.items() if r["vote"] in ("A", "B") and not r["same"]]


def agreement(score):
    right = []
    for i in decided:
        a, b = score(i)
        right.append(0.5 if a == b else float((a > b) == (base[i]["vote"] == "A")))
    return 100 * np.mean(right)


judges = {j: {r["id"]: r for r in map(json.loads, open(DATA / "arena" / f"{j}.jsonl"))}
          for j in ("JEV-27B", "JEV-35B", "AutoTrust-JEV-27B-v1.1")}
COLUMNS = [("Random", "Simple", lambda i: (0, 0)),
           ("Longer\ndocument", "Simple", lambda i: base[i]["length"]),
           ("Word\noverlap", "Simple", lambda i: base[i]["overlap"]),
           ("all-MiniLM\n-L6-v2", "Embedding model\nas judge", lambda i: base[i]["all-MiniLM-L6-v2"]),
           ("Qwen3-Emb.\n-0.6B", "Embedding model\nas judge", lambda i: base[i]["Qwen3-Embedding-0.6B"]),
           ("JEV-27B\n(default)", "LLM judge", lambda i: (judges["JEV-27B"][i]["ga"], judges["JEV-27B"][i]["gb"])),
           ("JEV-35B", "LLM judge", lambda i: (judges["JEV-35B"][i]["ga"], judges["JEV-35B"][i]["gb"])),
           ("AutoTrust\n-JEV-27B", "LLM judge",
            lambda i: (judges["AutoTrust-JEV-27B-v1.1"][i]["ga"], judges["AutoTrust-JEV-27B-v1.1"][i]["gb"]))]
values = [agreement(fn) for _, _, fn in COLUMNS]
SHADE = {"Simple": LIGHT, "Embedding model\nas judge": BLUE_TINT, "LLM judge": BLUE}

fig, ax = plt.subplots(figsize=(9, 5.4))
for k, ((name, group, _), v) in enumerate(zip(COLUMNS, values)):
    llm = group == "LLM judge"
    ax.barh(k, v, height=0.7, color=SHADE[group], zorder=2)
    ax.text(v + 1, k, f"{v:.0f}%", va="center", fontsize=16, fontweight="bold" if llm else "normal", color=INK)
ax.axvline(50, color=GRAY, lw=1.5, ls=(0, (4, 3)), zorder=1)
ax.set_yticks(range(len(COLUMNS)), [name.replace("\n", " ").replace(" -", "-").replace("Emb.", "Embedding")
                                     + ("" if group != "Embedding model\nas judge" else " as judge")
                                     for name, group, _ in COLUMNS], fontsize=15)
for label, (_, group, _) in zip(ax.get_yticklabels(), COLUMNS):
    if group == "LLM judge":
        label.set_fontweight("bold")
ax.set_ylim(len(COLUMNS) - 0.5, -0.5)
ax.set_xlim(0, 100)
ax.set_xticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"])
ax.set_xlabel(f"Agrees with the human vote ({len(decided):,} votes, MTEB Arena)")
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
save(fig, "arena_main")
