"""Figure: how often different ways of picking a side agree with people's votes on MTEB Arena's retrieval battles.

The 2,169 votes for A or B where the two shown documents differ. A method picks the side it scores higher; an exact tie
counts as half right. Simple methods and embedding models as judges are from arena_baselines.py; the LLM judges grade
each side 0-5 (arena_judge.py)."""

import json

import matplotlib.pyplot as plt
import numpy as np

from common import DATA
from plots import BLUE, GRAY, LIGHT, ORANGE, save

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
STYLE = {"Simple": dict(color=GRAY, marker="o"), "Embedding model\nas judge": dict(color=ORANGE, marker="s"),
         "LLM judge": dict(color=BLUE, marker="o")}

fig, ax = plt.subplots(figsize=(12, 4.8))
for k, ((name, group, _), v) in enumerate(zip(COLUMNS, values)):
    ax.scatter([k], [v], s=260, edgecolor="white", lw=1.5, zorder=3, **STYLE[group])
    ax.text(k, v + 3, f"{v:.0f}%", ha="center", fontsize=15, color=STYLE[group]["color"],
            fontweight="bold" if group == "LLM judge" else "normal")
ax.plot(range(len(values)), values, color=LIGHT, lw=3, zorder=1)
ax.axhline(50, color=GRAY, lw=1.5, ls=(0, (4, 3)), zorder=0)
ax.set_xticks(range(len(COLUMNS)), ["Random", "Longer\ndocument", "Word\noverlap", "MiniLM\nL6-v2", "Qwen3-Emb.\n0.6B",
                                    "JEV-27B", "JEV-35B", "AutoTrust\nJEV-27B"], fontsize=13.5)
ax.set_ylim(40, 95)
ax.set_yticks([50, 60, 70, 80, 90], ["50", "60", "70", "80", "90%"])
ax.set_ylabel("Agrees with the human vote")
ax.set_xlim(-0.5, len(COLUMNS) - 0.5)
for (start, end), group, color in (((0, 2), "Simple", GRAY), ((3, 4), "Embedding model as judge", ORANGE),
                                   ((5, 7), "LLM judge", BLUE)):
    ax.text((start + end) / 2, 41.5, group, ha="center", fontsize=14, color=color, fontweight="bold")
save(fig, "arena_methods")
