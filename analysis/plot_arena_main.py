"""Figure: how often different ways of picking a side agree with people's votes on MTEB Arena's retrieval battles.

The 2,169 votes for A or B where the two shown documents differ. A method picks the side it scores higher; an exact tie
counts as half right. Simple methods and embedding models as judges are from arena_baselines.py; the LLM judges grade
each side 0-5 (arena_judge.py)."""

import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from common import DATA
from plots import BLUE, BLUE_PALE, BLUE_TINT, GRAY, INK, save

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
values = np.array([[agreement(fn) for _, _, fn in COLUMNS]])

cmap = LinearSegmentedColormap.from_list("blue", [BLUE_PALE, BLUE_TINT, BLUE])
fig, ax = plt.subplots(figsize=(14, 2.2))
ax.imshow(values, cmap=cmap, vmin=45, vmax=88, aspect="auto")
for c, v in enumerate(values[0]):
    ax.text(c, 0, f"{v:.0f}%", ha="center", va="center", fontsize=22, color="white" if v > 72 else INK,
            fontweight="bold" if COLUMNS[c][1] == "LLM judge" else "normal")
for c in (2.5, 4.5):  # gaps between the groups
    ax.axvline(c, color="white", lw=8)
ax.set_xticks(range(len(COLUMNS)), [name for name, _, _ in COLUMNS], fontsize=14)
ax.set_yticks([])
ax.tick_params(length=0)
for side in ax.spines.values():
    side.set_visible(False)
for label, (_, group, _) in zip(ax.get_xticklabels(), COLUMNS):
    if group == "LLM judge":
        label.set_fontweight("bold")
        label.set_color(BLUE)
for (start, end), group in (((0, 2), "Simple"), ((3, 4), "Embedding model as judge"), ((5, 7), "LLM judge")):
    ax.annotate("", xy=(start - 0.4, -0.58), xytext=(end + 0.4, -0.58), annotation_clip=False,
                arrowprops=dict(arrowstyle="-", color=GRAY, lw=1.5))
    ax.text((start + end) / 2, -0.66, group, ha="center", va="bottom", fontsize=15, color=GRAY)
save(fig, "arena_main")
