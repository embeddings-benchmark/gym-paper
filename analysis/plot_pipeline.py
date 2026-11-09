"""Figure: how MTEB Gym works, on one real query from SciFact.

The query writer (DeepSeek-V4-Flash) writes a query from three documents; each model retrieves its top 10; the judge
(JEV-27B) grades every retrieved document 0-5; each model's score is its nDCG@10 with those grades, averaged over the
queries. Shown: query q50, the top 3 of a strong and a weak model, and the task's final top of the ranking."""

import json

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from common import GENERATOR, JUDGE, judged, ratings, records
from mteb_gym import corpus
from plots import CORAL, GRAY, INK, LIGHT, NAVY, save

TASK, QID = "SciFact", "q50"
QUERY = "What non-genetic factors can trigger cancer development in the presence of oncogenic mutations?"
MODELS = [("Qwen/Qwen3-Embedding-8B", "Qwen3-Embedding-8B"), ("sentence-transformers/all-MiniLM-L6-v2", "all-MiniLM-L6-v2")]

R = records()
rec = R[(JUDGE, "pointwise", GENERATOR)][TASK]
_, _, top, grades = judged(rec)
docs = corpus.load(TASK).docs
title = lambda d: (docs[d] if isinstance(docs[d], str) else json.dumps(docs[d]))[:34].rsplit(" ", 1)[0] + " …"
board = sorted(ratings(rec).items(), key=lambda x: -x[1])
grade_color = LinearSegmentedColormap.from_list("g", ["#f2f5fa", NAVY])

fig = plt.figure(figsize=(16, 5.2))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 5.2)
ax.axis("off")


def box(x, y, w, h, face="white", edge=LIGHT, lw=1.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", facecolor=face,
                                edgecolor=edge, lw=lw))


def arrow(x0, x1, y=2.5):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=22, color=GRAY, lw=2))


def step(x, n, text):
    ax.text(x, 4.85, f"{n}  {text}", fontsize=16, fontweight="bold", color=INK, va="center")


# 1. documents
step(0.2, 1, "Your documents")
for i in range(3):
    box(0.3 + 0.18 * i, 1.5 - 0.18 * i, 2.0, 2.4, face="#f7f7f7")
for j in range(6):
    ax.plot([0.75, 2.6 - (0.5 if j % 3 == 2 else 0)], [3.3 - 0.3 * j] * 2, color=LIGHT, lw=3, solid_capstyle="round")
ax.text(1.65, 0.75, "no queries, no labels", fontsize=13, color=GRAY, ha="center")
arrow(2.9, 3.5)

# 2. generated query
step(3.6, 2, "Write a query")
box(3.7, 1.7, 3.0, 1.9, face="#fbefe9", edge=CORAL)
ax.text(3.9, 2.65, "What non-genetic factors\ncan trigger cancer\ndevelopment in the presence\nof oncogenic mutations?",
        fontsize=12.5, color=INK, va="center", linespacing=1.35)
ax.text(5.2, 1.25, "DeepSeek-V4-Flash", fontsize=12, color=GRAY, ha="center")
arrow(6.9, 7.5)

# 3. retrieval and grades
step(7.6, 3, "Each model retrieves, the judge grades")
for k, (model, short) in enumerate(MODELS):
    y0 = 3.75 - 1.75 * k
    ax.text(7.7, y0 + 0.42, short, fontsize=13.5, fontweight="bold", color=INK)
    for i, d in enumerate(top[model][QID][:3]):
        y = y0 - 0.42 * i
        g = grades[QID][d]
        box(7.7, y - 0.17, 0.6, 0.34, face=grade_color(g / 5), edge="none")
        ax.text(8.0, y, f"{g:.1f}", fontsize=12, ha="center", va="center", color="white" if g > 2.5 else INK,
                fontweight="bold")
        ax.text(8.45, y, title(d), fontsize=12, va="center", color=INK)
ax.text(10.1, 0.3, "Judge: JEV-27B, grade 0–5 per document", fontsize=12, color=GRAY, ha="center")
arrow(12.3, 12.9)

# 4. ranking: the top 3 on this task, and where the two example models land
step(13.0, 4, "Rank the models")
place = {m: i + 1 for i, (m, _) in enumerate(board)}
shown = sorted({1, 2, 3} | {place[m] for m, _ in MODELS})
y = 3.75
for n, rank in enumerate(shown):
    if n and rank > shown[n - 1] + 1:
        ax.text(13.45, y + 0.12, "…", fontsize=16, color=GRAY, va="center", rotation=90)
        y -= 0.5
    m = board[rank - 1][0]
    example = m in dict(MODELS)
    ax.text(13.1, y, f"{rank}", fontsize=14, color=GRAY, va="center")
    ax.text(13.6, y, m.split("/")[-1], fontsize=13.5, va="center", color=NAVY if example else INK,
            fontweight="bold" if example else "normal")
    y -= 0.5
ax.text(14.4, 0.6, f"{TASK}: nDCG@10 with the judge's grades,\naveraged over its 100 queries", fontsize=12,
        color=GRAY, ha="center")
save(fig, "pipeline")
