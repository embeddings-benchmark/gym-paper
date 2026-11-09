"""Figure: the ranking people give in MTEB Arena against the gym's and against MTEB's labels, on the 8 models in both.

The human ranking: Bradley-Terry strengths fitted to all of the Arena's retrieval votes (15 models; a tie or "both bad"
counts as half a win for each side), as Chatbot Arena ranks models. The gym: mean score over the 28 tasks with
generated queries (JEV-27B). MTEB's labels: mean nDCG@10 over the same 28 tasks. rho: Spearman correlation."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata, spearmanr

from common import GENERATOR, JUDGE, TASKS, arena_battles, labels, ratings, records
from plots import BLUE, GRAY, INK, save

IN_BOTH = {"intfloat/e5-mistral-7b-instruct": "e5-mistral-7b", "Alibaba-NLP/gte-Qwen2-7B-instruct": "gte-Qwen2-7B",
           "intfloat/multilingual-e5-large-instruct": "multilingual-e5-large",
           "nomic-ai/nomic-embed-text-v1.5": "nomic-embed-v1.5", "BAAI/bge-large-en-v1.5": "bge-large-v1.5",
           "mixedbread-ai/mxbai-embed-large-v1": "mxbai-embed-large", "BM25": "BM25",
           "sentence-transformers/all-MiniLM-L6-v2": "all-MiniLM-L6-v2"}
OURS = {"BM25": "mteb/baseline-bm25s"}  # the Arena's name -> ours, where they differ


def human_strengths():
    battles = list({b["id"]: b for b in arena_battles()}.values())
    models = sorted({b["ma"] for b in battles} | {b["mb"] for b in battles})
    ix = {m: i for i, m in enumerate(models)}
    wins = np.zeros((len(models), len(models)))
    for b in battles:
        a, c = ix[b["ma"]], ix[b["mb"]]
        if b["vote"] == "A":
            wins[a, c] += 1
        elif b["vote"] == "B":
            wins[c, a] += 1
        else:
            wins[a, c] += 0.5
            wins[c, a] += 0.5
    games, p = wins + wins.T, np.ones(len(models))
    for _ in range(2000):  # Bradley-Terry by minorization-maximization
        p = wins.sum(1) / (games / (p[:, None] + p[None, :])).sum(1)
        p /= p.sum()
    return {m: np.log(p[ix[m]]) for m in models}


human = human_strengths()
R = records()
arena_models = list(IN_BOTH)
ours = [OURS.get(m, m) for m in arena_models]
gym = [np.mean([ratings(R[(JUDGE, "pointwise", GENERATOR)][t])[m] for t in TASKS]) for m in ours]
lab = [np.mean([labels(t)[m] for t in TASKS]) for m in ours]
by_human = rankdata([-human[m] for m in arena_models])

fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), gridspec_kw={"wspace": 0.3})
for ax, scores, ylabel, color in ((axes[0], gym, "Rank by MTEB Gym", BLUE), (axes[1], lab, "Rank by MTEB's labels", GRAY)):
    ranks = rankdata([-s for s in scores])
    ax.plot([0, 9], [0, 9], color=GRAY, lw=1.5, ls=(0, (4, 3)), zorder=1)
    ax.scatter(by_human, ranks, s=230, color=color, edgecolor="white", lw=1.5, zorder=3)
    for m, x, y in zip(arena_models, by_human, ranks):
        left = x >= 5.5  # points near the left edge get their name on the right
        ax.annotate(IN_BOTH[m], (x, y), xytext=(12 if left else -12, 0), textcoords="offset points",
                    ha="left" if left else "right", va="center", fontsize=12.5, color=INK)
    rho = spearmanr(scores, [human[m] for m in arena_models])[0]
    ax.text(0.97, 0.05, f"ρ = {rho:.2f}", transform=ax.transAxes, ha="right", fontsize=26, fontweight="bold",
            color=color)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_ticks(range(1, 9))
    ax.set_xlim(8.6, 0.4)
    ax.set_ylim(8.6, 0.4)
    ax.set_aspect("equal")
    ax.set_xlabel("Rank by human votes (MTEB Arena)")
    ax.set_ylabel(ylabel)
save(fig, "humans")
