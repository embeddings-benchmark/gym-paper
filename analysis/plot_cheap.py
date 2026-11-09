"""Appendix figure: the same ranking with 10x fewer queries.

Each point is one of the 28 models: its overall gym score (mean nDCG@10 with the judge's grades, over the 28 tasks;
generated queries, JEV-27B) with all 100 queries per task against with 10 of them (drawn with seed 0). The band is a
95% bootstrap interval of the fitted line. Inset: judge calls (graded documents) in each setting."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from common import GENERATOR, JUDGE, TASKS, judged, ndcg, records
from plots import BLUE, GRAY, INK, LIGHT, ORANGE, save

R = records()
full, cheap, calls_full, calls_cheap = {}, {}, 0, 0
for t in TASKS:
    models, qids, top, grades, _ = judged(R[(JUDGE, "pointwise", GENERATOR)][t])
    pick = set(np.random.default_rng(0).choice(qids, min(10, len(qids)), replace=False))
    for m in models:
        scores = {q: ndcg(top[m][q], grades[q], grades[q].values(), 10) for q in qids}
        full.setdefault(m, []).append(np.mean(list(scores.values())))
        cheap.setdefault(m, []).append(np.mean([scores[q] for q in pick]))
    calls_full += sum(len(grades[q]) for q in qids)
    calls_cheap += sum(len(grades[q]) for q in pick)
models = sorted(full)
x = np.array([100 * np.mean(full[m]) for m in models])
y = np.array([100 * np.mean(cheap[m]) for m in models])
rho = spearmanr(x, y)[0]

fig, ax = plt.subplots(figsize=(9, 6.2))
grid = np.linspace(x.min() - 2, x.max() + 2, 50)
rng = np.random.default_rng(0)
lines = []
for _ in range(2000):
    i = rng.integers(0, len(x), len(x))
    lines.append(np.polyval(np.polyfit(x[i], y[i], 1), grid))
lo, hi = np.percentile(lines, [2.5, 97.5], axis=0)
ax.fill_between(grid, lo, hi, color=LIGHT, alpha=0.8, lw=0)
ax.plot(grid, np.polyval(np.polyfit(x, y, 1), grid), color=GRAY, lw=2)
ax.scatter(x, y, s=150, color=BLUE, edgecolor="white", lw=1.2, zorder=3)
for m in ("Qwen/Qwen3-Embedding-8B", "sentence-transformers/all-MiniLM-L6-v2", "mteb/baseline-bm25s",
          "minishlab/potion-retrieval-32M"):
    i = models.index(m)
    ax.annotate(m.split("/")[-1].replace("baseline-bm25s", "BM25"), (x[i], y[i]), xytext=(10, -4),
                textcoords="offset points", fontsize=13, color=INK)
ax.text(0.03, 0.97, "Same ranking with\n10× fewer queries", transform=ax.transAxes, va="top", fontsize=22,
        fontweight="bold", color=INK, linespacing=1.15)
ax.text(0.03, 0.74, f"Spearman ρ = {rho:.2f}", transform=ax.transAxes, va="top", fontsize=17, color=BLUE,
        fontweight="bold")
ax.set_xlabel("Gym score, 100 queries per task")
ax.set_ylabel("Gym score, 10 queries per task")

inset = ax.inset_axes([0.68, 0.1, 0.28, 0.32])
inset.bar([0, 1], [calls_full / 1e3, calls_cheap / 1e3], color=[GRAY, ORANGE], width=0.6)
inset.set_xticks([0, 1], ["100 q.", "10 q."], fontsize=12)
inset.set_title("Judge calls (thousands)", fontsize=12)
inset.tick_params(axis="y", labelsize=11)
for i, v in enumerate([calls_full, calls_cheap]):
    inset.text(i, v / 1e3, f"{v / 1e3:.0f}k", ha="center", va="bottom", fontsize=12)
inset.set_ylim(0, calls_full / 1e3 * 1.25)
save(fig, "cheap")
