"""Appendix figure: how often each model beats each other model when the judge compares their results side by side.

JEV-27B comparing two models' top 10 for the same generated query (DeepSeek-V4-Flash), pooled over the 28 tasks; each
cell is the row model's mean chance of winning against the column model (score_a in the published verdicts, judged in
both orders). Models are ordered by the gym's overall rank."""

import glob
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from huggingface_hub import snapshot_download
from matplotlib.colors import LinearSegmentedColormap

from common import GENERATOR, GYM_RUNS, JUDGE, TASKS, ratings, records
from plots import BLUE, INK, ORANGE, save

R = records()
recs = R[(JUDGE, "pairwise", GENERATOR)]
query_sets = {(t, r["config"]["query_set"]) for t, r in recs.items()}
gym = {m: np.mean([ratings(R[(JUDGE, "pointwise", GENERATOR)][t])[m] for t in TASKS])
       for m in ratings(recs[TASKS[0]])}
models = sorted(gym, key=lambda m: -gym[m])
ix = {m: i for i, m in enumerate(models)}

folder = snapshot_download(GYM_RUNS, repo_type="dataset", allow_patterns=["verdicts/*/*.jsonl"])
total, count = np.zeros((len(models),) * 2), np.zeros((len(models),) * 2)
for path in glob.glob(f"{folder}/verdicts/*/*.jsonl"):
    for line in open(path):
        v = json.loads(line)
        if (v["task"], v["query_set"]) not in query_sets or v["judge"] != "autotrust/JEV-27B" or not all(v["parsed_ok"]):
            continue
        a, b = ix[v["model_a"]], ix[v["model_b"]]
        total[a, b] += v["score_a"]
        total[b, a] += 1 - v["score_a"]
        count[a, b] += 1
        count[b, a] += 1
win = np.where(count > 0, total / np.maximum(count, 1), np.nan)
np.fill_diagonal(win, np.nan)
print("verdicts used:", int(count.sum() / 2))

cmap = LinearSegmentedColormap.from_list("ob", [ORANGE, "#ffffff", BLUE])
names = [m.split("/")[-1].replace("baseline-bm25s", "BM25") for m in models]
fig, ax = plt.subplots(figsize=(13, 11.5))
im = ax.imshow(win, cmap=cmap, vmin=0, vmax=1)
ax.set_xticks(range(len(models)), names, rotation=90, fontsize=11)
ax.set_yticks(range(len(models)), names, fontsize=11)
ax.tick_params(length=0)
ax.xaxis.tick_top()
for side in ax.spines.values():
    side.set_visible(False)
bar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, ticks=[0, 0.5, 1])
bar.ax.set_yticklabels(["row loses", "even", "row wins"], fontsize=13)
bar.outline.set_visible(False)
ax.set_xlabel("Ordered by the gym's overall rank, best first", fontsize=15, color=INK, labelpad=10)
save(fig, "win_rates")
