"""9. MTEB Arena: how often a judge picks the side a person voted for, on the Arena's retrieval battles
(mteb/arena-results, retrieval_battle: 4,087 votes; one top document per side, as people saw it).

Grades are in data/arena/<judge>.jsonl, made by arena_judge.py against a served decision model: per battle, a
pointwise grade for each side (ga, gb, the expected grade 0-5) and a pairwise preference for side A judged in both
presentation orders (pa). Agreement counts only decisive votes (A or B) where the two sides differ; tie and
"both bad" votes are reported apart. 95% CI: 2,000 bootstrap resamples of the battles, seed 0.
"""

import collections
import json

import numpy as np

from common import DATA, write

JUDGES = ("JEV-27B", "AutoTrust-JEV-27B-v1.1", "JEV-35B")
SIDE = {"pointwise": lambda r: r["ga"] - r["gb"], "pairwise": lambda r: r["pa"] - 0.5}


def load(judge: str) -> list[dict]:
    rows = [json.loads(l) for l in open(DATA / "arena" / f"{judge}.jsonl")]
    return list({r["id"]: r for r in rows}.values())  # a battle judged twice keeps its last line


rows, corpora = [], []
for judge in JUDGES:
    R = load(judge)
    votes = collections.Counter(r["vote"] for r in R)
    D = [r for r in R if r["vote"] in ("A", "B") and not r["same"]]
    for mode, fn in SIDE.items():
        hit = np.array([(fn(r) > 0) == (r["vote"] == "A") for r in D if fn(r) != 0], float)
        boot = np.random.default_rng(0).choice(hit, (2000, len(hit))).mean(1)
        rows.append(dict(judge=judge, judging=mode, battles=len(R), decisive=len(D), scored=len(hit),
                         agreement=float(hit.mean()), ci_low=float(np.percentile(boot, 2.5)), ci_high=float(np.percentile(boot, 97.5)),
                         votes=" ".join(f"{k}={v}" for k, v in sorted(votes.items()))))
        for c in sorted({r["corpus"] for r in D}):
            k = [(fn(r) > 0) == (r["vote"] == "A") for r in D if r["corpus"] == c and fn(r) != 0]
            corpora.append(dict(judge=judge, judging=mode, corpus=c, scored=len(k), agreement=float(np.mean(k))))
write("9_arena", rows, __doc__.strip())
write("9_arena_by_corpus", corpora, "Agreement on decisive votes, by the Arena corpus the battle searched.")
for r in rows:
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
