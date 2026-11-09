"""MTEB Arena: how often the judge picks the same side as the person who voted.

The Arena's retrieval battles (mteb/arena-results) show a person two models' results for their own query, and the
person votes A, B, tie or both bad. The judges saw the same top document from each side as the person did. Grades are in
data/arena/<judge>.jsonl, made by arena_judge.py.

Counted: votes for A or B where the two documents differ. A pointwise judge picks the side with the higher grade; a
pairwise judge picks the side it prefers. 95% interval from 10,000 bootstrap resamples of the battles."""

import collections
import json

from common import DATA, interval, mean_ci, write

JUDGES = ("JEV-27B", "AutoTrust-JEV-27B-v1.1", "JEV-35B")
PICKS_A = {"pointwise": lambda b: b["ga"] - b["gb"], "pairwise": lambda b: b["pa"] - 0.5}

rows = []
for judge in JUDGES:
    battles = {b["id"]: b for b in map(json.loads, open(DATA / "arena" / f"{judge}.jsonl"))}.values()
    print(judge, dict(collections.Counter(b["vote"] for b in battles)))
    decided = [b for b in battles if b["vote"] in ("A", "B") and not b["same"]]
    for judging, picks_a in PICKS_A.items():
        hits = [(picks_a(b) > 0) == (b["vote"] == "A") for b in decided if picks_a(b) != 0]
        mean, low, high = mean_ci(hits)
        rows.append({"Judge": judge, "Judging": judging, "Votes": len(hits), "Agreement": mean,
                     "95% CI": interval(low, high)})
write("arena", rows, "How often the judge picks the side the person voted for, on MTEB Arena retrieval battles.")
