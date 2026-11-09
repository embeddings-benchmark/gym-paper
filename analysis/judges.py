"""Other judges and query writers, each swapped in for the default one at a time.

A judge is compared with JEV-27B on the same queries. A query writer is compared with DeepSeek-V4-Flash, with JEV-27B
judging both. The difference is the mean over the 28 tasks of the change in agreement, with a 95% bootstrap interval.

Two chat judges sometimes ran out of reasoning budget and fell back to low reasoning effort for a few grades. Those
grades are listed in data/low_effort_*.tsv and counted in the last column."""

from common import DATA, GENERATOR, JUDGE, TASKS, agreement, interval, mean_ci, overall_agreement, ratings, records, write

R = records()
LOW_EFFORT = {"Qwen3.8-27B": "low_effort_qwen.tsv", "DeepSeek-V4-Flash": "low_effort_deepseek.tsv"}


def scores(key):
    return {t: agreement(ratings(r), t) for t, r in R[key].items()}


rows = []
for judge, judging, queries in sorted(R):
    if judging != "pointwise" or (judge == JUDGE and queries in ("original", GENERATOR)):
        continue
    if judge == JUDGE:
        kind, name, default = "Query writer", queries, (JUDGE, "pointwise", GENERATOR)
    else:
        kind, name, default = "Judge", judge, (JUDGE, "pointwise", queries)
    new, old = scores((judge, judging, queries)), scores(default)
    assert sorted(new) == sorted(old) == sorted(TASKS)
    diff, low, high = mean_ci([new[t] - old[t] for t in TASKS])
    low_effort = sum(1 for _ in open(DATA / LOW_EFFORT[judge])) if kind == "Judge" and judge in LOW_EFFORT else 0
    rows.append({"": kind, "Model": name, "Queries": "original" if queries == "original" else "generated",
                 "Overall": overall_agreement({t: ratings(r) for t, r in R[(judge, judging, queries)].items()}),
                 "Per task": sum(new.values()) / len(new), "Per-task change": diff, "95% CI": interval(low, high),
                 "Low-effort grades": low_effort})
rows.sort(key=lambda r: (r[""], r["Queries"] != "generated", r["Model"]))
write("judges", rows, "Other judges and query writers against the defaults (JEV-27B, DeepSeek-V4-Flash).", wide=True)
