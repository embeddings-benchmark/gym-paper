"""Main table: how well the gym's ranking of the 28 models agrees with each task's labels.

JEV-27B judges. Queries are the task's own or written by DeepSeek-V4-Flash. Agreement is the Spearman correlation per
task, averaged over the 28 tasks, with a 95% bootstrap interval over tasks."""

from common import GENERATOR, JUDGE, TASKS, agreement, interval, mean_ci, overall_agreement, ratings, records, write

R = records()
SETTINGS = [(judging, queries) for judging in ("pointwise", "pairwise") for queries in ("original", GENERATOR)]
NAMES = {"original": "real", GENERATOR: "generated"}

per_task = {s: {t: agreement(ratings(R[(JUDGE, *s)][t]), t) for t in TASKS} for s in SETTINGS}

rows = []
for judging, queries in SETTINGS:
    mean, low, high = mean_ci(list(per_task[(judging, queries)].values()))
    overall = overall_agreement({t: ratings(R[(JUDGE, judging, queries)][t]) for t in TASKS})
    rows.append({"Judging": judging, "Queries": NAMES[queries], "Overall": overall, "Per task": mean,
                 "95% CI": interval(low, high)})
write("main_table", rows, bold={1}, rules={2}, caption= "Agreement with the labels' ranking: of the overall ranking over 28 tasks, and on each task averaged over the 28 (with a 95\\% bootstrap interval over tasks).")

rows = [{"Task": t} | {f"{j}, {NAMES[q]}": per_task[(j, q)][t] for j, q in SETTINGS} for t in TASKS]
write("main_table_per_task", rows, "Agreement with the labels on each task.", wide=True)
