"""Other judges and query writers, each swapped in for the default one at a time.

A judge is compared with JEV-27B on the same queries. A query writer is compared with DeepSeek-V4-Flash, with JEV-27B
judging both. The difference is the mean over the 28 tasks of the change in agreement, with a 95% bootstrap interval.

Two chat judges sometimes ran out of reasoning budget and fell back to low reasoning effort for a few grades. Those
grades are listed in data/low_effort_*.tsv and counted in the last column."""

from common import DATA, GENERATOR, JUDGE, TABLES, TASKS, agreement, interval, mean_ci, num, overall_agreement, ratings, records, write

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

# compact table for the main text: generated queries, the default first in each group
ABLATIONS = [("Judges, decision models", [(j, "pointwise", GENERATOR) for j in
                                          ["JEV-27B", "AutoTrust-JEV-27B-v1.1", "JEV-35B", "GEV-26B-Decide", "JEV-9B",
                                           "decider-4b"]]),
             ("Judges, general LLMs", [(j, "pointwise", GENERATOR) for j in ["Qwen3.8-27B", "DeepSeek-V4-Flash"]]),
             ("Query writers", [(JUDGE, "pointwise", w) for w in [GENERATOR, "MiniMax-M2.7", "Qwen3.8-27B", "Qwen3.5-4B"]])]
lines = [r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{5pt}", r"\begin{tabular}{lrr}", r"\toprule",
         r"\textbf{Model} & \textbf{Overall} & \textbf{Per task} \\"]
for kind, keys in ABLATIONS:
    lines += [r"\midrule", rf"\multicolumn{{3}}{{l}}{{\textit{{{kind}}}}} \\"]
    for key in keys:
        name = key[0] if kind.startswith("Judges") else key[2]
        default = (name == JUDGE and kind.startswith("Judges")) or (name == GENERATOR and kind == "Query writers")
        overall = overall_agreement({t: ratings(r) for t, r in R[key].items()})
        per_task = sum(scores(key).values()) / len(TASKS)
        cells = [name + (" (default)" if default else ""), num(overall), num(per_task)]
        if default:
            cells = [rf"\textbf{{{c}}}" for c in cells]
        lines.append(" & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Swapping in other judges and query writers, one at a time, on generated queries: Spearman "
          r"correlation with MTEB's ranking, of the overall ranking and per task (mean over 28 tasks). Decision models "
          r"are trained to judge and output a grade directly; general LLMs reason before grading.}",
          r"\label{tab:ablations}", r"\end{table}"]
(TABLES / "ablations.tex").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
