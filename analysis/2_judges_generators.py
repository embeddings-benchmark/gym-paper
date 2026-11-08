"""2. Every comparison judge and generator against JEV-27B on the same tasks: paired mean difference in Spearman with a
paired 95% bootstrap CI over tasks, and whether it passes the rule fixed before results (gain >= .03, CI excluding 0)."""

import csv
import statistics as st

from common import MAIN_GENERATOR, MAIN_JUDGE, DATA, TASKS, agree, boot_mean, labels, ratings, records, write

R = records()
rho = {k: {t: agree(ratings(r), labels(t))[0] for t, r in v.items()} for k, v in R.items()}
low = {}
for judge, f in (("Qwen3.8-27B", "low_effort_qwen.tsv"), ("DeepSeek-V4-Flash", "low_effort_deepseek.tsv")):
    rows_ = list(csv.reader(open(DATA / f), delimiter="\t"))
    low[judge] = len(rows_)


def compare(key, base):
    tasks = sorted(set(rho[key]) & set(rho[base]))
    diff = [rho[key][t] - rho[base][t] for t in tasks]
    d, lo, hi = boot_mean(diff)
    return len(tasks), d, lo, hi, d >= 0.03 and lo > 0


rows = []
for (judge, judging, queries), v in sorted(R.items()):
    if (judge, judging, queries) in {(MAIN_JUDGE, j, q) for j in ("pointwise", "pairwise") for q in ("original", MAIN_GENERATOR)}:
        continue
    # a judge is compared on the same queries; a generator is compared with JEV-27B judging DeepSeek-V4-Flash's queries
    base = (MAIN_JUDGE, judging, queries if judge != MAIN_JUDGE else MAIN_GENERATOR)
    if base not in rho:
        continue
    n, d, lo, hi, passes = compare((judge, judging, queries), base)
    vals = list(rho[(judge, judging, queries)].values())
    kind = "generator" if judge == MAIN_JUDGE else "judge"
    rows.append(dict(kind=kind, judge=judge, judging=judging, queries=queries, tasks=n, mean_rho=st.mean(vals),
                     median_rho=st.median(vals), diff_vs_jev27b=d, ci_low=lo, ci_high=hi, passes_rule=passes,
                     note=f"{low[judge]} grades at low reasoning effort" if judge in low else ""))
write("2_judges_generators", rows,
      "Each row against JEV-27B on the same tasks: judges on the same queries, generators against JEV-27B on "
      "DeepSeek-V4-Flash's queries. Rule fixed before results: replace JEV-27B only for a gain >= .03 with the 95% CI "
      "excluding 0. Low-effort grades: data/low_effort_*.tsv (temperature 0, no cap otherwise).")
for r in rows:
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
