"""3. Agreement at the top of the ranking: rank-biased overlap (Webber, Moffat and Zobel, 2010, "A similarity measure
for indefinite rankings", TOIS) at p = 0.9, between each method's ranking and the labels' ranking of the 28 models, and
how often the method's top pick is the labels' best model or in the labels' top 3. Rankings are complete and of equal
length, so RBO is computed exactly over the 28 depths with the extrapolated tail of Webber et al. (eq. 32)."""

import statistics as st

from common import MAIN_JUDGE, TASKS, boot_mean, labels, ratings, records, write
from history import label_history, published_history

P = 0.9


def rbo(a: list[str], b: list[str], p: float = P) -> float:
    k = len(a)
    overlap = [len(set(a[:d]) & set(b[:d])) for d in range(1, k + 1)]
    return float((1 - p) / p * sum(overlap[d - 1] / d * p**d for d in range(1, k + 1)) + overlap[-1] / k * p**k)


def order(scores, models):
    return sorted(models, key=lambda m: -scores[m])


def methods():  # the methods compared in 3-6
    R = records()
    out = {f"{MAIN_JUDGE} {j} {q}": {t: ratings(R[(MAIN_JUDGE, j, gen)][t]) for t in TASKS}
           for j in ("pointwise", "pairwise") for q, gen in (("original", "original"), ("synthetic", "DeepSeek-V4-Flash"))}
    out["history (labels)"] = {t: label_history(t) for t in TASKS}
    out["history (published)"] = {t: published_history(t) for t in TASKS}
    return out


if __name__ == "__main__":
    rows = []
    for name, by_task in methods().items():
        r, top1, top3 = [], [], []
        for t in TASKS:
            y = labels(t)
            s = {m: v for m, v in by_task[t].items() if v == v}  # a model with no history has no place
            models = sorted(set(s) & set(y))
            truth, mine = order(y, models), order(s, models)
            r.append(rbo(mine, truth))
            top1.append(mine[0] == truth[0])
            top3.append(mine[0] in order(y, sorted(y))[:3])
        m, lo, hi = boot_mean(r)
        rows.append(dict(method=name, tasks=len(r), mean_rbo=m, ci_low=lo, ci_high=hi, median_rbo=st.median(r),
                         top1=sum(top1), top_pick_in_label_top3=sum(top3)))
    write("3_top_weighted", rows, f"RBO at p = {P} (Webber et al., 2010) against the labels' ranking; top-1 and top-3 "
          "counted out of 28 tasks. History baselines: see history.py.")
    for x in rows:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in x.items()})
