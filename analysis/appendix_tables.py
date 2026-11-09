"""Appendix tables: Spearman correlation with MTEB's ranking on every task, for every setting that was run.

Rows: the 28 tasks grouped by benchmark, then the mean over tasks and the overall ranking (models ranked by their mean
score over the 28 tasks). Columns: one setting each. The best value in each row is bold.

    tables/all_main.tex      judging each document or two lists side by side, generated and original queries
    tables/all_judges.tex    every judge, on generated queries
    tables/all_writers.tex   every query writer, with JEV-27B judging"""

import numpy as np

from common import GENERATOR, JUDGE, TASKS, TABLES, agreement, num, overall_agreement, ratings, records
from plots import NAMES

R = records()
OTHER = ["SciFact", "NFCorpus", "FinQARetrieval", "AppsRetrieval", "AILACasedocs", "CUREv1"]
GROUPS = [("MTEB(eng, v2)", [t for t in TASKS if not t.startswith("Bright") and t not in OTHER]),
          ("BRIGHT", [t for t in TASKS if t.startswith("Bright")]),
          ("Other domains", OTHER)]


def column(key):
    per_task = {t: ratings(R[key][t]) for t in TASKS}
    values = {t: agreement(per_task[t], t) for t in TASKS}
    values["Mean per task"] = float(np.mean([values[t] for t in TASKS]))
    values["Overall ranking"] = overall_agreement(per_task)
    return values


def table(name, columns, caption, header_groups=None):
    cols = [(head, column(key)) for head, key in columns]
    lines = [r"\begin{table*}[t]", r"\centering\small", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule"]
    if header_groups:
        lines.append(" & " + " & ".join(rf"\multicolumn{{{n}}}{{c}}{{\textbf{{{g}}}}}" for g, n in header_groups)
                     + r" \\")
        start = 2
        rules = []
        for _, n in header_groups:
            rules.append(rf"\cmidrule(lr){{{start}-{start + n - 1}}}")
            start += n
        lines.append(" ".join(rules))
    lines.append(r"\textbf{Task} & " + " & ".join(rf"\textbf{{{h}}}" for h, _ in cols) + r" \\")

    def row(label, key):
        vals = [c[key] for _, c in cols]
        best = max(vals)
        cells = [rf"\textbf{{{num(v)}}}" if abs(v - best) < 5e-4 else num(v) for v in vals]
        return f"{label} & " + " & ".join(cells) + r" \\"

    for group, ts in GROUPS:
        lines += [r"\midrule", rf"\multicolumn{{{len(cols) + 1}}}{{l}}{{\textit{{{group}}}}} \\"]
        lines += [row(NAMES[t], t) for t in ts]
    lines += [r"\midrule", row("Mean per task", "Mean per task"), row("Overall ranking", "Overall ranking"),
              r"\bottomrule", r"\end{tabular}", rf"\caption{{{caption}}}", rf"\label{{tab:{name}}}", r"\end{table*}"]
    (TABLES / f"{name}.tex").write_text("\n".join(lines) + "\n")
    print("wrote", TABLES / f"{name}.tex")


table("all_main",
      [("Generated", (JUDGE, "pointwise", GENERATOR)), ("Original", (JUDGE, "pointwise", "original")),
       ("Generated", (JUDGE, "pairwise", GENERATOR)), ("Original", (JUDGE, "pairwise", "original"))],
      "Spearman correlation with MTEB's ranking on every task, with JEV-27B grading each document or comparing two "
      "models' lists side by side, on generated and on original queries. Best per row in bold.",
      header_groups=[("Each document", 2), ("Side by side", 2)])

JUDGES = [("JEV-27B", "JEV-27B"), ("AutoTrust-JEV-27B-v1.1", "AutoTrust"), ("JEV-35B", "JEV-35B"),
          ("Qwen3.8-27B", "Qwen3.8-27B"), ("GEV-26B-Decide", "GEV-26B"), ("JEV-9B", "JEV-9B"),
          ("decider-4b", "decider-4b"), ("DeepSeek-V4-Flash", "DeepSeek")]
table("all_judges", [(short, (j, "pointwise", GENERATOR)) for j, short in JUDGES],
      "Spearman correlation with MTEB's ranking on every task for each judge, on generated queries "
      "(DeepSeek-V4-Flash). JEV-27B is the default. Best per row in bold.")

WRITERS = [(GENERATOR, "DeepSeek-V4-Flash"), ("MiniMax-M2.7", "MiniMax-M2.7"), ("Qwen3.8-27B", "Qwen3.8-27B"),
           ("Qwen3.5-4B", "Qwen3.5-4B")]
table("all_writers", [(short, (JUDGE, "pointwise", w)) for w, short in WRITERS] +
      [("Original", (JUDGE, "pointwise", "original"))],
      "Spearman correlation with MTEB's ranking on every task for each query writer, with JEV-27B judging; the last "
      "column is the tasks' original queries. DeepSeek-V4-Flash is the default. Best per row in bold.")
