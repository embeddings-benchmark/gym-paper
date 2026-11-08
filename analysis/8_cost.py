"""8. Cost and speed, from measured logs only. Anything not measured is left out.

- JEV-27B judging the Qwen3.5-4B queries (data/logs/small_judge.log): grades and wall time from the log's own timestamps.
- Qwen3.5-4B generation (data/logs/small_gen.log): wall time from its timestamps.
- Single-GPU grade rates for JEV-9B, GEV-26B, JEV-35B, d1-3B (data/logs/speed/models-*.out, the measured lines of the run logs): SciFact, 200 grades.
- Chat judges' completion tokens per grade (data/logs/chat_tokens): logged only in the final no-cap passes, which mostly
  regraded the hardest documents, so they overstate a typical grade. A decision model's output is one token.
"""

import ast
import glob
import json
import re
import statistics as st
from datetime import datetime, timedelta

from common import DATA, write

LOGS = DATA / "logs"


def span(lines):
    ts = [datetime.strptime(m.group(1), "%H:%M:%S") for l in lines if (m := re.match(r"(\d\d:\d\d:\d\d) ", l))]
    if not ts:
        return None
    d = ts[-1] - ts[0]
    return (d if d >= timedelta(0) else d + timedelta(days=1)).total_seconds()


rows = []
judge = [l for l in open(LOGS / "small_judge.log") if re.match(r"\d\d:\d\d:\d\d gym-jev", l)]
grades = sum(ast.literal_eval(l[l.index("{"):])["n_grades"] for l in judge if "n_grades" in l)
sec = span(judge)
rows.append(dict(what="JEV-27B pointwise judging, Qwen3.5-4B queries, 28 tasks", measured=f"{grades} grades",
                 time=f"{sec / 60:.0f} min between first and last finished task (starts not logged)",
                 rate=f"{grades / sec:.0f} grades/s on 6 GPUs" if sec else "", source="data/logs/small_judge.log"))
gen = [l for l in open(LOGS / "small_gen.log") if re.match(r"\d\d:\d\d:\d\d ", l)]
rows.append(dict(what="Qwen3.5-4B generation, 28 tasks x 100 queries", measured=f"{len(gen)} tasks logged",
                 time=f"{span(gen) / 60:.1f} min between first and last finished task (2 servers; the agent measured about 3 min in all)", rate="", source="data/logs/small_gen.log"))
for f in sorted(glob.glob(str(LOGS / "speed" / "models-*.out"))):
    route = ""
    for line in open(f):
        if line.startswith("==="):
            route = line.strip("= \n")[:60]  # the step the log was in, such as "direct" or a serve command
        m = re.match(r"(one at a time|32 at a time)?\s*([\d.]+) grades/s", line.strip())
        if m:
            rows.append(dict(what=f"{f.split('models-')[1].split('-')[0]}, {route or 'see log'}, {m.group(1) or 'one client'}",
                             measured="200 SciFact grades", time="", rate=f"{m.group(2)} grades/s, 1 GPU", source=f.split("/")[-1]))
tok = {"Qwen3.8-27B": [], "DeepSeek-V4-Flash": []}
for f in glob.glob(str(LOGS / "chat_tokens" / "*.jsonl")):
    j = "Qwen3.8-27B" if "Qwen" in f else "DeepSeek-V4-Flash"
    for line in open(f):
        try:
            tok[j].append(json.loads(line)["tokens"])
        except (ValueError, KeyError):
            pass
for j, v in tok.items():
    rows.append(dict(what=f"{j} chat judge, completion tokens per grade", measured=f"{len(v)} logged grades (biased to hard ones)",
                     time="", rate=f"median {st.median(v):.0f}, max {max(v)} tokens", source="data/logs/chat_tokens"))
write("8_cost", rows, __doc__.strip())
for r in rows:
    print(r)
