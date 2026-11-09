"""Supporting figure: the overall ranking agreement with MTEB under every setting we varied, as a grid of tiles.

Fewer queries per task (num_queries.py), fewer documents judged per model (num_documents.py), other judges and other
query writers (judges.py). All with generated queries; the gym's defaults are outlined."""

import csv

from common import GENERATOR, JUDGE, TABLES
from plots import CREAM, GRAIN, NAVY, ORANGE, FIGURES, render
import base64


def rows(name):
    return list(csv.DictReader(open(TABLES / name)))


queries = [(f'{r["n"]}', float(r["agreement"])) for r in rows("num_queries_overall.csv") if r["queries"] == GENERATOR]
documents = [(f'top {r["n"]}', float(r["agreement"])) for r in rows("num_documents_overall.csv") if r["queries"] == GENERATOR]
ablations, group = {}, None
for r in rows("ablations.csv"):
    group = r[""] or group
    ablations.setdefault(group, []).append((r["Model"].replace(" (default)", ""), float(r["Overall"])))

ORG = {"JEV": "autotrust", "AutoTrust": "autotrust", "GEV": "autotrust", "decider": None, "Qwen": "Qwen",
       "DeepSeek": "deepseek-ai", "MiniMax": "MiniMaxAI"}
SIZE = {"JEV-27B": "27B", "AutoTrust-JEV-27B-v1.1": "27B", "JEV-35B": "35B", "GEV-26B-Decide": "26B", "JEV-9B": "9B",
        "decider-4b": "4B", "Qwen3.8-27B": "27B", "Qwen3.5-4B": "4B", "DeepSeek-V4-Flash": "", "MiniMax-M2.7": ""}


def logo(name):
    org = next(o for k, o in ORG.items() if name.startswith(k))
    if org is None:  # no organization logo: the model's initial
        return f'<span class="initial">{name[0]}</span>'
    data = base64.b64encode((FIGURES / "logos" / f"{org}.png").read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}" width="30" height="30">'


def tile(label, value, model=False, kind="", default=None):
    head = f'<div class="mh">{logo(label)}<span>{label}</span></div>' if model else f'<div class="big">{label}</div>'
    size = f'<em>{SIZE[label]}</em>' if model and SIZE.get(label) else ""
    tag = f'<small class="kind">{kind}</small>' if kind else ""
    cls = "tile model" if model else "tile"
    if label == default:
        cls += " default"
    return f'<div class="{cls}">{tag}{size}{head}<b>{value:.2f}</b></div>'


def row(title, sub, tiles):
    return f'<div class="row"><div class="lbl">{title}<small>{sub}</small></div>{"".join(tiles)}</div>'


judges = ([tile(m, v, True, "decision model", JUDGE) for m, v in ablations["Judges: decision models"]]
          + [tile(m, v, True, "general LLM") for m, v in ablations["Judges: general LLMs"]])
writers = [tile(m, v, True, default=GENERATOR) for m, v in ablations["Query writers"]]
values = [v for _, v in queries + documents] + [v for g in ablations.values() for _, v in g]
FOOT = f"""<div class="foot">
  <div class="call"><b>EVERY SETTING</b><span>overall ranking<br>ρ = {min(values):.2f} – {max(values):.2f}</span></div>
  <div class="legend"><div><i></i>Spearman with MTEB's<br>ranking, 28 models, 28 tasks</div><div><i class="d"></i>default</div></div>
</div>"""
BODY = (row("Queries", "per task", [tile(n, v, default="100") for n, v in queries]
            + [f'<div class="gap"></div><div class="lbl">Documents<small>judged per model</small></div>']
            + [tile(n, v, default="top 10") for n, v in documents])
        + row("Judge", "8 models", judges)
        + row("Query writer", "4 models", writers + [FOOT]))

W, H = 1990, 462
HTML = f"""<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Nunito:wght@700;800;900&display=swap" rel="stylesheet">
<style>
* {{ box-sizing: border-box; margin: 0; }}
body {{ font-family: Nunito, sans-serif; color: {NAVY}; background: {CREAM}; width: {W}px; height: {H}px; padding: 30px 30px; }}
.row {{ display: flex; gap: 12px; margin-bottom: 16px; align-items: stretch; }}
.lbl {{ width: 200px; flex: none; display: flex; flex-direction: column; justify-content: center; font-size: 27px;
        font-weight: 900; line-height: 1.1; white-space: nowrap; }}
.lbl small {{ font-size: 17px; font-weight: 800; color: #4b6390; margin-top: 3px; }}
.tile {{ position: relative; width: 150px; height: 112px; border-radius: 10px; background: {NAVY}; background-image: {GRAIN};
         color: #fff; display: flex; flex-direction: column; justify-content: center; align-items: center; gap: 4px;
         box-shadow: 0 3px 8px rgba(18,58,122,.12); }}
.tile.model {{ width: 206px; }}
.tile.default {{ box-shadow: 0 0 0 4px {CREAM}, 0 0 0 8px {ORANGE}; }}
.big {{ font-size: 21px; font-weight: 800; opacity: .85; }}
.tile b {{ font-size: 36px; font-weight: 900; line-height: 1; }}
.mh {{ display: flex; align-items: center; gap: 8px; font-size: 16.5px; font-weight: 900; max-width: 186px; }}
.initial {{ width: 30px; height: 30px; border-radius: 7px; background: #fff; color: {NAVY}; display: inline-flex; align-items: center; justify-content: center; font-size: 19px; font-weight: 900; flex: none; }}
.mh img {{ border-radius: 7px; background: #fff; flex: none; }}
.mh span {{ white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.tile em {{ position: absolute; top: 7px; right: 9px; font-style: normal; font-size: 13px; font-weight: 900;
            background: rgba(255,255,255,.18); border-radius: 6px; padding: 1px 6px; }}
.kind {{ position: absolute; top: 7px; left: 9px; font-size: 12px; font-weight: 800; letter-spacing: .3px; opacity: .75; }}
.foot {{ display: flex; gap: 18px; margin-left: 20px; align-items: stretch; }}
.gap {{ width: 40px; }}
.call {{ background: {ORANGE}; color: #fff; border-radius: 16px; padding: 14px 26px; display: flex; flex-direction: column;
         justify-content: center; }}
.call b {{ font-size: 18px; font-weight: 900; letter-spacing: 1.2px; }}
.call span {{ font-size: 24px; font-weight: 900; line-height: 1.2; }}
.legend {{ background: #fff; border-radius: 16px; border: 1.5px solid #e6e1d6; padding: 14px 26px; display: flex; gap: 34px;
           align-items: center; font-size: 18px; font-weight: 800; line-height: 1.2; }}
.legend div {{ display: flex; align-items: center; gap: 12px; }}
.legend i {{ width: 44px; height: 32px; border-radius: 7px; background: {NAVY}; }}
.legend i.d {{ box-shadow: 0 0 0 3px #fff, 0 0 0 6px {ORANGE}; }}
</style></head><body>
{BODY}
</body></html>"""

render(HTML, "settings_grid", W, H)
