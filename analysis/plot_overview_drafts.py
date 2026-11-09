"""Three draft concepts for the MTEB Gym overview figure (HTML printed to PDF/PNG with headless Chromium).

overview_arena:   one query, models compete, the judge referees, a podium comes out; checked against MTEB and humans.
overview_compare: how MTEB ranks models today (human queries and labels) against MTEB Gym (your documents, two LLMs).
overview_funnel:  from corpora to one leaderboard, with the real counts at each step.
All numbers come from the published records (generated queries, JEV-27B, pointwise)."""

import base64

import numpy as np
from playwright.sync_api import sync_playwright

from common import GENERATOR, JUDGE, TASKS, judged, labels, overall_agreement, ratings, records
from plots import BLUE, FIGURES, ORANGE

R = records()
recs = R[(JUDGE, "pointwise", GENERATOR)]
per_task = {t: ratings(recs[t]) for t in TASKS}
models = sorted(per_task[TASKS[0]])
gym = sorted(models, key=lambda m: -np.mean([per_task[t][m] for t in TASKS]))
mteb = sorted(models, key=lambda m: -np.mean([labels(t)[m] for t in TASKS]))
RHO, RHO_HUMAN = overall_agreement(per_task), 0.857  # human value: plot_humans.py
N_QUERIES = sum(r["diagnostics"]["queries_rated"] for r in recs.values())
N_GRADES = sum(r["diagnostics"]["n_grades"] for r in recs.values())
_, _, top, grades, _ = judged(recs["SciFact"])
SCALE = ["#eef2f9", "#cbd7ee", "#9fb4df", "#6b8bcd", "#3a64b8", BLUE]


def img(org, size, radius=8):
    data = base64.b64encode((FIGURES / "logos" / f"{org}.png").read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}" width="{size}" height="{size}" style="border-radius:{radius}px">'


def mlogo(m, size=30):
    return img(m.split("/")[0], size)


def short(m):
    return m.split("/")[-1].replace("baseline-bm25s", "BM25").replace("-embedding-english-r2", "-embedding-r2")


def squares(m, qid="q50"):
    return "".join(f'<i style="background:{SCALE[min(int(round(grades[qid].get(d, 0))), 5)]}"></i>'
                   for d in top[m][qid][:10])


ICON = {
    "docs": '<path d="M8 3h7l5 5v13H8z"/><path d="M15 3v5h5"/><path d="M4 7v14h11"/>',
    "pen": '<path d="M12 20h9"/><path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4Z"/>',
    "user": '<circle cx="12" cy="8" r="4"/><path d="M4 21a8 8 0 0 1 16 0"/>',
    "tag": '<path d="M20.6 13.4 13.4 20.6a2 2 0 0 1-2.8 0L3 13V3h10l7.6 7.6a2 2 0 0 1 0 2.8z"/><circle cx="7.5" cy="7.5" r="1.5"/>',
    "query": '<circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/>',
    "list": '<path d="M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01"/>',
    "star": '<path d="m12 3 2.7 5.6 6.1.9-4.4 4.3 1 6.1L12 17l-5.4 2.9 1-6.1-4.4-4.3 6.1-.9z"/>',
    "trophy": '<path d="M8 21h8M12 17v4M7 4h10v5a5 5 0 0 1-10 0z"/><path d="M17 5h3v2a3 3 0 0 1-3 3M7 5H4v2a3 3 0 0 0 3 3"/>',
}


def icon(name, size=28, color="currentColor", width=2):
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" '
            f'stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round">{ICON[name]}</svg>')


BASE = f"""<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,500;0,600;0,700;0,800;1,600&display=swap" rel="stylesheet">
<style>* {{ box-sizing: border-box; margin: 0; }}
body {{ font-family: Inter, sans-serif; color: #1a2233; background: #fff; display: inline-block; padding: 18px; }}
.sq i {{ display: inline-block; width: 13px; height: 13px; border-radius: 3px; margin-right: 2px; }}
.chip {{ display: inline-flex; align-items: center; gap: 8px; border-radius: 999px; padding: 6px 14px; font-weight: 700; }}
"""


def podium(order, accent):
    heights, places = [110, 150, 80], [1, 0, 2]
    cols = ""
    for slot, i in enumerate(places):
        m = order[i]
        cols += (f'<div class="pcol"><div class="plogo">{mlogo(m, 40)}</div><div class="pname">{short(m)}</div>'
                 f'<div class="pblock" style="height:{heights[slot]}px;background:{accent}">{i + 1}</div></div>')
    return f'<div class="podium">{cols}</div>'


# ---------------------------------------------------------------- A: arena
arena_models = ["Qwen/Qwen3-Embedding-8B", "ibm-granite/granite-embedding-english-r2", "lightonai/DenseOn",
                "sentence-transformers/all-MiniLM-L6-v2", "mteb/baseline-bm25s", "microsoft/harrier-oss-v1-0.6b"]
pos = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]
cards = "".join(f'<div class="mc">{mlogo(m, 30)}<div><b>{short(m)}</b><div class="sq">{squares(m)}</div></div></div>'
                for m in arena_models)
ARENA = BASE + f""".wrap {{ display: flex; align-items: center; gap: 26px; }}
.src {{ display: flex; flex-direction: column; align-items: center; gap: 12px; width: 210px; }}
.dark {{ background: #111; color: #fff; border-radius: 16px; padding: 20px; width: 210px; text-align: center; }}
.dark b {{ font-size: 22px; display: block; margin-top: 8px; }} .dark span {{ color: #9aa3b5; font-size: 14px; }}
.writer {{ display: flex; align-items: center; gap: 10px; font-size: 15px; font-weight: 700; }}
.ring {{ position: relative; border: 2.5px dashed #c9d2e3; border-radius: 28px; padding: 26px; width: 760px; }}
.ring .tag {{ position: absolute; top: -15px; left: 28px; background: #fff; padding: 0 10px; font-weight: 800;
              font-size: 18px; letter-spacing: .5px; color: #5a6478; }}
.q {{ margin: 0 auto 22px; width: 520px; border: 2.5px solid #1a2233; border-radius: 14px; padding: 14px 18px;
      font-size: 19px; font-weight: 700; text-align: center; background: #fff7f1; }}
.q small {{ display: block; font-size: 13px; color: {ORANGE}; font-weight: 800; letter-spacing: .6px; margin-bottom: 4px; }}
.grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }}
.mc .sq {{ white-space: nowrap; }}
.mc .sq i {{ width: 11px; height: 11px; }}
.mc {{ display: flex; gap: 10px; align-items: center; border: 1.5px solid #dfe3ea; border-radius: 12px; padding: 10px 12px; }}
.mc b {{ font-size: 14px; display: block; margin-bottom: 5px; white-space: nowrap; }}
.ref {{ margin: 20px auto 0; width: 560px; white-space: nowrap; display: flex; align-items: center; gap: 14px; justify-content: center;
        background: #1a2233; color: #fff; border-radius: 999px; padding: 10px 18px; font-size: 16px; font-weight: 700; }}
.ref span {{ color: #c9cfdb; font-weight: 600; }}
.right {{ display: flex; flex-direction: column; align-items: center; gap: 18px; width: 330px; }}
.podium {{ display: flex; align-items: flex-end; gap: 10px; }}
.pcol {{ display: flex; flex-direction: column; align-items: center; width: 102px; }}
.pname {{ font-size: 12.5px; font-weight: 700; text-align: center; height: 34px; margin-top: 4px; }}
.pblock {{ width: 100%; border-radius: 8px 8px 0 0; color: #fff; font-size: 30px; font-weight: 800;
           display: flex; justify-content: center; padding-top: 8px; }}
.cards {{ display: flex; gap: 12px; }}
.card {{ border: 2px solid #1a2233; border-radius: 14px; padding: 12px 16px; text-align: center; width: 156px; }}
.card .v {{ font-size: 32px; font-weight: 800; color: {BLUE}; }} .card .l {{ font-size: 13.5px; font-weight: 600; color: #5a6478; }}
.arrow {{ font-size: 40px; color: #1a2233; }}
</style></head><body><div class="wrap">
<div class="src"><div class="dark">{icon("docs", 52, "#fff", 1.6)}<b>Your corpus</b><span>no queries · no labels</span></div>
  <div class="writer">{img("deepseek-ai", 34)}query writer</div></div>
<div class="arrow">→</div>
<div class="ring"><span class="tag">ONE QUERY, 28 MODELS</span>
  <div class="q"><small>GENERATED QUERY</small>What non-genetic factors can trigger cancer development in the presence of oncogenic mutations?</div>
  <div class="grid">{cards}</div>
  <div class="ref">{img("autotrust", 30)} Judge grades every result 0–5 <span>· × {N_QUERIES:,} queries</span></div>
</div>
<div class="arrow">→</div>
<div class="right">{podium(gym, BLUE)}
  <div class="cards"><div class="card"><div class="v">{RHO:.2f}</div><div class="l">agreement with<br>MTEB's labels</div></div>
  <div class="card"><div class="v">{RHO_HUMAN:.2f}</div><div class="l">agreement with<br>human votes</div></div></div></div>
</div></body></html>"""

# ---------------------------------------------------------------- B: compare
def lb(order):
    return "".join(f'<div class="lr"><span>{i + 1}</span>{mlogo(m, 24)}{short(m)}</div>' for i, m in enumerate(order[:3]))


def step(ic, title, sub, human=False):
    color = "#5a6478" if human else BLUE
    return (f'<div class="st">{icon(ic, 30, color, 2)}<div><b>{title}</b><span>{sub}</span></div></div>')


COMPARE = BASE + f""".cols {{ display: flex; gap: 40px; align-items: stretch; }}
.col {{ width: 520px; border-radius: 20px; padding: 24px; display: flex; flex-direction: column; gap: 14px; }}
.old {{ background: #f3f4f7; }} .new {{ background: #eef3fb; border: 2.5px solid {BLUE}; }}
.col h2 {{ font-size: 28px; font-weight: 800; }} .col h2 small {{ font-size: 16px; font-weight: 600; color: #5a6478; margin-left: 8px; }}
.st {{ display: flex; gap: 14px; align-items: center; background: #fff; border-radius: 12px; padding: 12px 16px; }}
.st b {{ display: block; font-size: 18px; }} .st span {{ font-size: 14.5px; color: #5a6478; }}
.down {{ text-align: center; font-size: 30px; line-height: 1; color: #8a93a6; }}
.board {{ background: #1a2233; color: #fff; border-radius: 14px; padding: 14px 18px; }}
.board .h {{ display: flex; align-items: center; gap: 8px; font-weight: 800; font-size: 16px; margin-bottom: 8px; }}
.lr {{ display: flex; align-items: center; gap: 10px; font-size: 16px; font-weight: 600; padding: 3px 0; }}
.lr span {{ width: 18px; color: #8a93a6; }}
.mid {{ display: flex; flex-direction: column; justify-content: flex-end; align-items: center; padding-bottom: 52px;
        gap: 6px; width: 170px; margin: 0 -40px; z-index: 2; }}
.eq {{ background: #fff; border: 2.5px solid #1a2233; border-radius: 999px; padding: 10px 16px; font-size: 26px;
       font-weight: 800; color: {BLUE}; white-space: nowrap; }}
.eqs {{ font-size: 13px; font-weight: 700; color: #5a6478; text-align: center; }}
</style></head><body><div class="cols">
<div class="col old"><h2>MTEB<small>today</small></h2>
  {step("user", "People write queries", "for each dataset", True)}
  {step("tag", "People label relevant documents", "for every query", True)}
  {step("list", "Models retrieve; nDCG on the labels", "only datasets that have labels", True)}
  <div class="down">↓</div>
  <div class="board"><div class="h">{icon("trophy", 20, "#fff")}Leaderboard</div>{lb(mteb)}</div></div>
<div class="mid"><div class="eq">ρ = {RHO:.2f}</div><div class="eqs">same ranking<br>28 models · 28 tasks</div></div>
<div class="col new"><h2>MTEB Gym<small>any corpus</small></h2>
  {step("docs", "Your documents", "no queries, no labels")}
  <div class="st">{img("deepseek-ai", 30)}<div><b>An LLM writes queries</b><span>from the documents and the task</span></div></div>
  <div class="st">{img("autotrust", 30)}<div><b>An LLM judge grades results</b><span>0–5 for each model's top 10</span></div></div>
  <div class="down">↓</div>
  <div class="board"><div class="h">{icon("trophy", 20, "#fff")}Leaderboard</div>{lb(gym)}</div></div>
</div></body></html>"""

# ---------------------------------------------------------------- C: funnel
stages = [("docs", "28 corpora", "your documents", 260), ("query", f"{N_QUERIES:,}", "generated queries", 220),
          ("list", "28 × top 10", "results per query", 180), ("star", f"{N_GRADES // 1000}k", "judge grades", 140),
          ("trophy", "1", "leaderboard", 100)]
blocks = ""
for i, (ic, big, small, h) in enumerate(stages):
    who = ""
    if i == 1:
        who = f'<div class="who">{img("deepseek-ai", 26)}query writer</div>'
    if i == 2:
        who = f'<div class="who">{"".join(mlogo(m, 22) for m in arena_models[:4])}</div>'
    if i == 3:
        who = f'<div class="who">{img("autotrust", 26)}judge</div>'
    shade = ["#111", "#24324d", "#2e4f8f", "#3a64b8", BLUE][i]
    blocks += (f'<div class="stage"><div class="blk" style="height:{h}px;background:{shade}">{icon(ic, 34, "#fff", 1.8)}'
               f'<b>{big}</b><span>{small}</span></div>{who}</div>')
    if i < len(stages) - 1:
        blocks += '<div class="chev">›</div>'
FUNNEL = BASE + f""".row {{ display: flex; align-items: center; gap: 6px; }}
.stage {{ display: flex; flex-direction: column; align-items: center; gap: 12px; width: 190px; }}
.blk {{ width: 190px; border-radius: 16px; color: #fff; display: flex; flex-direction: column; align-items: center;
        justify-content: center; gap: 6px; }}
.blk b {{ font-size: 30px; font-weight: 800; }} .blk span {{ font-size: 15px; color: #dfe6f3; font-weight: 600; }}
.who {{ display: flex; align-items: center; gap: 6px; font-size: 14px; font-weight: 700; color: #3d4556; height: 28px; }}
.chev {{ font-size: 60px; color: #c9d2e3; font-weight: 300; margin-top: -40px; }}
.checks {{ display: flex; flex-direction: column; gap: 14px; margin-left: 24px; margin-top: -40px; }}
.ck {{ border: 2px solid #1a2233; border-radius: 14px; padding: 10px 18px; width: 230px; }}
.ck b {{ font-size: 30px; color: {BLUE}; font-weight: 800; }} .ck span {{ display: block; font-size: 14px; color: #5a6478; font-weight: 600; }}
</style></head><body><div class="row">{blocks}
<div class="checks"><div class="ck"><b>ρ = {RHO:.2f}</b><span>with MTEB's ranking (human labels)</span></div>
<div class="ck"><b>ρ = {RHO_HUMAN:.2f}</b><span>with human votes (MTEB Arena)</span></div></div>
</div></body></html>"""

with sync_playwright() as p:
    browser = p.chromium.launch()
    for name, html in (("overview_arena", ARENA), ("overview_compare", COMPARE), ("overview_funnel", FUNNEL)):
        page = browser.new_page(viewport={"width": 2400, "height": 1000}, device_scale_factor=2)
        page.set_content(html, wait_until="networkidle")
        size = page.locator("body").bounding_box()
        w, h = size["width"], size["height"]
        page.set_viewport_size({"width": int(w) + 1, "height": int(h) + 1})
        page.pdf(path=str(FIGURES / f"{name}.pdf"), width=f"{w}px", height=f"{h}px", print_background=True,
                 page_ranges="1")
        page.screenshot(path=str(FIGURES / f"{name}.png"), clip={"x": 0, "y": 0, "width": w, "height": h})
        print("wrote", name)
    browser.close()
