"""Figure 1: MTEB Gym in four steps, drawn on real SciFact data (HTML printed to PDF/PNG with headless Chromium).

Given a corpus, an LLM writes queries from sampled documents; every embedding model retrieves its top 10 and an LLM
judge grades each result 0-5; the models are ranked by nDCG@10 with those grades, which matches MTEB's ranking.
Icons: Lucide (ISC); logos from Hugging Face."""

import base64
import glob
import json

import numpy as np
from playwright.sync_api import sync_playwright

from common import GENERATOR, JUDGE, TASKS, judged, labels, overall_agreement, ratings, records, runs_folder
from mteb_gym import corpus
from plots import FIGURES

TASK, QID, OTHER_QIDS = "SciFact", "q50", ("q26", "q6")
STRONG, WEAK = "Qwen/Qwen3-Embedding-8B", "sentence-transformers/all-MiniLM-L6-v2"
NAVY, INK, PANEL, CREAM, ORANGE = "#123a7a", "#14213d", "#e3ebf7", "#fbf8f1", "#ff5a0a"

R = records()
recs = R[(JUDGE, "pointwise", GENERATOR)]
per_task = {t: ratings(recs[t]) for t in TASKS}
models = sorted(per_task[TASKS[0]])
score = {m: np.mean([per_task[t][m] for t in TASKS]) for m in models}
gym = sorted(models, key=lambda m: -score[m])
mteb = sorted(models, key=lambda m: -np.mean([labels(t)[m] for t in TASKS]))
RHO = overall_agreement(per_task)

config = recs[TASK]["config"]
_, _, top, grades, _ = judged(recs[TASK])
docs = corpus.load(TASK).docs
folder = runs_folder(f"*/{TASK}/*")
queries = {q["qid"]: q for f in glob.glob(f"{folder}/queries/{TASK}/*") for q in json.load(open(f))["queries"]
           if config["query_set"].startswith(f.rsplit("/", 1)[1][:-5])}
query = queries[QID]


SEED = next(d for d in query["seed_doc_ids"] if d in top[STRONG][QID][:3])  # the sampled document the query is about
probs = next([row["probabilities"][str(g)] for g in range(6)]
             for f in glob.glob(f"{folder}/grades/{TASK}/{config['judge_model'].replace('/', '_')}-*.jsonl")
             for row in map(json.loads, open(f))
             if row["query_set"] == config["query_set"] and row["qid"] == QID and row["doc_id"] == SEED)
PAIR_QID = "q30"  # a query with a pairwise verdict for STRONG vs WEAK
pair = next(json.loads(line) for f in glob.glob(f"{folder}/verdicts/{TASK}/{STRONG.replace('/', '_')}__{WEAK.replace('/', '_')}-*.jsonl")
            for line in open(f) if json.loads(line)["query_set"] == config["query_set"] and json.loads(line)["qid"] == PAIR_QID)



def data_uri(path, mime):
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def icon(name, size, color=NAVY, width=2):
    svg = (FIGURES / "icons" / "line" / f"{name}.svg").read_text().split("-->", 1)[-1]
    svg = svg.replace("currentColor", color).replace('stroke-width="2"', f'stroke-width="{width}"')
    return svg.replace('width="24"', f'width="{size}"').replace('height="24"', f'height="{size}"')


def logo(org, size):
    return (f'<img src="{data_uri(FIGURES / "logos" / f"{org}.png", "image/png")}" width="{size}" height="{size}" '
            f'style="border-radius:{size // 4}px">')


def short(m):
    return m.split("/")[-1]


def title(d, n):
    text = docs[d] if isinstance(docs[d], str) else json.dumps(docs[d])
    return text[:n].rsplit(" ", 1)[0] + "…" if len(text) > n else text


def actor(name, org, x, y):
    """A step's actor: what it does as an icon, the model that does it as a logo badge."""
    return (f'<div class="actor" style="left:{x}px;top:{y}px">{icon(name, 50, NAVY, 2)}'
            f'<span class="badge">{logo(org, 30)}</span></div>')


def qcard(q, x, y, main=False):
    return (f'<div class="qcard{" main" if main else ""}" style="left:{x}px;top:{y}px">'
            f'{icon("search", 24, NAVY, 2.4)}<span>{q["text"]}</span></div>')


def color(value, high, low):
    return BLUE if value >= high else "#7d97c9" if value >= low else "#d1342b"


def pointwise(x, y):
    rows = "".join(f'<div class="li">{icon("file-text", 24, "#8a9bbd", 2)}<span>{title(d, 30)}</span>'
                   f'<b style="color:{color(grades[QID][d], 3, 2)}">{grades[QID][d]:.1f}</b></div>'
                   for d in (top[STRONG][QID][1], top[STRONG][QID][0], top[WEAK][QID][2]))
    return (f'<div class="card" style="left:{x}px;top:{y}px"><div class="ch">{icon("list-checks", 28, NAVY, 2.2)}'
            f'Pointwise<small>grade 0–5</small></div>{rows}</div>')


def pairwise(x, y):
    a = pair["score_a"]
    rows = "".join(f'<div class="li">{logo(m.split("/")[0], 26)}<span>{short(m)}</span>'
                   f'<b style="color:{color(p, .5, .5)}">{p:.2f}</b></div>' for m, p in ((STRONG, a), (WEAK, 1 - a)))
    return (f'<div class="card" style="left:{x}px;top:{y}px"><div class="ch">{icon("git-compare", 28, NAVY, 2.2)}'
            f'Pairwise<small>which top 10 is better</small></div>{rows}</div>')


def dashed(path):
    return f'<path d="{path}" marker-end="url(#h)"/>'


BLUE = "#1f5bd6"
FILES = "".join(f'<div class="file">{icon("file-text", 30, "#8fb4f0", 2)}<span>{title(d, 24)}</span></div>'
                for d in [query["seed_doc_ids"][2], *list(docs)[:4]])
BOARD = "".join(f'<div class="row"><span class="k">{i + 1}</span><span class="nm">{short(m)}</span>'
                f'<b>{score[m]:.1f}</b><span class="mt">#{mteb.index(m) + 1}</span></div>'
                for i, m in enumerate(gym[:4])) + '<div class="rest">… 24 more</div>'
GRAIN = ("url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='160' height='160'>"
         "<filter id='n'><feTurbulence type='fractalNoise' baseFrequency='.85' numOctaves='2' stitchTiles='stitch'/>"
         "<feColorMatrix values='0 0 0 0 1  0 0 0 0 1  0 0 0 0 1  0 0 0 .07 0'/></filter>"
         "<rect width='100%' height='100%' filter='url(%23n)'/></svg>\")")
W, H = 2420, 550
ARROWS = f"""<svg class="lines" width="{W}" height="{H}" viewBox="0 0 {W} {H}" fill="none" stroke="{NAVY}" stroke-width="3"
  stroke-dasharray="9 8" stroke-linecap="round">
  <defs><marker id="h" viewBox="0 0 12 12" refX="9" refY="6" markerWidth="8" markerHeight="8" orient="auto">
    <path d="M2 2L10 6L2 10" fill="none" stroke="{NAVY}" stroke-width="2" stroke-dasharray="none"/></marker></defs>
  {dashed("M398 176 C 420 116, 450 96, 492 96")}{dashed("M454 228 L 492 228")}{dashed("M398 280 C 420 340, 450 360, 492 360")}
  {dashed("M1112 176 C 1130 126, 1150 112, 1178 112")}{dashed("M1112 280 C 1130 330, 1180 350, 1238 350")}
  {dashed("M1782 186 C 1770 130, 1730 112, 1690 112")}{dashed("M1782 272 C 1774 300, 1764 310, 1752 316")}
  <path d="M1892 228 L 1930 228" stroke-dasharray="none" stroke-width="4" marker-end="url(#h)"/>
</svg>"""

HTML = f"""<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Nunito:wght@600;700;800;900&display=swap" rel="stylesheet">
<style>
* {{ box-sizing: border-box; margin: 0; }}
body {{ font-family: Nunito, sans-serif; color: {NAVY}; background: {CREAM}; width: {W}px; height: {H}px; position: relative;
        overflow: hidden; }}
body > div {{ position: absolute; }}
.lines {{ position: absolute; left: 0; top: 0; z-index: 1; }}
.corpus {{ left: 30px; top: 46px; width: 370px; height: 364px; border-radius: 12px; background: {NAVY};
           background-image: {GRAIN}; color: #fff; padding: 20px 28px; box-shadow: 0 8px 20px rgba(18,58,122,.18); }}
.corpus h3 {{ display: flex; align-items: center; gap: 14px; font-size: 30px; font-weight: 800; padding-bottom: 14px;
              border-bottom: 1.5px solid rgba(255,255,255,.25); margin-bottom: 8px; }}
.file {{ display: flex; align-items: center; gap: 14px; font-size: 23px; font-weight: 700; padding: 8px 0; white-space: nowrap; }}
.panel {{ top: 28px; height: 400px; border-radius: 18px; background: {PANEL}; background-image: {GRAIN}; }}
.actor {{ z-index: 3; width: 112px; height: 112px; border-radius: 50%; background: #fff; border: 3px solid {NAVY};
          display: flex; align-items: center; justify-content: center;
          box-shadow: 0 0 0 9px rgba(18,58,122,.10), 0 8px 18px rgba(18,58,122,.2); }}
.badge {{ position: absolute; right: -6px; bottom: -2px; background: #fff; border-radius: 10px; padding: 3px; line-height: 0;
          box-shadow: 0 2px 6px rgba(18,58,122,.3); }}
.qcard {{ z-index: 2; width: 440px; background: #fff; border-radius: 12px; padding: 14px 18px; display: flex; gap: 12px;
          align-items: flex-start; font-size: 22px; font-weight: 800; line-height: 1.3;
          box-shadow: 0 8px 20px rgba(18,58,122,.14); border: 1.5px solid #d3dcea; }}
.qcard svg {{ flex: none; margin-top: 3px; }}
.qcard.main {{ border: 3px solid {NAVY}; }}
.card {{ z-index: 2; width: 450px; background: #fff; border-radius: 14px; padding: 14px 22px 8px;
         box-shadow: 0 8px 20px rgba(18,58,122,.14); border: 1.5px solid #d3dcea; }}
.ch {{ display: flex; align-items: center; gap: 12px; font-size: 25px; font-weight: 900; padding-bottom: 10px;
       border-bottom: 1.5px solid #e3e8f0; }}
.ch small {{ margin-left: auto; font-size: 17px; font-weight: 800; color: #6b7fa6; }}
.li {{ display: flex; align-items: center; gap: 12px; font-size: 20px; font-weight: 800; padding: 9px 0; }}
.li + .li {{ border-top: 1px solid #f0f2f6; }}
.li svg, .li img {{ flex: none; }}
.li span {{ flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.li b {{ font-size: 23px; font-weight: 900; font-variant-numeric: tabular-nums; }}
.board {{ left: 1944px; top: 46px; width: 460px; height: 364px; background: #fff; border-radius: 14px;
          border: 1.5px solid #d3dcea; box-shadow: 0 8px 20px rgba(18,58,122,.10); padding: 16px 24px; }}
.board h3 {{ display: flex; align-items: center; gap: 12px; font-size: 29px; font-weight: 900; padding-bottom: 12px;
             border-bottom: 1.5px solid #e3e8f0; margin-bottom: 4px; }}
.board h3 small {{ margin-left: auto; font-size: 17px; color: {ORANGE}; font-weight: 900; }}
.row {{ display: flex; align-items: center; gap: 12px; font-size: 22px; font-weight: 800; height: 62px;
        border-bottom: 1px solid #eef1f6; }}
.row .k {{ width: 30px; height: 30px; border-radius: 50%; background: {NAVY}; color: #fff; font-size: 17px; font-weight: 900;
           display: flex; align-items: center; justify-content: center; flex: none; }}
.row .nm {{ flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.row b {{ font-weight: 900; color: {BLUE}; font-size: 24px; }}
.row .mt {{ width: 44px; text-align: right; color: {ORANGE}; font-weight: 900; font-size: 22px; }}
.rest {{ position: static; padding: 10px 0 0 42px; font-size: 18px; font-weight: 800; color: #8a9bbd; }}
.cap {{ top: 452px; text-align: center; font-size: 30px; font-weight: 800; line-height: 1.2; }}
.cap small {{ display: block; font-size: 21px; font-weight: 700; color: #4b6390; margin-top: 4px; }}
</style></head><body>
{ARROWS}
<div class="corpus"><h3>{icon("folder-open", 34, "#fff", 2)}{TASK}</h3>{FILES}</div>
{actor("pen-line", "deepseek-ai", 342, 172)}
<div class="panel" style="left:470px;width:560px"></div>
{qcard(queries[OTHER_QIDS[0]], 506, 56)}
{qcard(query, 546, 180, main=True)}
{qcard(queries[OTHER_QIDS[1]], 506, 316)}
{actor("search", "Qwen", 1028, 172)}
<div class="panel" style="left:1140px;width:690px"></div>
{pointwise(1186, 52)}
{pairwise(1286, 262)}
{actor("scale", "autotrust", 1770, 172)}
<div class="board"><h3>{icon("trophy", 32, NAVY, 2.2)}Ranking<small>MTEB</small></h3>{BOARD}</div>
<div class="cap" style="left:30px;width:370px">Given a corpus<small>no queries, no relevance labels</small></div>
<div class="cap" style="left:470px;width:560px">an LLM writes realistic queries<small>{GENERATOR}, 100 per task</small></div>
<div class="cap" style="left:1140px;width:690px">each model retrieves, a judge grades<small>28 embedding models · {JUDGE}, a decision model</small></div>
<div class="cap" style="left:1944px;width:460px">then ranks the models<small>same top 3 as MTEB · ρ = {RHO:.2f}</small></div>
</body></html>"""

(FIGURES / "figure1.html").write_text(HTML)
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": W, "height": H}, device_scale_factor=2)
    page.set_content(HTML, wait_until="networkidle")
    page.pdf(path=str(FIGURES / "figure1.pdf"), width=f"{W}px", height=f"{H}px", print_background=True, page_ranges="1")
    page.screenshot(path=str(FIGURES / "figure1.png"))
    browser.close()
print("wrote", FIGURES / "figure1.pdf")
