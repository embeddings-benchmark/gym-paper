"""Figure 1: MTEB Gym in four steps, drawn on real SciFact data (HTML printed to PDF/PNG with headless Chromium).

Given a corpus, an LLM writes queries from sampled documents; every embedding model retrieves its top 10 and an LLM
judge grades each result 0-5; the models are ranked by nDCG@10 with those grades, which matches MTEB's ranking.
Icons: Microsoft Fluent Emoji (MIT) and Lucide (ISC); logos from Hugging Face."""

import base64
import glob
import json

import numpy as np
from playwright.sync_api import sync_playwright

from common import GENERATOR, JUDGE, TASKS, judged, labels, overall_agreement, ratings, records, runs_folder
from mteb_gym import corpus
from plots import FIGURES

TASK, QID, OTHER_QIDS = "SciFact", "q50", ("q26", "q10")
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


def data_uri(path, mime):
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def emoji(name, size):
    return f'<img src="{data_uri(FIGURES / "icons" / f"{name}.svg", "image/svg+xml")}" width="{size}" height="{size}">'


def line(name, size, color):
    svg = (FIGURES / "icons" / "line" / f"{name}.svg").read_text().split("-->", 1)[-1]
    return svg.replace("currentColor", color).replace('width="24"', f'width="{size}"').replace('height="24"', f'height="{size}"')


def logo(org, size):
    return f'<img src="{data_uri(FIGURES / "logos" / f"{org}.png", "image/png")}" width="{size}" height="{size}" style="border-radius:6px">'


def short(m):
    return m.split("/")[-1]


def title(d, n):
    text = docs[d] if isinstance(docs[d], str) else json.dumps(docs[d])
    return text[:n].rsplit(" ", 1)[0] + "…" if len(text) > n else text


def actor(name, org, x, y):
    return (f'<div class="actor" style="left:{x}px;top:{y}px">{emoji(name, 62)}'
            f'<span class="badge">{logo(org, 30)}</span></div>')


def qcard(q, x, y, r, main=False):
    return (f'<div class="qcard{" main" if main else ""}" style="left:{x}px;top:{y}px;transform:rotate({r}deg)">'
            f'{line("search", 22, NAVY)}<span>{q["text"]}</span></div>')


def rcard(m, d, x, y, r):
    g = grades[QID][d]
    fill = "#1f5bd6" if g >= 3 else "#8fb0ec" if g >= 2 else "#f4d5cf"
    ink = "#fff" if g >= 2 else "#b3261e"
    rows = "".join(f'<i style="width:{w}%"></i>' for w in (80,))
    return (f'<div class="rcard" style="left:{x}px;top:{y}px;transform:rotate({r}deg)"><div class="rh">'
            f'{logo(m.split("/")[0], 26)}<span>{title(d, 44)}</span></div>{rows}'
            f'<b class="grade" style="background:{fill};color:{ink}">{g:.1f}</b></div>')


def dashed(path):
    return f'<path d="{path}" marker-end="url(#h)"/>'


FILES = "".join(f'<div class="file">{line("file-text", 30, "#9fc0f5")}<span>{title(d, 24)}</span></div>'
                for d in [query["seed_doc_ids"][2], *list(docs)[:4]])
BOARD = "".join(f'<div class="row"><span class="k">{i + 1}</span>{logo(m.split("/")[0], 30)}<span class="nm">{short(m)}</span>'
                f'<b>{score[m]:.1f}</b><span class="mt">#{mteb.index(m) + 1}</span></div>' for i, m in enumerate(gym[:5]))
GRAIN = ("url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='160' height='160'>"
         "<filter id='n'><feTurbulence type='fractalNoise' baseFrequency='.85' numOctaves='2' stitchTiles='stitch'/>"
         "<feColorMatrix values='0 0 0 0 1  0 0 0 0 1  0 0 0 0 1  0 0 0 .07 0'/></filter>"
         "<rect width='100%' height='100%' filter='url(%23n)'/></svg>\")")
W, H = 2240, 530
ARROWS = f"""<svg class="lines" width="{W}" height="{H}" viewBox="0 0 {W} {H}" fill="none" stroke="{NAVY}" stroke-width="3"
  stroke-dasharray="9 8" stroke-linecap="round">
  <defs><marker id="h" viewBox="0 0 12 12" refX="9" refY="6" markerWidth="8" markerHeight="8" orient="auto">
    <path d="M2 2L10 6L2 10" fill="none" stroke="{NAVY}" stroke-width="2" stroke-dasharray="none"/></marker></defs>
  {dashed("M418 168 C 440 110, 470 92, 500 92")}{dashed("M440 214 C 470 214, 480 214, 500 214")}{dashed("M418 262 C 440 320, 470 340, 500 340")}
  {dashed("M1022 214 C 1045 214, 1060 214, 1082 214")}
  {dashed("M1124 168 C 1150 110, 1170 96, 1194 96")}{dashed("M1124 262 C 1150 330, 1170 330, 1194 330")}
  {dashed("M1690 162 C 1680 110, 1660 96, 1626 92")}{dashed("M1646 214 L 1634 214")}{dashed("M1690 266 C 1680 320, 1660 332, 1612 332")}
  <path d="M1760 214 L 1796 214" stroke-dasharray="none" stroke-width="4" marker-end="url(#h)"/>
</svg>"""

HTML = f"""<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Nunito:wght@600;700;800;900&display=swap" rel="stylesheet">
<style>
* {{ box-sizing: border-box; margin: 0; }}
body {{ font-family: Nunito, sans-serif; color: {NAVY}; background: {CREAM}; width: {W}px; height: {H}px; position: relative;
        overflow: hidden; }}
.lines {{ position: absolute; left: 0; top: 0; z-index: 1; }}
.corpus {{ position: absolute; left: 30px; top: 40px; width: 380px; height: 350px; border-radius: 12px; background: {NAVY};
           background-image: {GRAIN}; color: #fff; padding: 22px 26px; box-shadow: 0 8px 20px rgba(18,58,122,.18); }}
.corpus h3 {{ display: flex; align-items: center; gap: 14px; font-size: 30px; font-weight: 800; padding-bottom: 14px;
              border-bottom: 1.5px solid rgba(255,255,255,.25); margin-bottom: 12px; }}
.file {{ display: flex; align-items: center; gap: 14px; font-size: 23px; font-weight: 700; padding: 6px 0; white-space: nowrap; }}
.panel {{ position: absolute; top: 24px; height: 382px; border-radius: 18px; background: {PANEL}; background-image: {GRAIN}; }}
.actor {{ position: absolute; z-index: 3; width: 104px; height: 104px; border-radius: 50%; background: #fff;
          border: 3px solid {NAVY}; display: flex; align-items: center; justify-content: center;
          box-shadow: 0 0 0 9px rgba(18,58,122,.10), 0 8px 18px rgba(18,58,122,.2); }}
.badge {{ position: absolute; right: -6px; bottom: -4px; background: #fff; border-radius: 9px; padding: 3px; line-height: 0;
          box-shadow: 0 2px 6px rgba(18,58,122,.3); }}
.qcard {{ position: absolute; z-index: 2; width: 430px; background: #fff; border-radius: 12px; padding: 14px 18px;
          display: flex; gap: 12px; align-items: flex-start; font-size: 22px; font-weight: 800; line-height: 1.3;
          box-shadow: 0 8px 20px rgba(18,58,122,.14); border: 1.5px solid #d3dcea; }}
.qcard svg {{ flex: none; margin-top: 3px; }}
.qcard.main {{ border: 3px solid {NAVY}; }}
.rcard {{ position: absolute; z-index: 2; width: 400px; padding-right: 92px !important; background: #fff; border-radius: 12px; padding: 12px 16px 14px;
          box-shadow: 0 8px 20px rgba(18,58,122,.14); border: 1.5px solid #d3dcea; }}
.rh {{ display: flex; align-items: center; gap: 10px; font-size: 20px; font-weight: 800; margin-bottom: 6px;
       line-height: 1.2; align-items: flex-start; }}
.rcard i {{ display: block; height: 9px; border-radius: 5px; background: #e6ebf3; margin: 7px 0 0 36px; }}
.grade {{ position: absolute; right: 18px; top: 50%; margin-top: -28px; width: 56px; height: 56px; border-radius: 50%; display: flex;
          align-items: center; justify-content: center; font-size: 22px; font-weight: 900;
          box-shadow: 0 0 0 7px rgba(255,255,255,.9), 0 3px 8px rgba(18,58,122,.2); }}
.board {{ position: absolute; left: 1808px; top: 40px; width: 414px; height: 350px; background: #fff; border-radius: 14px;
          border: 1.5px solid #d3dcea; box-shadow: 0 8px 20px rgba(18,58,122,.10); padding: 18px 22px; }}
.board h3 {{ display: flex; align-items: center; gap: 12px; font-size: 28px; font-weight: 900; padding-bottom: 12px;
             border-bottom: 1.5px solid #e3e8f0; margin-bottom: 6px; }}
.row {{ display: flex; align-items: center; gap: 9px; font-size: 19px; font-weight: 800; padding: 8px 0;
        border-bottom: 1px solid #eef1f6; }}
.row .k {{ width: 16px; color: #8a9bbd; }}
.row .nm {{ flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.row b {{ font-weight: 900; color: #1f5bd6; }}
.row .mt {{ width: 42px; text-align: right; color: {ORANGE}; font-weight: 900; }}
.cap {{ position: absolute; top: 432px; text-align: center; font-size: 30px; font-weight: 800; line-height: 1.2; }}
.cap small {{ display: block; font-size: 21px; font-weight: 700; color: #4b6390; margin-top: 4px; }}
.who {{ position: absolute; z-index: 3; font-size: 17px; font-weight: 800; color: {NAVY}; text-align: center; width: 160px;
        line-height: 1.15; }}
</style></head><body>
{ARROWS}
<div class="corpus"><h3>{line("database", 34, "#fff")}{TASK} corpus</h3>{FILES}</div>
{actor("writing_hand_flat_default", "deepseek-ai", 362, 162)}
<div class="panel" style="left:470px;width:560px"></div>
{qcard(queries[OTHER_QIDS[0]], 510, 50, -2)}
{qcard(query, 540, 166, 1, main=True)}
{qcard(queries[OTHER_QIDS[1]], 500, 300, -1)}
{actor("robot_flat", "Qwen", 1030, 162)}
<div class="panel" style="left:1140px;width:560px"></div>
{actor("balance_scale_flat", "autotrust", 1648, 162)}
{rcard(STRONG, top[STRONG][QID][1], 1196, 52, -1.5)}
{rcard(STRONG, top[STRONG][QID][0], 1250, 168, 1.2)}
{rcard(WEAK, top[WEAK][QID][2], 1200, 284, -1)}
<div class="board"><h3>{emoji("trophy_flat", 36)}Model ranking</h3>{BOARD}</div>
<div class="cap" style="left:30px;width:380px">Given a corpus<small>no queries, no relevance labels</small></div>
<div class="cap" style="left:470px;width:560px">an LLM writes realistic queries<small>{GENERATOR}, 100 per task</small></div>
<div class="cap" style="left:1140px;width:560px">each model retrieves, an LLM judges<small>28 models · {JUDGE} grades the top 10, 0–5</small></div>
<div class="cap" style="left:1820px;width:390px">then ranks by nDCG@10<small><b style="color:{ORANGE}">#</b> = MTEB rank · ρ = {RHO:.2f}</small></div>
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
