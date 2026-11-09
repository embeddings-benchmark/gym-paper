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


PAIR_QID = "q30"  # a query with a pairwise verdict for STRONG vs WEAK
pair = next(json.loads(line) for f in glob.glob(f"{folder}/verdicts/{TASK}/{STRONG.replace('/', '_')}__{WEAK.replace('/', '_')}-*.jsonl")
            for line in open(f) if json.loads(line)["query_set"] == config["query_set"] and json.loads(line)["qid"] == PAIR_QID)


def data_uri(path, mime):
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def emoji(name, size):
    return f'<img src="{data_uri(FIGURES / "icons" / f"{name}_3d.png", "image/png")}" width="{size}" height="{size}">'


def line(name, size, color):
    svg = (FIGURES / "icons" / "line" / f"{name}.svg").read_text().split("-->", 1)[-1]
    return svg.replace("currentColor", color).replace('width="24"', f'width="{size}"').replace('height="24"', f'height="{size}"')


def logo(org, size, cls=""):
    return (f'<img class="{cls}" src="{data_uri(FIGURES / "logos" / f"{org}.png", "image/png")}" width="{size}" '
            f'height="{size}" style="border-radius:{size // 4}px">')


def short(m):
    return m.split("/")[-1]


def title(d, n):
    text = docs[d] if isinstance(docs[d], str) else json.dumps(docs[d])
    return text[:n].rsplit(" ", 1)[0] + "…" if len(text) > n else text


def actor(name, org, x, y, label):
    return (f'<div class="actor" style="left:{x}px;top:{y}px">{emoji(name, 70)}<span class="badge">{logo(org, 30)}</span>'
            f'<span class="who">{label}</span></div>')


def qcard(q, x, y, r, main=False):
    return (f'<div class="qcard{" main" if main else ""}" style="left:{x}px;top:{y}px;transform:rotate({r}deg)">'
            f'{emoji("magnifying_glass_tilted_right", 28)}<span>{q["text"]}</span></div>')


def grade_badge(g):
    fill = "#1f5bd6" if g >= 3 else "#8fb0ec" if g >= 2 else "#f4d5cf"
    return f'<b class="grade" style="background:{fill};color:{"#fff" if g >= 2 else "#b3261e"}">{g:.1f}</b>'


def rcard(m, d, x, y, r):
    return (f'<div class="rcard" style="left:{x}px;top:{y}px;transform:rotate({r}deg)">{logo(m.split("/")[0], 26)}'
            f'<span>{title(d, 40)}</span>{grade_badge(grades[QID][d])}</div>')


def side(m, win):
    lines = "".join(f'<i style="width:{w}%"></i>' for w in (90, 72, 84))
    mark = emoji("check_mark_button", 30) if win else ""
    return (f'<div class="side{" win" if win else ""}"><div class="sh">{logo(m.split("/")[0], 24)}<span>{short(m)}</span>{mark}</div>'
            f'<small>top 10</small>{lines}</div>')


def dashed(path):
    return f'<path d="{path}" marker-end="url(#h)"/>'


orgs = ["Qwen", "Snowflake", "microsoft", "google", "nomic-ai"]
CROWD = "".join(logo(o, 38, "crowd") for o in orgs) + f'<span class="more">+{len(models) - len(orgs)}</span>'
FILES = "".join(f'<div class="file">{line("file-text", 28, "#9fc0f5")}<span>{title(d, 24)}</span></div>'
                for d in [query["seed_doc_ids"][2], *list(docs)[:3]])
MEDALS = ["1st_place_medal", "2nd_place_medal", "3rd_place_medal"]
BOARD = "".join(f'<div class="row"><span class="k">{emoji(MEDALS[i], 30) if i < 3 else i + 1}</span>{logo(m.split("/")[0], 28)}'
                f'<span class="nm">{short(m)}</span><b>{score[m]:.1f}</b><span class="mt">#{mteb.index(m) + 1}</span></div>'
                for i, m in enumerate(gym[:5]))
GRAIN = ("url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='160' height='160'>"
         "<filter id='n'><feTurbulence type='fractalNoise' baseFrequency='.85' numOctaves='2' stitchTiles='stitch'/>"
         "<feColorMatrix values='0 0 0 0 1  0 0 0 0 1  0 0 0 0 1  0 0 0 .07 0'/></filter>"
         "<rect width='100%' height='100%' filter='url(%23n)'/></svg>\")")
W, H = 2330, 560
ARROWS = f"""<svg class="lines" width="{W}" height="{H}" viewBox="0 0 {W} {H}" fill="none" stroke="{NAVY}" stroke-width="3"
  stroke-dasharray="9 8" stroke-linecap="round">
  <defs><marker id="h" viewBox="0 0 12 12" refX="9" refY="6" markerWidth="8" markerHeight="8" orient="auto">
    <path d="M2 2L10 6L2 10" fill="none" stroke="{NAVY}" stroke-width="2" stroke-dasharray="none"/></marker></defs>
  {dashed("M418 172 C 440 114, 470 96, 500 96")}{dashed("M444 224 L 500 224")}{dashed("M418 276 C 440 334, 470 352, 500 352")}
  {dashed("M1028 224 L 1080 224")}
  {dashed("M1124 172 C 1150 114, 1166 104, 1190 104")}{dashed("M1138 226 C 1160 250, 1166 280, 1176 300")}
  {dashed("M1752 176 C 1740 120, 1730 104, 1710 104")}{dashed("M1752 276 C 1740 330, 1730 352, 1716 352")}
  <path d="M1836 224 L 1866 224" stroke-dasharray="none" stroke-width="4" marker-end="url(#h)"/>
</svg>"""

HTML = f"""<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Nunito:wght@600;700;800;900&display=swap" rel="stylesheet">
<style>
* {{ box-sizing: border-box; margin: 0; }}
body {{ font-family: Nunito, sans-serif; color: {NAVY}; background: {CREAM}; width: {W}px; height: {H}px; position: relative;
        overflow: hidden; }}
.lines {{ position: absolute; left: 0; top: 0; z-index: 1; }}
.corpus {{ position: absolute; left: 30px; top: 44px; width: 380px; height: 360px; border-radius: 12px; background: {NAVY};
           background-image: {GRAIN}; color: #fff; padding: 18px 26px; box-shadow: 0 8px 20px rgba(18,58,122,.18); }}
.corpus h3 {{ display: flex; align-items: center; gap: 14px; font-size: 30px; font-weight: 800; padding-bottom: 12px;
              border-bottom: 1.5px solid rgba(255,255,255,.25); margin-bottom: 10px; }}
.file {{ display: flex; align-items: center; gap: 14px; font-size: 23px; font-weight: 700; padding: 6px 0; white-space: nowrap; }}
.panel {{ position: absolute; top: 28px; height: 392px; border-radius: 18px; background: {PANEL}; background-image: {GRAIN}; }}
.actor {{ position: absolute; z-index: 3; width: 112px; height: 112px; border-radius: 50%; background: #fff;
          border: 3px solid {NAVY}; display: flex; align-items: center; justify-content: center;
          box-shadow: 0 0 0 9px rgba(18,58,122,.10), 0 8px 18px rgba(18,58,122,.2); }}
.badge {{ position: absolute; right: -6px; bottom: -2px; background: #fff; border-radius: 10px; padding: 3px; line-height: 0;
          box-shadow: 0 2px 6px rgba(18,58,122,.3); }}
.who {{ position: absolute; top: -42px; left: 50%; transform: translateX(-50%); white-space: nowrap; font-size: 18px;
        font-weight: 900; background: #fff; color: {NAVY}; border: 2.5px solid {NAVY}; padding: 2px 12px; border-radius: 999px; }}
.qcard {{ position: absolute; z-index: 2; width: 440px; background: #fff; border-radius: 12px; padding: 14px 18px;
          display: flex; gap: 12px; align-items: flex-start; font-size: 22px; font-weight: 800; line-height: 1.3;
          box-shadow: 0 8px 20px rgba(18,58,122,.14); border: 1.5px solid #d3dcea; }}
.qcard img {{ flex: none; margin-top: 1px; }}
.qcard.main {{ border: 3px solid {NAVY}; }}
.mode {{ position: absolute; z-index: 2; font-size: 15px; font-weight: 900; letter-spacing: 1px; color: #4b6390; }}
.mode b {{ color: {NAVY}; }}
.rcard {{ position: absolute; z-index: 2; width: 268px; height: 100px; background: #fff; border-radius: 12px; padding: 12px 36px 12px 14px;
          box-shadow: 0 8px 20px rgba(18,58,122,.14); border: 1.5px solid #d3dcea; display: flex; gap: 9px;
          font-size: 18px; font-weight: 800; line-height: 1.22; }}
.rcard img {{ flex: none; }}
.grade {{ position: absolute; right: -14px; top: -16px; width: 52px; height: 52px; border-radius: 50%; display: flex;
          align-items: center; justify-content: center; font-size: 20px; font-weight: 900;
          box-shadow: 0 0 0 5px #fff, 0 3px 8px rgba(18,58,122,.25); }}
.pair {{ position: absolute; z-index: 2; left: 1180px; top: 276px; width: 540px; background: #fff; border-radius: 12px;
         padding: 12px 14px; box-shadow: 0 8px 20px rgba(18,58,122,.14); border: 1.5px solid #d3dcea;
         display: flex; align-items: center; gap: 10px; transform: rotate(-.8deg); }}
.side {{ flex: 1; border: 1.5px solid #e1e7f0; border-radius: 10px; padding: 8px 10px; min-width: 0; }}
.side.win {{ border: 2.5px solid #1f5bd6; background: #f3f7ff; }}
.sh {{ display: flex; align-items: center; gap: 7px; font-size: 16.5px; font-weight: 900; white-space: nowrap; }}
.sh span {{ flex: 1; overflow: hidden; text-overflow: ellipsis; }}
.side small {{ display: block; font-size: 13px; font-weight: 800; color: #8a9bbd; margin: 2px 0 1px 31px; }}
.side i {{ display: block; height: 7px; border-radius: 4px; background: #e6ebf3; margin: 6px 0 0 31px; }}
.vs {{ flex: none; text-align: center; font-size: 26px; font-weight: 900; line-height: 1; }}
.vs small {{ display: block; font-size: 15px; font-weight: 800; color: #1f5bd6; margin-top: 4px; }}
.crowd {{ margin-left: -12px; border: 2.5px solid #fff; background: #fff; border-radius: 50% !important; object-fit: contain; padding: 4px; box-shadow: 0 2px 6px rgba(18,58,122,.2); }}
.models {{ position: absolute; left: 1000px; top: 300px; display: flex; align-items: center; padding-left: 10px; z-index: 3; }}
.more {{ margin-left: 6px; font-size: 18px; font-weight: 900; background: {CREAM}; border-radius: 8px; padding: 0 4px; }}
.board {{ position: absolute; left: 1880px; top: 44px; width: 430px; height: 360px; background: #fff; border-radius: 14px;
          border: 1.5px solid #d3dcea; box-shadow: 0 8px 20px rgba(18,58,122,.10); padding: 14px 20px; }}
.board h3 {{ display: flex; align-items: center; gap: 10px; font-size: 26px; font-weight: 900; padding-bottom: 10px;
             border-bottom: 1.5px solid #e3e8f0; margin-bottom: 4px; }}
.board h3 small {{ margin-left: auto; font-size: 14px; color: {ORANGE}; font-weight: 900; letter-spacing: .5px; }}
.row {{ display: flex; align-items: center; gap: 9px; font-size: 19px; font-weight: 800; padding: 7px 0;
        border-bottom: 1px solid #eef1f6; height: 52px; }}
.row .k {{ width: 30px; text-align: center; color: #8a9bbd; line-height: 0; }}
.row .k:not(:has(img)) {{ line-height: 1; }}
.row .nm {{ flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.row b {{ font-weight: 900; color: #1f5bd6; }}
.row .mt {{ width: 38px; text-align: right; color: {ORANGE}; font-weight: 900; }}
.cap {{ position: absolute; top: 452px; text-align: center; font-size: 30px; font-weight: 800; line-height: 1.2; }}
.cap small {{ display: block; font-size: 21px; font-weight: 700; color: #4b6390; margin-top: 4px; }}
</style></head><body>
{ARROWS}
<div class="corpus"><h3>{emoji("open_file_folder", 40)}{TASK} corpus</h3>{FILES}<div class="file" style="opacity:.7;font-size:19px">… {len(docs):,} documents</div></div>
{actor("writing_hand", "deepseek-ai", 358, 168, "Query writer").replace('class="who"', 'class="who" style="top:126px"')}
<div class="panel" style="left:470px;width:560px"></div>
{qcard(queries[OTHER_QIDS[0]], 510, 54, -2)}
{qcard(query, 540, 176, 1, main=True)}
{qcard(queries[OTHER_QIDS[1]], 500, 312, -1)}
{actor("robot", "Qwen", 1026, 168, "Embedding models")}
<div class="models">{CROWD}</div>
<div class="panel" style="left:1140px;width:620px"></div>
<div class="mode" style="left:1166px;top:42px"><b>POINTWISE</b> · grade each document 0–5</div>
{rcard(STRONG, top[STRONG][QID][1], 1180, 80, -1.5)}
{rcard(WEAK, top[WEAK][QID][2], 1468, 92, 1.2)}
<div class="mode" style="left:1166px;top:240px"><b>PAIRWISE</b> · which top 10 is better?</div>
<div class="pair">{side(STRONG, pair["score_a"] > .5)}<div class="vs">vs<small>p = {pair["score_a"]:.2f}</small></div>{side(WEAK, pair["score_a"] < .5)}</div>
{actor("balance_scale", "autotrust", 1724, 168, "Judge")}
<div class="board"><h3>{emoji("person_lifting_weights", 40)}MTEB Gym<small>MTEB</small></h3>{BOARD}</div>
<div class="cap" style="left:30px;width:380px">Given a corpus<small>no queries, no relevance labels</small></div>
<div class="cap" style="left:470px;width:560px">an LLM writes realistic queries<small>{GENERATOR}, 100 per task</small></div>
<div class="cap" style="left:1140px;width:620px">each model retrieves, an LLM judges<small>28 models · {JUDGE}, pointwise or pairwise</small></div>
<div class="cap" style="left:1880px;width:430px">then ranks the models<small>same top 3 as MTEB · ρ = {RHO:.2f}</small></div>
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
