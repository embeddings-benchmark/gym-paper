"""Figure 1: MTEB Gym on one real SciFact query, drawn as one picture (HTML printed to PDF/PNG with headless Chromium).

The query writer reads documents sampled from the corpus and writes a query; every embedding model retrieves its top 10;
the judge grades each retrieved document 0-5; nDCG@10 with those grades, averaged over queries and tasks, ranks the
models, and the ranking matches MTEB's. Everything shown comes from the published runs. Logos from Hugging Face."""

import base64
import glob
import json

import numpy as np
from playwright.sync_api import sync_playwright

from common import GENERATOR, JUDGE, TASKS, judged, labels, ndcg, overall_agreement, ratings, records, runs_folder
from mteb_gym import corpus
from plots import FIGURES

TASK, QID = "SciFact", "q50"
STRONG, WEAK = "Qwen/Qwen3-Embedding-8B", "sentence-transformers/all-MiniLM-L6-v2"
HUMAN_GYM, HUMAN_MTEB = 0.857, 0.762  # plot_humans.py
INK, BLUE, ORANGE, GRAY, LIGHT, PALE = "#1a2233", "#154aa8", "#ff5a0a", "#5b6880", "#dfe2e8", "#eef2fa"

R = records()
recs = R[(JUDGE, "pointwise", GENERATOR)]
per_task = {t: ratings(recs[t]) for t in TASKS}
models = sorted(per_task[TASKS[0]])
gym = sorted(models, key=lambda m: -np.mean([per_task[t][m] for t in TASKS]))
mteb = sorted(models, key=lambda m: -np.mean([labels(t)[m] for t in TASKS]))
RHO = overall_agreement(per_task)

config = recs[TASK]["config"]
_, _, top, grades, _ = judged(recs[TASK])
docs = corpus.load(TASK).docs
folder = runs_folder(f"*/{TASK}/*")
query = next(q for f in glob.glob(f"{folder}/queries/{TASK}/*") for q in json.load(open(f))["queries"]
             if config["query_set"].startswith(f.rsplit("/", 1)[1][:-5]) and q["qid"] == QID)
probs = {}
for f in glob.glob(f"{folder}/grades/{TASK}/{config['judge_model'].replace('/', '_')}-*.jsonl"):
    for line in open(f):
        row = json.loads(line)
        if row["query_set"] != config["query_set"]:
            break
        if row["qid"] == QID:
            probs[row["doc_id"]] = [row["probabilities"][str(g)] for g in range(6)]
nd = {m: 100 * ndcg(top[m][QID], grades[QID], grades[QID].values(), 10) for m in (STRONG, WEAK)}
SEED = next(d for d in query["seed_doc_ids"] if d in top[STRONG][QID][:3])  # the sampled document the query is about


def logo(org, size):
    data = base64.b64encode((FIGURES / "logos" / f"{org}.png").read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}" width="{size}" height="{size}">'


def emoji(name, size):
    data = base64.b64encode((FIGURES / "icons" / f"{name}.svg").read_bytes()).decode()
    return f'<img src="data:image/svg+xml;base64,{data}" width="{size}" height="{size}">'


def actor(name, org):
    """A role drawn as an icon, with the model that plays it as a small logo badge."""
    return f'{emoji(name, 54)}<span class="org">{logo(org, 30)}</span>'


def step(n, text):
    return f'<span class="num">{n}</span>{text}'


def short(m):
    return m.split("/")[-1]


def title(d, n=60):
    text = docs[d] if isinstance(docs[d], str) else json.dumps(docs[d])
    return text[:n].rsplit(" ", 1)[0] + " …" if len(text) > n else text


def shade(g):
    return ["#eef1f6", "#dde4f2", "#bccbeb", "#86a1d9", "#4a72c4", BLUE][int(round(g))]


def at(x, y, html, cls="", style=""):
    return f'<div class="{cls}" style="left:{x}px;top:{y}px;{style}">{html}</div>'


def sheet(x, y, d=None, r=0):
    text = f'<b>{title(d, 46)}</b>' if d else ""
    lines = "".join(f'<i style="width:{w}%"></i>' for w in (92, 100, 84, 96, 70, 88, 60))
    return at(x, y, text + lines, "sheet", f"transform:rotate({r}deg)")


def card(x, y, d, rank, faded=False):
    g = grades[QID][d]
    badge = f'<span class="badge" style="background:{shade(g)};color:{"#fff" if g >= 2.5 else INK}">{g:.1f}</span>'
    cls = "card" + (" seed" if d == SEED else "") + (" faded" if faded else "")
    return at(x, y, f'<span class="rk">{rank}</span><span class="ct">{title(d, 52)}</span>{badge}', cls)


def results(m, y, faded=False):
    cards = "".join(card(560 + i * 168, y, d, i + 1, faded) for i, d in enumerate(top[m][QID][:3]))
    org = m.split("/")[0]
    return (at(470, y + 4, logo(org, 44), "node small") + at(560, y - 30, short(m), "mname") + cards
            + at(1064, y + 22, "…", "dots"))


def podium(x, y, name, sub, order):
    heights = {0: 92, 1: 66, 2: 46}
    blocks = ""
    for place in (1, 0, 2):
        h, m = heights[place], order[place]
        blocks += (f'<div class="col"><div class="pl">{logo(m.split("/")[0], 38)}</div>'
                   f'<div class="blk" style="height:{h}px">{place + 1}</div></div>')
    names = "".join(f'<div><b>{i + 1}</b> {short(m)}</div>' for i, m in enumerate(order[:3]))
    return at(x, y, f'<div class="ph">{name}<span>{sub}</span></div><div class="pod">{blocks}</div><div class="pn">{names}</div>',
              "podium")


def bars(x, y, p):
    cols = "".join(f'<div class="bc"><div style="height:{2 + 48 * v / max(p)}px;background:{BLUE if v == max(p) else "#9fb4df"}"></div>'
                   f'<span>{g}</span></div>' for g, v in enumerate(p))
    return at(x, y, f'<div class="bars">{cols}</div>', "dist")


W, H, TOP = 1930, 690, 40
ARROWS = f"""<svg class="arrows" width="{W}" height="{H}" viewBox="0 0 {W} {H}" fill="none" stroke="{INK}" stroke-width="2.4"
  stroke-linecap="round" stroke-linejoin="round">
  <defs><marker id="a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M1 1L9 5L1 9" fill="none" stroke="{INK}" stroke-width="1.8"/></marker>
  <marker id="b" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M1 1L9 5L1 9" fill="none" stroke="{BLUE}" stroke-width="1.8"/></marker></defs>
  <path d="M160 238 C 160 150, 200 112, 262 108" marker-end="url(#a)"/>
  <path d="M215 330 C 330 330, 360 300, 456 292" marker-end="url(#a)"/>
  <path d="M600 182 L 600 222" marker-end="url(#a)"/>
  <path d="M902 108 L 1000 108" marker-end="url(#a)"/>
  <g stroke="{BLUE}" stroke-dasharray="2 7" stroke-width="2.6">
    <path d="M1042 150 C 1030 200, 1000 220, 1010 254" marker-end="url(#b)"/>
  </g>
  <path d="M1096 292 L 1150 292" marker-end="url(#a)"/>
  <path d="M1096 422 L 1150 422" marker-end="url(#a)"/>
  <path d="M1262 300 L 1392 300" marker-end="url(#a)"/>
</svg>"""

HTML = f"""<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,400;0,500;0,600;0,700;0,800;1,500;1,600&display=swap" rel="stylesheet">
<style>
* {{ box-sizing: border-box; margin: 0; }}
body {{ font-family: Inter, sans-serif; color: {INK}; background: #fff; width: {W}px; height: {H}px; position: relative;
        overflow: hidden; }}
.canvas {{ position: absolute; left: 0; top: {TOP}px; width: {W}px; height: {H - TOP}px; }}
.canvas > div, body > div.st {{ position: absolute; }}
.st {{ display: flex; align-items: center; gap: 10px; font-size: 24px; font-weight: 800; letter-spacing: -.3px; white-space: nowrap; }}
.num {{ width: 34px; height: 34px; border-radius: 50%; background: {INK}; color: #fff; font-size: 18px; display: flex; align-items: center; justify-content: center; }}
.org {{ position: absolute; right: -8px; bottom: -6px; background: #fff; border-radius: 9px; padding: 2px; box-shadow: 0 2px 6px rgba(26,34,51,.25); line-height: 0; }}
.org img {{ border-radius: 7px; }}
.ic {{ display: flex; align-items: center; gap: 10px; }}
.arrows {{ position: absolute; left: 0; top: 0; }}
.sheet {{ width: 168px; height: 206px; background: #fff; border: 1.5px solid #c3c9d4; border-radius: 6px; padding: 14px;
          box-shadow: 0 4px 10px rgba(26,34,51,.07); display: flex; flex-direction: column; gap: 9px; }}
.sheet.seed {{ border: 2.5px solid {BLUE}; }}
.sheet b {{ font-size: 13.5px; line-height: 1.25; font-weight: 700; color: #39445a; }}
.sheet i {{ display: block; height: 6px; border-radius: 3px; background: #e4e8ee; }}
.lbl {{ font-size: 21px; font-weight: 800; }}
.lbl span {{ display: block; font-size: 16px; font-weight: 500; color: {GRAY}; margin-top: 3px; }}
.note {{ font-size: 16px; font-style: italic; font-weight: 600; color: #39445a; line-height: 1.3; }}
.task {{ width: 250px; font-size: 16px; font-weight: 600; line-height: 1.35; border-left: 3px solid {INK}; padding-left: 10px; }}
.task small {{ display: block; font-size: 12.5px; font-weight: 700; color: {GRAY}; letter-spacing: .6px; }}
.node {{ width: 84px; height: 84px; border-radius: 50%; background: #fff; border: 2.5px solid {INK};
         display: flex; align-items: center; justify-content: center; box-shadow: 0 6px 16px rgba(26,34,51,.12); }}
.node img {{ border-radius: 8px; }}
.node.small {{ width: 72px; height: 72px; border-width: 2px; }}
.bubble {{ width: 520px; border: 2.5px solid {BLUE}; background: {PALE}; border-radius: 18px; padding: 12px 18px 14px;
           font-size: 22px; font-weight: 700; line-height: 1.32; }}
.bubble small {{ display: block; font-size: 12.5px; font-weight: 800; color: {BLUE}; letter-spacing: .7px; margin-bottom: 3px; }}
.bubble:before {{ content: ""; position: absolute; left: -16px; top: 46px; border: 9px solid transparent;
                  border-right: 14px solid {BLUE}; }}
.mname {{ font-size: 17px; font-weight: 800; }}
.card {{ width: 154px; height: 92px; background: #fff; border: 1.5px solid #c3c9d4; border-radius: 9px; padding: 8px 10px;
         box-shadow: 0 4px 10px rgba(26,34,51,.07); display: flex; gap: 6px; }}
.card.seed {{ border: 2.5px solid {BLUE}; }}
.card.faded {{ opacity: .35; }}
.rk {{ font-size: 14px; font-weight: 800; color: {GRAY}; }}
.ct {{ font-size: 13.5px; line-height: 1.3; color: #39445a; font-weight: 500; overflow: hidden; }}
.badge {{ position: absolute; right: -13px; top: -13px; width: 40px; height: 40px; border-radius: 50%;
          display: flex; align-items: center; justify-content: center; font-size: 14.5px; font-weight: 800;
          border: 2.5px solid #fff; box-shadow: 0 2px 6px rgba(26,34,51,.18); font-variant-numeric: tabular-nums; }}
.dots {{ font-size: 26px; color: {GRAY}; font-weight: 700; }}
.nd {{ text-align: center; }}
.nd small {{ display: block; font-size: 13px; font-weight: 700; color: {GRAY}; }}
.nd b {{ font-size: 34px; font-weight: 800; font-variant-numeric: tabular-nums; }}
.dist .bars {{ display: flex; align-items: flex-end; gap: 5px; width: 132px; height: 52px; border-bottom: 1.5px solid {INK}; }}
.bc {{ flex: 1; position: relative; display: flex; flex-direction: column; justify-content: flex-end; height: 100%; }}
.bc div {{ border-radius: 3px 3px 0 0; }}
.bc span {{ position: absolute; bottom: -20px; width: 100%; text-align: center; font-size: 12.5px; font-weight: 700; color: {GRAY}; }}
.podium {{ width: 206px; }}
.ph {{ font-size: 19px; font-weight: 800; text-align: center; }}
.ph span {{ display: block; font-size: 14px; font-weight: 500; color: {GRAY}; }}
.pod {{ display: flex; align-items: flex-end; justify-content: center; gap: 4px; margin-top: 8px; }}
.col {{ display: flex; flex-direction: column; align-items: center; gap: 5px; width: 58px; }}
.pl img {{ border-radius: 50%; border: 2px solid #fff; box-shadow: 0 2px 6px rgba(26,34,51,.2); background: #fff; }}
.blk {{ width: 100%; background: {BLUE}; color: #fff; font-size: 20px; font-weight: 800; display: flex; justify-content: center;
        padding-top: 6px; border-radius: 4px 4px 0 0; }}
.mteb .blk {{ background: {ORANGE}; }}
.pn {{ margin-top: 8px; font-size: 14px; font-weight: 600; line-height: 1.45; border-top: 1.5px solid {INK}; padding-top: 6px; }}
.pn b {{ color: {GRAY}; display: inline-block; width: 14px; }}
.rho {{ text-align: center; font-size: 15px; font-weight: 700; color: {GRAY}; }}
.rho b {{ display: block; font-size: 27px; color: {INK}; font-weight: 800; white-space: nowrap; }}
.human {{ font-size: 15px; font-weight: 600; color: #39445a; text-align: center; width: 400px; line-height: 1.35; }}
.human b {{ color: {BLUE}; }} .human em {{ font-style: normal; font-weight: 700; color: {ORANGE}; }}
</style></head><body>
{at(196, 14, step(1, "Write queries"), "st")}{at(560, 14, step(2, "Retrieve"), "st")}{at(1040, 14, step(3, "Judge"), "st")}{at(1460, 14, step(4, "Rank the models"), "st")}
<div class="canvas">
{ARROWS}
{sheet(46, 268, r=-5)}{sheet(36, 256, r=-1.5)}{sheet(28, 244, SEED, 0).replace('class="sheet"', 'class="sheet seed"')}
{at(28, 464, f'<div class="ic">{emoji("card_file_box_flat", 32)}Your corpus</div><span>{TASK}: {len(docs):,} documents,<br>no queries, no labels</span>', "lbl")}
{at(28, 556, f'<small>TASK, IN ONE SENTENCE</small>“{config["task_instruction"]}”', "task")}
{at(30, 150, "reads 3 sampled<br>documents", "note", "width:120px")}
{at(264, 66, actor("writing_hand_flat_default", "deepseek-ai"), "node")}
{at(232, 158, f'Query writer<span>{GENERATOR}</span>', "lbl", "text-align:center;width:200px;left:206px")}
{at(380, 34, f'<small>GENERATED QUERY · 1 OF 100 PER TASK</small>{query["text"]}', "bubble")}
{at(612, 188, "every model retrieves its top 10", "note")}
{at(318, 380, emoji("robot_flat", 46), "")}{at(300, 432, "28 embedding<br>models, run<br>as in mteb", "note", "width:150px")}
{results(STRONG, 262)}{results(WEAK, 392)}
{at(470, 534, "", "node small", "opacity:.35")}{at(484, 552, "+26", "mname", "opacity:.6;font-size:19px")}
{"".join(at(560 + i * 168, 522, '<i></i>', "card faded", "height:60px") for i in range(3))}
{at(1000, 66, actor("balance_scale_flat", "autotrust"), "node")}
{at(1100, 72, f'Judge<span>{JUDGE}, grades each<br>query–document pair 0–5</span>', "lbl")}{at(816, 206, "grades every<br>retrieved document", "note", f"color:{BLUE};width:150px;text-align:right")}
{bars(1104, 158, probs[SEED])}
{at(1250, 158, f'<small>P(grade), for the<br>outlined document</small><b style="color:{BLUE};font-size:26px">= {grades[QID][SEED]:.1f}</b>', "nd", "text-align:left")}
{at(1146, 270, f'<small>nDCG@10</small><b>{nd[STRONG]:.0f}</b>', "nd", "width:100px")}
{at(1146, 400, f'<small>nDCG@10</small><b>{nd[WEAK]:.0f}</b>', "nd", "width:100px")}
{at(1150, 470, "mean over 100 queries,<br>then over 28 tasks", "note", "width:200px")}
{podium(1400, 40, "MTEB Gym", "no labels", gym).replace('<div class="ph">', '<div class="ph">' + emoji("trophy_flat", 34) + "<br>")}
{at(1604, 200, f'<b>ρ = {RHO:.2f}</b>same ranking<br>of 28 models', "rho", "width:112px")}
{podium(1712, 40, "MTEB", "human relevance labels", mteb).replace('<div class="ph">', '<div class="ph">' + emoji("busts_in_silhouette_flat", 34) + "<br>").replace('class="podium"', 'class="podium mteb"')}
{at(1440, 400, f'{emoji("bust_in_silhouette_flat", 22)} Agreement with human votes in MTEB Arena:<br>gym <b>{HUMAN_GYM:.2f}</b>, <em>MTEB labels {HUMAN_MTEB:.2f}</em>', "human")}
</div></body></html>"""

(FIGURES / "figure1.html").write_text(HTML)
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": W, "height": H}, device_scale_factor=2)
    page.set_content(HTML, wait_until="networkidle")
    page.pdf(path=str(FIGURES / "figure1.pdf"), width=f"{W}px", height=f"{H}px", print_background=True, page_ranges="1")
    page.screenshot(path=str(FIGURES / "figure1.png"))
    browser.close()
print("wrote", FIGURES / "figure1.pdf")
