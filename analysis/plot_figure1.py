"""Figure 1: the MTEB Gym pipeline on one real SciFact query (HTML printed to PDF/PNG with headless Chromium).

Left to right: the input (a corpus and one sentence describing the task), (1) the query writer reads sampled documents
and writes a query, (2) every embedding model retrieves its top 10, (3) the judge grades each retrieved document 0-5,
(4) nDCG@10 with those grades ranks the models. Below: how the gym's ranking compares with MTEB's.
Everything shown is taken from the published runs. Icons: Lucide (ISC); logos from Hugging Face."""

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
SMALL_JUDGE = 0.957  # plot_small_models.py: decider-4b
INK, BLUE, ORANGE, GRAY, LIGHT, PALE = "#1a2233", "#154aa8", "#ff5a0a", "#5b6880", "#dfe2e8", "#eef2fa"

R = records()
recs = R[(JUDGE, "pointwise", GENERATOR)]
per_task = {t: ratings(recs[t]) for t in TASKS}
models = sorted(per_task[TASKS[0]])
gym_score = {m: np.mean([per_task[t][m] for t in TASKS]) for m in models}
gym = sorted(models, key=lambda m: -gym_score[m])
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
task_board = sorted(ratings(recs[TASK]).items(), key=lambda x: -x[1])
task_place = {m: i + 1 for i, (m, _) in enumerate(task_board)}
SEED = next(d for d in query["seed_doc_ids"] if d in top[STRONG][QID][:3])  # the sampled document the query is about


def b64(path, mime):
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def icon(name, size=30, color=BLUE, width=1.8):
    svg = (FIGURES / "icons" / "line" / f"{name}.svg").read_text().split("-->", 1)[-1]
    svg = svg.replace("currentColor", color).replace('stroke-width="2"', f'stroke-width="{width}"')
    return svg.replace('width="24"', f'width="{size}"').replace('height="24"', f'height="{size}"')


def logo(org, size=24):
    return f'<img src="{b64(FIGURES / "logos" / f"{org}.png", "image/png")}" width="{size}" height="{size}" style="border-radius:5px">'


def short(m):
    return m.split("/")[-1]


def title(d, n=40):
    text = docs[d] if isinstance(docs[d], str) else json.dumps(docs[d])
    return text[:n].rsplit(" ", 1)[0] + " …" if len(text) > n else text


def chip(g):
    """A grade 0-5, darker blue for higher grades."""
    shade = ["#f4f6fa", "#e3e9f5", "#c4d2ee", "#8ea8dc", "#4a72c4", BLUE][int(round(g))]
    fg = "#fff" if g >= 2.5 else INK
    return f'<span class="chip" style="background:{shade};color:{fg}">{g:.1f}</span>'


def doc_line(d, rank=None, grade=None, mark=False):
    k = f'<span class="k">{rank}</span>' if rank else icon("file-text", 22, GRAY, 1.6)
    g = chip(grade) if grade is not None else ""
    return f'<div class="doc{" mark" if mark else ""}">{k}<span class="dt">{title(d)}</span>{g}</div>'


def ranked(m, grade=False):
    rows = "".join(doc_line(d, i + 1, grades[QID][d] if grade else None, d == SEED) for i, d in enumerate(top[m][QID][:3]))
    return f'<div class="list"><div class="lh">{logo(m.split("/")[0], 22)}{short(m)}</div>{rows}<div class="more">… 7 more</div></div>'


def bars(p):
    cols = "".join(f'<div class="bc"><div class="bar" style="height:{3 + 92 * v / max(p)}px;background:{BLUE if v == max(p) else "#9fb4df"}"></div>'
                   f'<span>{g}</span></div>' for g, v in enumerate(p))
    return f'<div class="bars">{cols}</div>'


def panel(n, verb, what, actor, body, cls=""):
    head = f'<span class="n">{n}</span>' if n else icon("database", 28, INK, 1.8)
    return (f'<div class="panel {cls}"><div class="ph">{head}<div><b>{verb}</b><span>{what}</span></div></div>'
            f'<div class="actor">{actor}</div><div class="pb">{body}</div></div>')


ARROW = (f'<div class="arrow"><svg width="30" height="30" viewBox="0 0 30 30"><path d="M3 15h22M18 8l7 7-7 7" '
         f'stroke="{INK}" stroke-width="2.6" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg></div>')
LOGOS = "".join(logo(o, 22) for o in ["Qwen", "BAAI", "Snowflake", "nomic-ai"])
board = "".join(f'<div class="row"><span class="k">{i + 1}</span><span class="nm">{short(m)}</span>'
                f'<b>{gym_score[m]:.1f}</b><span class="mt">{mteb.index(m) + 1}</span></div>' for i, m in enumerate(gym[:5]))

inputs = panel(None, "Your corpus", "and the task in one sentence", f'<span class="none">no queries, no labels</span>', f"""
  <div class="stack">{"".join(doc_line(d) for d in list(docs)[:4])}<div class="more">… {len(docs):,} documents</div></div>
  <div class="task"><small>Task</small>“{config["task_instruction"]}”</div>""", "input")

write = panel(1, "Write queries", "from sampled documents", f'{logo("deepseek-ai")}<span>Query writer · {GENERATOR}</span>', f"""
  <small class="cap">reads 3 sampled documents</small>
  <div class="stack">{"".join(doc_line(d, mark=d == SEED) for d in query["seed_doc_ids"])}</div>
  <div class="down">↓</div>
  <div class="query"><small>Generated query</small>{query["text"]}</div>
  <div class="foot">100 queries per task</div>""")

retrieve = panel(2, "Retrieve", "the top 10 from each model", f'<span class="logos">{LOGOS}</span><span>28 embedding models</span>', f"""
  {ranked(STRONG)}{ranked(WEAK)}
  <div class="foot">models run as in mteb</div>""")

judge = panel(3, "Judge", "each retrieved document, 0–5", f'{logo("autotrust")}<span>Judge · {JUDGE}</span>', f"""
  <div class="pair"><div class="pq">{icon("search", 20, GRAY, 2)}<span>{query["text"]}</span></div>
    <div class="pq">{icon("file-text", 20, GRAY, 1.8)}<span>{title(SEED, 62)}</span></div></div>
  <small class="cap" style="margin-top:4px">judge's probability per grade</small><div class="dist">{bars(probs[SEED])}<div class="ev"><small>grade =<br>expected value</small>{chip(grades[QID][SEED])}</div></div>
  <div class="grid">
    <div class="gl">{logo("Qwen", 20)}{"".join(chip(grades[QID][d]) for d in top[STRONG][QID][:3])}<i>…</i></div>
    <div class="gl">{logo("sentence-transformers", 20)}{"".join(chip(grades[QID][d]) for d in top[WEAK][QID][:3])}<i>…</i></div>
  </div>""")

rank = panel(4, "Rank", "by nDCG@10 with the grades", icon("trophy", 24, BLUE, 1.8) + "<span>no labels used</span>", f"""
  <small class="cap">this query, nDCG@10</small>
  <div class="nd"><span>{short(STRONG)}</span><div class="track"><div style="width:{nd[STRONG]:.0f}%"></div></div><b>{nd[STRONG]:.0f}</b></div>
  <div class="nd"><span>{short(WEAK)}</span><div class="track"><div style="width:{nd[WEAK]:.0f}%"></div></div><b>{nd[WEAK]:.0f}</b></div>
  <div class="down">↓ <small>mean over 100 queries, then 28 tasks</small></div>
  <div class="board"><div class="bh"><span>Gym ranking</span><span>score</span><span class="mt">MTEB</span></div>{board}
  <div class="more">… 23 more</div></div>""", "out")

HTML = f"""<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
* {{ box-sizing: border-box; margin: 0; }}
body {{ font-family: Inter, sans-serif; color: {INK}; background: #fff; display: inline-block; padding: 16px; width: 1760px;
        font-size: 17px; }}
.flow {{ display: flex; align-items: stretch; }}
.arrow {{ display: flex; align-items: center; justify-content: center; width: 34px; flex: none; }}
.panel {{ flex: 1; border: 1.5px solid #c9ced8; border-radius: 12px; background: #fff; display: flex; flex-direction: column;
          min-width: 0; }}
.panel.input {{ flex: .74; background: #f6f7f9; }}
.panel.out {{ flex: 1.14; }}
.ph {{ display: flex; align-items: center; gap: 11px; padding: 12px 14px 8px; }}
.ph b {{ display: block; font-size: 21px; font-weight: 800; letter-spacing: -.2px; }}
.ph div > span {{ display: block; font-size: 15px; font-weight: 500; color: {GRAY}; margin-top: 1px; }}
.n {{ flex: none; width: 30px; height: 30px; border-radius: 50%; background: {BLUE}; color: #fff; font-weight: 800;
      font-size: 17px; display: flex !important; align-items: center; justify-content: center; margin: 0 !important; }}
.actor {{ display: flex; align-items: center; gap: 8px; margin: 0 14px; padding: 7px 10px; border-radius: 8px;
          background: {PALE}; font-size: 15px; font-weight: 700; color: {BLUE}; white-space: nowrap; overflow: hidden; }}
.input .actor {{ background: #eceef2; color: {GRAY}; }}
.logos {{ display: flex; gap: 3px; }}
.pb {{ flex: 1; padding: 10px 14px 12px; display: flex; flex-direction: column; gap: 7px; }}
.cap {{ font-size: 13px; font-weight: 700; color: {GRAY}; text-transform: uppercase; letter-spacing: .5px; }}
.stack {{ display: flex; flex-direction: column; gap: 4px; }}
.doc {{ display: flex; align-items: center; gap: 8px; padding: 5px 8px; border: 1px solid {LIGHT}; border-radius: 7px;
        background: #fff; font-size: 15px; color: #39445a; min-width: 0; }}
.doc.mark {{ border-color: {BLUE}; box-shadow: 0 0 0 1px {BLUE}; }}
.doc svg {{ flex: none; }}
.dt {{ flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.k {{ flex: none; width: 18px; text-align: right; font-weight: 800; color: {GRAY}; font-size: 14px; }}
.more {{ font-size: 13.5px; color: {GRAY}; padding-left: 8px; font-weight: 500; }}
.task {{ margin-top: 8px; border-left: 3px solid {INK}; padding: 4px 0 4px 10px; font-size: 16px; font-weight: 600;
         line-height: 1.35; }}
.task small, .query small {{ display: block; font-size: 13px; font-weight: 700; color: {GRAY}; text-transform: uppercase;
                            letter-spacing: .5px; margin-bottom: 2px; }}
.down {{ text-align: center; font-size: 20px; color: {INK}; line-height: 1; }}
.down small {{ font-size: 13.5px; color: {GRAY}; font-weight: 600; }}
.query {{ border: 1.5px solid {BLUE}; background: {PALE}; border-radius: 9px; padding: 8px 11px; font-size: 16.5px;
          font-weight: 700; line-height: 1.35; }}
.query small {{ color: {BLUE}; }}
.foot {{ margin-top: auto; font-size: 13.5px; color: {GRAY}; font-weight: 600; text-align: center; padding-top: 4px;
         border-top: 1px dashed {LIGHT}; }}
.list {{ border: 1px solid {LIGHT}; border-radius: 9px; padding: 6px; display: flex; flex-direction: column; gap: 4px;
         background: #fafbfc; }}
.lh {{ display: flex; align-items: center; gap: 7px; font-weight: 800; font-size: 15px; padding: 0 2px 2px; }}
.chip {{ flex: none; display: inline-flex; align-items: center; justify-content: center; width: 40px; height: 24px;
         border-radius: 6px; font-size: 14px; font-weight: 800; font-variant-numeric: tabular-nums; }}
.pair {{ display: flex; flex-direction: column; gap: 4px; }}
.pq {{ display: flex; gap: 8px; align-items: flex-start; font-size: 14.5px; line-height: 1.3; color: #39445a;
       padding: 5px 8px; border: 1px solid {LIGHT}; border-radius: 7px; }}
.pq svg {{ flex: none; margin-top: 1px; }}
.dist {{ display: flex; align-items: flex-end; gap: 14px; padding: 4px 6px 0; }}
.bars {{ flex: 1; display: flex; align-items: flex-end; gap: 6px; height: 98px; border-bottom: 1.5px solid {INK}; }}
.bc {{ flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: flex-end; height: 100%;
       position: relative; }}
.bar {{ width: 100%; border-radius: 4px 4px 0 0; }}
.bc span {{ position: absolute; bottom: -21px; font-size: 13.5px; font-weight: 700; color: {GRAY}; }}
.ev {{ display: flex; flex-direction: column; align-items: center; gap: 5px; padding-bottom: 6px; }}
.ev small {{ font-size: 12.5px; color: {GRAY}; font-weight: 600; text-align: center; line-height: 1.2; }}
.grid {{ display: flex; flex-direction: column; gap: 5px; margin-top: 26px; }}
.dist {{ margin-top: 6px; }}
.gl {{ display: flex; align-items: center; justify-content: space-between; gap: 6px; font-size: 14px; font-weight: 700; }}
.gl span {{ flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.gl i {{ color: {GRAY}; }}
.nd {{ display: flex; align-items: center; gap: 8px; font-size: 14.5px; font-weight: 700; }}
.nd span {{ width: 170px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.track {{ flex: 1; height: 14px; background: #f0f2f5; border-radius: 4px; overflow: hidden; }}
.track div {{ height: 100%; background: {BLUE}; border-radius: 4px; }}
.nd b {{ width: 26px; text-align: right; font-variant-numeric: tabular-nums; }}
.board {{ border: 1.5px solid {INK}; border-radius: 9px; padding: 6px 10px; display: flex; flex-direction: column; gap: 3px; }}
.bh {{ display: flex; font-size: 12.5px; font-weight: 700; color: {GRAY}; text-transform: uppercase; letter-spacing: .4px;
       padding-bottom: 3px; border-bottom: 1px solid {LIGHT}; }}
.bh span:first-child {{ flex: 1; }}
.row {{ display: flex; align-items: center; gap: 8px; font-size: 14.5px; font-weight: 600; }}
.row .nm {{ flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.row b {{ font-variant-numeric: tabular-nums; }}
.mt {{ width: 44px; text-align: center; color: {ORANGE}; font-weight: 800; }}
.bh .mt {{ color: {ORANGE}; }}
.result {{ display: flex; align-items: stretch; gap: 14px; margin-top: 14px; }}
.result .lbl {{ display: flex; align-items: center; font-size: 15px; font-weight: 800; color: {GRAY}; text-transform: uppercase;
                letter-spacing: .6px; width: 150px; line-height: 1.25; }}
.stat {{ flex: 1; display: flex; align-items: center; gap: 14px; border-radius: 10px; background: #f6f7f9; padding: 10px 16px; }}
.stat b {{ white-space: nowrap; font-size: 30px; font-weight: 800; color: {BLUE}; font-variant-numeric: tabular-nums; letter-spacing: -.5px; }}
.stat span {{ font-size: 15.5px; font-weight: 600; line-height: 1.3; color: #39445a; }}
.stat em {{ font-style: normal; color: {ORANGE}; font-weight: 800; }}
</style></head><body>
<div class="flow">{inputs}{ARROW}{write}{ARROW}{retrieve}{ARROW}{judge}{ARROW}{rank}</div>
<div class="result"><div class="lbl">Compared with MTEB</div>
  <div class="stat"><b>ρ = {RHO:.2f}</b><span>Spearman with <em>MTEB's ranking</em> from human<br>relevance labels (28 models, 28 tasks)</span></div>
  <div class="stat"><b>{HUMAN_GYM:.2f}</b><span>agreement with human votes in MTEB Arena,<br>above <em>MTEB's labels ({HUMAN_MTEB:.2f})</em></span></div>
  <div class="stat"><b>{SMALL_JUDGE:.2f}</b><span>agreement with MTEB's ranking<br>with a 4B judge (decider-4b)</span></div>
</div>
</body></html>"""

(FIGURES / "figure1.html").write_text(HTML)
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1800, "height": 1000}, device_scale_factor=2)
    page.set_content(HTML, wait_until="networkidle")
    size = page.locator("body").bounding_box()
    width, height = size["width"], size["height"]
    page.set_viewport_size({"width": int(width) + 1, "height": int(height) + 1})
    page.pdf(path=str(FIGURES / "figure1.pdf"), width=f"{width}px", height=f"{height}px", print_background=True,
             page_ranges="1")
    page.screenshot(path=str(FIGURES / "figure1.png"), clip={"x": 0, "y": 0, "width": width, "height": height})
    browser.close()
print("wrote", FIGURES / "figure1.pdf")
