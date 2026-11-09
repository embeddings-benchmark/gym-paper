"""Figure: overview of MTEB Gym, as an illustration (HTML printed to PDF/PNG with headless Chromium).

Read left to right as one sentence: given only a corpus, an LLM writes queries, every model retrieves and a judge grades
each result, and the models are ranked. Real data from SciFact (generated queries, JEV-27B, pointwise). Icons: Microsoft
Fluent Emoji (MIT), in figures/icons; organization logos from Hugging Face, in figures/logos."""

import base64
import json

from huggingface_hub import hf_hub_download
from playwright.sync_api import sync_playwright

from common import GENERATOR, GYM_RUNS, JUDGE, agreement, judged, ratings, records
from mteb_gym import corpus
from plots import BLUE, FIGURES

TASK, QID = "SciFact", "q50"
QUERY_CARDS = ["q7", "q43", QID]
RESULT_MODELS = ["mteb/baseline-bm25s", "Qwen/Qwen3-Embedding-8B", "sentence-transformers/all-MiniLM-L6-v2"]
FILLER = ["16737163", "7227763", "13312471", "2374637"]
NAVY, CREAM, INK = "#123a7a", "#fbf7ef", "#14213d"

R = records()
rec = R[(JUDGE, "pointwise", GENERATOR)][TASK]
_, _, top, grades, _ = judged(rec)
docs = corpus.load(TASK).docs
queries = {}
for line in open(hf_hub_download(GYM_RUNS, "tables/queries/queries.jsonl", repo_type="dataset")):
    q = json.loads(line)
    if q["task"] == TASK and q["query_set"] == rec["config"]["query_set"].rsplit("-", 1)[0]:
        queries[q["qid"]] = q
board = sorted(ratings(rec).items(), key=lambda x: -x[1])
place = {m: i + 1 for i, (m, _) in enumerate(board)}
rho = agreement(ratings(rec), TASK)


def asset(path, mime):
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def icon(name, size, style=""):
    mime = "image/svg+xml" if name.endswith(".svg") else "image/png"
    return f'<img src="{asset(FIGURES / "icons" / name, mime)}" width="{size}" height="{size}" style="{style}">'


def logo(m, size=26):
    return (f'<img src="{asset(FIGURES / "logos" / (m.split("/")[0] + ".png"), "image/png")}" width="{size}" '
            f'height="{size}" style="border-radius:7px">')


def title(d, n=30):
    text = docs[d] if isinstance(docs[d], str) else json.dumps(docs[d])
    return text[:n].rsplit(" ", 1)[0] + "…" if len(text) > n else text


def short(m):
    return m.split("/")[-1].replace("baseline-bm25s", "BM25").replace("-embedding-english-r2", "-embedding-r2")


def grade_badge(g):
    if g >= 3:
        bg, fg, ring = "#1f5bd6", "#fff", "#c9d8f7"
    elif g >= 2:
        bg, fg, ring = "#8fb0ec", "#fff", "#e1e9fa"
    else:
        bg, fg, ring = "#f2d4cf", "#b3261e", "#fbeae7"
    return f'<span class="gb" style="background:{bg};color:{fg};box-shadow:0 0 0 4px {ring}">{g:.1f}</span>'


GRAIN = ("url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='160' height='160'>"
         "<filter id='n'><feTurbulence type='fractalNoise' baseFrequency='.85' numOctaves='2' stitchTiles='stitch'/>"
         "<feColorMatrix values='0 0 0 0 1  0 0 0 0 1  0 0 0 0 1  0 0 0 .09 0'/></filter>"
         "<rect width='100%' height='100%' filter='url(%23n)'/></svg>\")")

seeds = queries[QID]["seed_doc_ids"]
rows = [(seeds[0], True), (FILLER[0], False), (seeds[1], True), (FILLER[1], False), (FILLER[2], False),
        (seeds[2], True), (FILLER[3], False)]
corpus_html = "".join(f'<div class="doc {"on" if s else ""}">{icon("page_facing_up_flat.svg", 22)}<span>{title(d)}</span>'
                      f'{icon("writing_hand_flat_default.svg", 22, "margin-left:auto") if s else ""}</div>'
                      for d, s in rows)

QPOS = [(14, 14, -4), (138, 2, 3.5), (58, 118, -1)]
qcards = "".join(
    f'<div class="qc {"front" if q == QID else ""}" style="left:{x}px;top:{y}px;transform:rotate({r}deg)">'
    f'<div class="qh">{icon("magnifying_glass_tilted_right_flat.svg", 18)}query {q[1:]}</div>'
    f'<div class="qt">{queries[q]["text"]}</div></div>' for q, (x, y, r) in zip(QUERY_CARDS, QPOS))

RPOS = [(30, 2, -2), (0, 112, -2.5), (214, 126, 2.5)]
rcards = ""
for m, (x, y, r) in zip(RESULT_MODELS, RPOS):
    lines = "".join(f'<div class="rl"><span class="bar"></span>{grade_badge(grades[QID][d])}</div>'
                    for d in top[m][QID][:3])
    rcards += (f'<div class="rc {"front" if m != RESULT_MODELS[0] else ""}" style="left:{x}px;top:{y}px;'
               f'transform:rotate({r}deg)"><div class="rh">{logo(m, 22)}<span>{short(m)}</span></div>{lines}</div>')

lb_rows, last = "", 0
for r in sorted({1, 2, 3, place[RESULT_MODELS[2]], place[RESULT_MODELS[1]], place[RESULT_MODELS[0]]}):
    if r > last + 1:
        lb_rows += '<div class="lr dots">⋮</div>'
    m, s = board[r - 1]
    lb_rows += (f'<div class="lr"><span class="k">{r}</span>{logo(m, 26)}'
                f'<span class="n">{short(m)}</span><span class="s">{s:.1f}</span></div>')
    last = r


def dashed(path, head):
    return (f'<path d="{path}" stroke="{NAVY}" stroke-width="2.6" stroke-dasharray="7 7" fill="none" '
            f'stroke-linecap="round"/><path d="{head}" stroke="{NAVY}" stroke-width="2.6" fill="none" '
            f'stroke-linecap="round" stroke-linejoin="round"/>')


FAN = ('<svg width="92" height="330" viewBox="0 0 92 330">'
       + dashed("M4 60 C 40 60, 46 110, 84 120", "M74 112 L85 120 L73 127")
       + dashed("M4 165 C 34 165, 50 165, 84 165", "M74 158 L85 165 L74 172")
       + dashed("M4 270 C 40 270, 46 220, 84 210", "M73 203 L85 210 L74 218") + "</svg>")
ARROW = (f'<svg width="56" height="30" viewBox="0 0 56 30"><path d="M4 15h42M36 5l11 10-11 10" stroke="{NAVY}" '
         f'stroke-width="3.2" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>')

HTML = f"""<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Nunito:wght@600;700;800;900&display=swap" rel="stylesheet">
<style>
* {{ box-sizing: border-box; margin: 0; }}
body {{ font-family: Nunito, sans-serif; color: {INK}; background: {CREAM}; display: inline-block; padding: 26px 26px 18px; }}
.fig {{ display: flex; align-items: flex-start; }}
.col {{ display: flex; flex-direction: column; align-items: center; gap: 18px; }}
.cap {{ font-size: 22px; font-weight: 800; color: {NAVY}; text-align: center; }}
.mid {{ height: 330px; display: flex; align-items: center; justify-content: center; }}
.navy {{ width: 330px; height: 330px; border-radius: 18px; background: {NAVY}; background-image: {GRAIN};
         color: #fff; padding: 20px 20px; box-shadow: 0 10px 24px rgba(18,58,122,.22); }}
.navy h3 {{ display: flex; align-items: center; gap: 12px; font-size: 25px; font-weight: 800; padding-bottom: 12px;
            border-bottom: 1.5px solid rgba(255,255,255,.22); margin-bottom: 10px; }}
.navy h3 small {{ margin-left: auto; font-size: 14px; font-weight: 700; color: #a9bde6; }}
.doc {{ display: flex; align-items: center; gap: 11px; font-size: 16.5px; font-weight: 700; color: #b8c8ea;
        padding: 4px 8px; border-radius: 9px; white-space: nowrap; }}
.doc.on {{ background: rgba(255,255,255,.14); color: #fff; }}
.hub {{ width: 108px; height: 108px; border-radius: 50%; background: #fff; border: 3px solid {NAVY};
        display: flex; align-items: center; justify-content: center; box-shadow: 0 8px 20px rgba(18,58,122,.18); }}
.hubwrap {{ display: flex; flex-direction: column; align-items: center; gap: 8px; }}
.hubname {{ font-size: 14.5px; font-weight: 800; color: {NAVY}; text-align: center; line-height: 1.2; }}
.hubname span {{ display: block; font-weight: 700; color: #5d6f93; font-size: 13px; }}
.soft {{ position: relative; width: 410px; height: 330px; border-radius: 22px; background: #e7eef9;
         background-image: {GRAIN}; }}
.qc {{ position: absolute; width: 250px; background: #fff; border-radius: 12px; border: 1.5px solid #d7dfee;
       box-shadow: 0 6px 16px rgba(20,33,61,.10); }}
.qc.front {{ border: 2px solid {NAVY}; box-shadow: 0 12px 24px rgba(20,33,61,.18); z-index: 3; }}
.qh {{ display: flex; align-items: center; gap: 7px; font-size: 13.5px; font-weight: 800; color: #5d6f93;
       padding: 7px 12px; border-bottom: 1.5px solid #edf1f7; }}
.qt {{ font-size: 16px; font-weight: 800; line-height: 1.32; padding: 9px 13px 12px; height: 96px; overflow: hidden; }}
.qc.front .qt {{ height: auto; }}
.qc.front {{ width: 290px; }}
.qc:not(.front) .qt {{ color: #9aa6bf; }}
.rc {{ position: absolute; width: 230px; background: #fff; border-radius: 12px; border: 1.5px solid #d7dfee;
       box-shadow: 0 6px 16px rgba(20,33,61,.10); padding-bottom: 8px; }}
.rc.front {{ border: 2px solid {NAVY}; box-shadow: 0 12px 24px rgba(20,33,61,.18); z-index: 3; }}
.rh {{ display: flex; align-items: center; gap: 8px; font-size: 15px; white-space: nowrap; font-weight: 800; padding: 8px 12px;
       border-bottom: 1.5px solid #edf1f7; margin-bottom: 4px; }}
.rl {{ display: flex; align-items: center; gap: 12px; padding: 5px 14px; }}
.bar {{ flex: 1; height: 9px; border-radius: 5px; background: #e3e8f1; }}
.gb {{ display: inline-flex; align-items: center; justify-content: center; width: 40px; height: 26px;
       border-radius: 13px; font-size: 14px; font-weight: 900; }}
.judge {{ position: absolute; right: 26px; top: 14px; flex-direction: row; display: flex; align-items: center; gap: 10px; z-index: 5; }}
.judge .hub {{ width: 84px; height: 84px; }}
.judge .hubname {{ text-align: left; }}
.board {{ width: 360px; height: 330px; border-radius: 18px; background: #fff; border: 1.5px solid #d7dfee;
          box-shadow: 0 10px 24px rgba(20,33,61,.10); overflow: hidden; display: flex; flex-direction: column; }}
.board h3 {{ display: flex; align-items: center; gap: 11px; font-size: 24px; font-weight: 800; padding: 15px 20px;
             border-bottom: 1.5px solid #edf1f7; }}
.board h3 small {{ margin-left: auto; font-size: 14px; color: #5d6f93; font-weight: 700; }}
.lr {{ display: flex; align-items: center; gap: 12px; padding: 7px 20px; font-size: 17px; font-weight: 700; }}
.lr.dots {{ padding: 0 0 0 28px; color: #9aa6bf; line-height: .9; }}
.lr .k {{ width: 22px; text-align: right; color: #9aa6bf; font-weight: 800; }}
.lr .n {{ flex: 1; white-space: nowrap; }}
.lr .s {{ font-weight: 900; color: {BLUE}; }}
.foot {{ margin-top: auto; display: flex; align-items: center; gap: 10px; padding: 11px 20px; background: #f3f6fb;
         border-top: 1.5px solid #edf1f7; font-size: 14.5px; font-weight: 700; color: #3c4a66; }}
.foot b {{ font-size: 21px; color: #1d8a4e; font-weight: 900; white-space: nowrap; }}
</style></head><body><div class="fig">

<div class="col"><div class="navy">
  <h3>{icon("card_file_box_flat.svg", 30)}{TASK}<small>5,183 docs</small></h3>
  {corpus_html}
</div><div class="cap">Given only your documents,</div></div>

<div class="mid">{FAN}</div>
<div class="col"><div class="mid"><div class="hubwrap"><div class="hub">{icon("writing_hand_3d_default.png", 66)}</div>
  <div class="hubname">Query writer<span>DeepSeek-V4-Flash</span></div></div></div></div>
<div class="mid" style="width:70px">{ARROW}</div>

<div class="col"><div class="soft">{qcards}</div><div class="cap">an LLM writes realistic queries,</div></div>

<div class="mid" style="width:70px">{ARROW}</div>

<div class="col"><div class="soft" style="width:490px"><div style="position:absolute;left:22px;top:22px">{rcards}</div>
  <div class="judge"><div class="hub">{icon("balance_scale_3d.png", 54)}</div>
    <div class="hubname">Judge<span>JEV-27B</span></div></div></div>
  <div class="cap">each model retrieves, a judge grades every result,</div></div>

<div class="mid" style="width:70px">{ARROW}</div>

<div class="col"><div class="board">
  <h3>{icon("trophy_3d.png", 32)}Leaderboard<small>{TASK}</small></h3>
  {lb_rows}
  <div class="foot">{icon("check_mark_button_flat.svg", 22)}<b>ρ = {rho:.2f}</b>agreement with MTEB</div>
</div><div class="cap">and the models are ranked.</div></div>

</div></body></html>"""

(FIGURES / "overview.html").write_text(HTML)
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 2400, "height": 700}, device_scale_factor=2)
    page.set_content(HTML, wait_until="networkidle")
    size = page.locator("body").bounding_box()
    width, height = size["width"], size["height"]
    page.set_viewport_size({"width": int(width) + 1, "height": int(height) + 1})
    page.pdf(path=str(FIGURES / "overview.pdf"), width=f"{width}px", height=f"{height}px", print_background=True,
             page_ranges="1")
    page.screenshot(path=str(FIGURES / "overview.png"), clip={"x": 0, "y": 0, "width": width, "height": height})
    browser.close()
print("wrote", FIGURES / "overview.pdf")
