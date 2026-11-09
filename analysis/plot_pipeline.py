"""Figure: how MTEB Gym works, on real data from SciFact (generated queries, JEV-27B).

Drawn as HTML (organization logos from Hugging Face, in figures/logos) and printed to PDF with headless Chromium.
1. The corpus and the task description; the three documents sampled for query q50 are highlighted.
2. The query writer turns them into a query; repeated for 100 queries per task.
3. Every model returns its top 10 for the query; each square is a returned document, colored by the judge's grade.
4. The task's ranking: nDCG@10 with the judge's grades, averaged over the 100 queries."""

import base64
import json

from huggingface_hub import hf_hub_download
from playwright.sync_api import sync_playwright

from common import GENERATOR, GYM_RUNS, JUDGE, judged, ratings, records
from mteb_gym import corpus
from plots import BLUE, FIGURES, ORANGE

TASK, QID = "SciFact", "q50"
OTHER_QUERIES = ["q7", "q43"]
ROWS = ["Qwen/Qwen3-Embedding-8B", "ibm-granite/granite-embedding-english-r2", "lightonai/DenseOn",
        "sentence-transformers/all-MiniLM-L6-v2", "mteb/baseline-bm25s"]
FILLER = ["16737163", "7227763", "13312471", "2374637"]
LOGO = {"Qwen": "Qwen", "ibm-granite": "ibm-granite", "lightonai": "lightonai", "sentence-transformers":
        "sentence-transformers", "mteb": "mteb", "microsoft": "microsoft"}
SCALE = ["#eef2f9", "#cbd7ee", "#9fb4df", "#6b8bcd", "#3a64b8", BLUE]  # grade 0..5

R = records()
rec = R[(JUDGE, "pointwise", GENERATOR)][TASK]
_, _, top, grades, _ = judged(rec)
docs = corpus.load(TASK).docs
queries = {}
for line in open(hf_hub_download(GYM_RUNS, "tables/queries/queries.jsonl", repo_type="dataset")):
    q = json.loads(line)
    if q["task"] == TASK and q["generator"] != "original":
        queries[q["qid"]] = q
board = sorted(ratings(rec).items(), key=lambda x: -x[1])
place = {m: i + 1 for i, (m, _) in enumerate(board)}


def logo(name, size=22, radius=6):
    data = base64.b64encode((FIGURES / "logos" / f"{name}.png").read_bytes()).decode()
    return (f'<img src="data:image/png;base64,{data}" width="{size}" height="{size}" '
            f'style="border-radius:{radius}px;flex:none">')


def model_logo(m, size=22):
    return logo(LOGO.get(m.split("/")[0], "mteb"), size)


def title(d, n=26):
    text = docs[d] if isinstance(docs[d], str) else json.dumps(docs[d])
    return text[:n].rsplit(" ", 1)[0] + "…" if len(text) > n else text


def short(m):
    return (m.split("/")[-1].replace("baseline-bm25s", "BM25").replace("-embedding-english-r2", "-embedding-r2"))


ICON = {
    "db": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>'
          '<path d="M3 12c0 1.66 4 3 9 3s9-1.34 9-3"/>',
    "file": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/>',
    "search": '<circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/>',
    "rank": '<path d="M8 21h8"/><path d="M12 17v4"/><path d="M7 4h10v5a5 5 0 0 1-10 0z"/>'
            '<path d="M17 5h3v2a3 3 0 0 1-3 3"/><path d="M7 5H4v2a3 3 0 0 0 3 3"/>',
}


def icon(name, size=20, color="currentColor", width=2):
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" '
            f'stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round">{ICON[name]}</svg>')


seeds = queries[QID]["seed_doc_ids"]
corpus_rows = [(seeds[0], True), (FILLER[0], False), (seeds[1], True), (FILLER[1], False), (FILLER[2], False),
               (seeds[2], True), (FILLER[3], False)]
corpus_html = "".join(f'<div class="doc {"on" if s else ""}">{icon("file", 16, "#fff" if s else "#7d8597", 1.8)}'
                      f'<span>{title(d)}</span></div>' for d, s in corpus_rows)

cards = ""
for k, qid in enumerate(OTHER_QUERIES + [QID]):
    front = qid == QID
    cards += (f'<div class="qcard {"front" if front else ""}" style="left:{16 * (2 - k)}px;top:{44 - 22 * k}px">'
              f'<div class="qhead">{icon("search", 13, "#5a6478", 2.2)}<span>query {qid[1:]}</span></div>'
              f'<div class="qtext">{queries[qid]["text"]}</div></div>')


def squares(m):
    cells = ""
    for d in top[m][QID][:10]:
        g = grades[QID].get(d, 0.0)
        color = SCALE[min(int(round(g)), 5)]
        cells += f'<span class="sq" style="background:{color}" title="{g:.1f}"></span>'
    return cells


retrieve_rows = ""
for m in ROWS:
    if m == "sentence-transformers/all-MiniLM-L6-v2":
        retrieve_rows += '<div class="more">+ 23 more models</div>'
    retrieve_rows += (f'<div class="mrow">{model_logo(m, 20)}<span class="mname">{short(m)}</span>'
                      f'<span class="cells">{squares(m)}</span></div>')
legend = "".join(f'<span class="lg"><span class="sq" style="background:{c}"></span>{i}</span>'
                 for i, c in enumerate(SCALE))

top_score = board[0][1]
rank_rows, last = "", 0
for r in sorted({1, 2, 3, place[ROWS[0]], place[ROWS[3]], place[ROWS[4]]}):
    if r > last + 1:
        rank_rows += '<div class="rrow dots">⋮</div>'
    m, s = board[r - 1]
    rank_rows += (f'<div class="rrow"><span class="rk">{r}</span>{model_logo(m, 20)}<span class="rn">{short(m)}</span>'
                  f'<span class="rs">{s:.1f}</span></div>')
    last = r

ARROW = ('<svg width="36" height="18" viewBox="0 0 36 18"><path d="M2 9h28" stroke="#1a2233" stroke-width="2" '
         'fill="none"/><path d="M25 3l6 6-6 6" stroke="#1a2233" stroke-width="2" fill="none" stroke-linecap="round" '
         'stroke-linejoin="round"/></svg>')
FAN = ('<svg width="44" height="170" viewBox="0 0 44 170">' + "".join(
    f'<path d="M2 {y} C 26 {y}, 22 85, 40 85" stroke="#1a2233" stroke-width="2" stroke-dasharray="2 5" fill="none" '
    f'stroke-linecap="round"/>' for y in (18, 85, 152)) + '</svg>')

HTML = f"""<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Nunito:wght@500;600;700;800&display=swap" rel="stylesheet">
<style>
* {{ box-sizing: border-box; margin: 0; }}
body {{ font-family: Nunito, sans-serif; color: #1a2233; background: #fff; display: inline-block; padding: 14px; }}
.fig {{ display: flex; align-items: flex-start; gap: 8px; }}
.step {{ display: flex; flex-direction: column; gap: 12px; align-items: center; }}
.label {{ font-size: 16px; font-weight: 700; color: #3d4556; display: flex; align-items: center; gap: 8px; }}
.label b {{ display: inline-flex; width: 22px; height: 22px; border-radius: 50%; background: #1a2233; color: #fff;
            align-items: center; justify-content: center; font-size: 12.5px; }}
.mid {{ height: 330px; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 8px; }}
.dark {{ background: #111; border-radius: 14px; padding: 18px 16px; width: 270px; height: 330px; color: #fff;
         display: flex; flex-direction: column; }}
.dark h3 {{ display: flex; align-items: center; gap: 9px; font-size: 20px; font-weight: 800; }}
.dark .sub {{ font-size: 13px; color: #9aa3b5; margin: 2px 0 10px 29px; }}
.doc {{ display: flex; align-items: center; gap: 9px; font-size: 14px; color: #8a93a6; padding: 4px 8px;
        border-radius: 7px; white-space: nowrap; }}
.doc.on {{ color: #fff; background: rgba(255,90,10,.28); font-weight: 700; }}
.task {{ margin-top: auto; font-size: 12.5px; color: #c9cfdb; border-top: 1px solid #2c3140; padding-top: 9px;
         line-height: 1.35; }}
.task span {{ color: #8a93a6; }}
.circle {{ width: 72px; height: 72px; border-radius: 50%; border: 2px solid #1a2233; background: #fff;
           display: flex; align-items: center; justify-content: center; overflow: hidden; }}
.who {{ font-size: 13px; font-weight: 800; text-align: center; line-height: 1.25; width: 120px; }}
.who span {{ display: block; font-weight: 600; color: #5a6478; }}
.blue {{ background: #edf2fb; border-radius: 14px; width: 300px; height: 330px; position: relative; }}
.stack {{ position: absolute; left: 18px; top: 52px; }}
.qcard {{ position: absolute; width: 238px; background: #fff; border: 1.5px solid #c9d2e3; border-radius: 9px;
          overflow: hidden; }}
.qcard.front {{ border-color: #1a2233; }}
.qhead {{ display: flex; align-items: center; gap: 6px; font-size: 12px; font-weight: 700; color: #5a6478;
          background: #f1f3f7; padding: 5px 10px; border-bottom: 1px solid #dfe3ea; }}
.qtext {{ font-size: 15px; font-weight: 700; line-height: 1.35; padding: 10px 12px; height: 108px; }}
.qcard:not(.front) .qtext {{ color: #b5bccb; }}
.chip {{ position: absolute; right: 16px; bottom: 16px; background: #1a2233; color: #fff; font-size: 13px;
         font-weight: 800; padding: 5px 11px; border-radius: 14px; }}
.panel {{ background: #fff; border: 1.5px solid #c9d2e3; border-radius: 14px; width: 430px; height: 330px;
          overflow: hidden; display: flex; flex-direction: column; }}
.phead {{ display: flex; align-items: center; gap: 10px; padding: 10px 14px; background: #f1f3f7;
          border-bottom: 1.5px solid #dfe3ea; font-size: 14px; font-weight: 700; color: #3d4556; line-height: 1.25; }}
.phead .who2 {{ font-size: 16px; font-weight: 800; color: #1a2233; }}
.pbody {{ padding: 10px 14px 8px; flex: 1; display: flex; flex-direction: column; }}
.colhead {{ display: flex; justify-content: space-between; font-size: 12px; font-weight: 700; color: #8a93a6;
            text-transform: uppercase; letter-spacing: .4px; margin: 0 0 6px 30px; }}
.mrow {{ display: flex; align-items: center; gap: 10px; padding: 5px 0; }}
.mname {{ font-size: 14.5px; font-weight: 700; width: 152px; white-space: nowrap; }}
.cells {{ display: flex; gap: 4px; }}
.sq {{ width: 17px; height: 17px; border-radius: 4px; display: inline-block; }}
.more {{ font-size: 13px; color: #8a93a6; font-weight: 600; padding: 3px 0 3px 30px; }}
.legend {{ margin-top: auto; display: flex; align-items: center; gap: 10px; font-size: 12.5px; color: #5a6478;
           font-weight: 700; border-top: 1px solid #e6e9ef; padding-top: 8px; }}
.lg {{ display: inline-flex; align-items: center; gap: 4px; }}
.lg .sq {{ width: 13px; height: 13px; }}
.result {{ background: #fff; border: 1.5px solid #c9d2e3; border-radius: 14px; width: 300px; height: 330px;
           overflow: hidden; display: flex; flex-direction: column; }}
.result h3 {{ display: flex; align-items: center; gap: 9px; font-size: 19px; font-weight: 800; padding: 13px 16px;
              background: #f1f3f7; border-bottom: 1.5px solid #dfe3ea; }}
.rlist {{ padding: 8px 16px; flex: 1; }}
.rrow {{ display: flex; align-items: center; gap: 9px; font-size: 14.5px; padding: 5px 0; }}
.rrow.dots {{ color: #9aa3b5; padding: 0 0 0 4px; line-height: 1; }}
.rk {{ width: 20px; color: #8a93a6; font-weight: 700; text-align: right; }}
.rn {{ flex: 1; font-weight: 700; white-space: nowrap; }}
.rs {{ font-weight: 800; color: {BLUE}; font-variant-numeric: tabular-nums; }}
.foot {{ font-size: 12px; color: #8a93a6; padding: 0 16px 12px; line-height: 1.35; }}
</style></head><body><div class="fig">

<div class="step"><div class="dark">
  <h3>{icon("db", 20, "#fff")} {TASK}</h3>
  <div class="sub">5,183 documents · no labels</div>
  {corpus_html}
  <div class="task"><span>Task:</span> {rec["config"]["task_instruction"]}</div>
</div><div class="label"><b>1</b>Sample documents</div></div>

<div class="mid">{FAN}</div>
<div class="mid"><div class="circle">{logo("deepseek-ai", 50, 10)}</div>
  <div class="who">Query writer<span>DeepSeek-V4-Flash</span></div></div>
<div class="mid">{ARROW}</div>

<div class="step"><div class="blue"><div class="stack">{cards}</div><div class="chip">× 100 queries</div></div>
  <div class="label"><b>2</b>Write queries</div></div>

<div class="mid">{ARROW}</div>

<div class="step"><div class="panel">
  <div class="phead"><div class="circle" style="width:42px;height:42px;border-width:1.5px;flex:none">
    {logo("autotrust", 30, 6)}</div><div><div class="who2">Judge: JEV-27B</div>
    grades every retrieved document 0–5</div></div>
  <div class="pbody">
    <div class="colhead"><span>28 models</span><span>top 10 for query 50, colored by grade</span></div>
    {retrieve_rows}
    <div class="legend">judge's grade {legend}</div>
  </div>
</div><div class="label"><b>3</b>Retrieve and judge</div></div>

<div class="mid">{ARROW}</div>

<div class="step"><div class="result">
  <h3>{icon("rank", 21, BLUE)} Model ranking</h3>
  <div class="rlist">{rank_rows}</div>
  <div class="foot">nDCG@10 with the judge's grades, mean over 100 queries</div>
</div><div class="label"><b>4</b>Rank the models</div></div>

</div></body></html>"""

(FIGURES / "pipeline.html").write_text(HTML)
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 2000, "height": 600}, device_scale_factor=2)
    page.set_content(HTML, wait_until="networkidle")
    size = page.locator("body").bounding_box()
    width, height = size["width"], size["height"]
    page.set_viewport_size({"width": int(width) + 1, "height": int(height) + 1})
    page.pdf(path=str(FIGURES / "pipeline.pdf"), width=f"{width}px", height=f"{height}px", print_background=True,
             page_ranges="1")
    page.screenshot(path=str(FIGURES / "pipeline.png"), clip={"x": 0, "y": 0, "width": width, "height": height})
    browser.close()
print("wrote", FIGURES / "pipeline.pdf")
