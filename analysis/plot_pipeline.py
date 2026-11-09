"""Figure: how MTEB Gym works, on real data from SciFact (generated queries, JEV-27B).

Drawn as HTML and printed to PDF with headless Chromium (playwright), so text and shapes stay vector.
1. The corpus: the three documents the query writer sampled for query q50 are highlighted.
2. The query writer (DeepSeek-V4-Flash) writes one query from them; a few other generated queries behind it.
3. Two models' top 3 for q50 with the judge's grades (0-5).
4. The task's ranking: the gym's score is nDCG@10 with the judge's grades, averaged over the 100 queries."""

import json

from huggingface_hub import hf_hub_download
from playwright.sync_api import sync_playwright

from common import GENERATOR, GYM_RUNS, JUDGE, judged, ratings, records
from mteb_gym import corpus
from plots import BLUE, FIGURES, ORANGE

TASK, QID = "SciFact", "q50"
OTHER_QUERIES = ["q7", "q43"]
MODELS = ["Qwen/Qwen3-Embedding-8B", "sentence-transformers/all-MiniLM-L6-v2"]
FILLER = ["16737163", "7227763", "13312471", "2374637"]  # unsampled documents shown in the corpus panel

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


def title(d, n=34):
    text = docs[d] if isinstance(docs[d], str) else json.dumps(docs[d])
    return text[:n].rsplit(" ", 1)[0] + "…" if len(text) > n else text


def short(m):
    return m.split("/")[-1]


ICON = {
    "db": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>'
          '<path d="M3 12c0 1.66 4 3 9 3s9-1.34 9-3"/>',
    "file": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/>',
    "pen": '<path d="M12 20h9"/><path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4Z"/>',
    "scale": '<path d="M12 3v17"/><path d="M5 7h14"/><path d="M5 7 2 14a3 3 0 0 0 6 0Z"/>'
             '<path d="m19 7-3 7a3 3 0 0 0 6 0Z"/><path d="M8 21h8"/>',
    "rank": '<path d="M3 3v18h18"/><path d="M7 16v-5"/><path d="M12 16V7"/><path d="M17 16v-8"/>',
    "query": '<circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/>',
}


def icon(name, size=22, color="currentColor", width=2):
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" '
            f'stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round">{ICON[name]}</svg>')


def badge(g):
    bg, fg = (BLUE, "#fff") if g >= 3 else ("#9fb4df", "#fff") if g >= 2 else ("#e3e8f2", "#1a2233")
    return f'<span class="badge" style="background:{bg};color:{fg}">{g:.1f}</span>'


seeds = queries[QID]["seed_doc_ids"]
order = [(seeds[0], True), (FILLER[0], False), (seeds[1], True), (FILLER[1], False), (FILLER[2], False),
         (seeds[2], True), (FILLER[3], False)]
corpus_html = "".join(
    f'<div class="row {"seed" if s else ""}">{icon("file", 17, "#fff" if s else "#8a93a6", 1.8)}'
    f'<span>{title(d, 30)}</span></div>' for d, s in order)

cards = ""
for k, qid in enumerate(OTHER_QUERIES + [QID]):
    front = qid == QID
    cards += (f'<div class="qcard {"front" if front else ""}" style="left:{18 * (2 - k)}px;top:{48 - 24 * k}px">'
              f'<div class="qhead">{icon("query", 14, "#5a6478", 2)}<span>generated query</span></div>'
              f'<div class="qtext">{queries[qid]["text"]}</div></div>')

model_html = ""
for m in MODELS:
    rows = "".join(f'<div class="grow">{badge(grades[QID][d])}<span>{title(d, 30)}</span></div>'
                   for d in top[m][QID][:3])
    model_html += f'<div class="mcard"><div class="mhead">{short(m)}</div>{rows}</div>'

top_score = board[0][1]
rank_rows, last = "", 0
for r in sorted({1, 2, 3, 4, 5, place[MODELS[0]], place[MODELS[1]]}):
    if r > last + 1:
        rank_rows += '<div class="rrow dots">⋮</div>'
    m, s = board[r - 1]
    rank_rows += (f'<div class="rrow {"mark" if m in MODELS else ""}"><span class="rk">{r}</span>'
                  f'<span class="rn">{short(m).replace("-embedding-english-r2", "-embedding-r2")}</span><span class="bar"><i style="width:'
                  f'{100 * (s - 75) / (top_score - 75):.0f}%"></i></span><span class="rs">{s:.1f}</span></div>')
    last = r

ARROW = ('<svg width="40" height="20" viewBox="0 0 40 20"><path d="M2 10h32" stroke="#1a2233" stroke-width="2" '
         'fill="none"/><path d="M28 4l7 6-7 6" stroke="#1a2233" stroke-width="2" fill="none" stroke-linecap="round" '
         'stroke-linejoin="round"/></svg>')
FAN = ('<svg width="56" height="150" viewBox="0 0 56 150">' + "".join(
    f'<path d="M2 {y} C 30 {y}, 30 75, 52 75" stroke="#1a2233" stroke-width="2" stroke-dasharray="2 6" '
    f'fill="none" stroke-linecap="round"/>' for y in (12, 75, 138)) + '</svg>')

HTML = f"""<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Nunito:wght@500;600;700;800&display=swap" rel="stylesheet">
<style>
* {{ box-sizing: border-box; margin: 0; }}
body {{ font-family: Nunito, sans-serif; color: #1a2233; background: #fff; display: inline-block; padding: 16px; }}
.fig {{ display: flex; align-items: flex-start; gap: 10px; }}
.step {{ display: flex; flex-direction: column; gap: 12px; }}
.label {{ font-size: 16px; font-weight: 700; color: #5a6478; text-align: center; }}
.label b {{ display: inline-flex; width: 22px; height: 22px; border-radius: 50%; background: {ORANGE}; color: #fff;
            align-items: center; justify-content: center; font-size: 13px; margin-right: 8px; }}
.hub {{ height: 350px; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 8px; }}
.dark {{ background: #111; border-radius: 14px; padding: 20px 18px; width: 300px; height: 350px; color: #fff; }}
.dark h3 {{ display: flex; align-items: center; gap: 10px; font-size: 21px; font-weight: 800; }}
.dark .sub {{ font-size: 13.5px; color: #9aa3b5; margin: 3px 0 12px 32px; }}
.row {{ display: flex; align-items: center; gap: 10px; font-size: 14.5px; color: #8a93a6; padding: 5px 8px;
        border-radius: 7px; white-space: nowrap; }}
.row.seed {{ color: #fff; background: rgba(255,90,10,.25); box-shadow: inset 3px 0 0 {ORANGE}; font-weight: 700; }}
.circle {{ width: 76px; height: 76px; border-radius: 50%; border: 2px solid #1a2233; background: #fff;
           display: flex; align-items: center; justify-content: center; }}
.hubname {{ width: 130px; font-size: 13.5px; font-weight: 800; color: #1a2233; text-align: center; line-height: 1.25; }}
.hubname span {{ font-weight: 600; color: #5a6478; }}
.blue {{ background: #eef3fb; border-radius: 14px; width: 340px; height: 350px; position: relative; }}
.stack {{ position: absolute; left: 22px; top: 46px; }}
.qcard {{ position: absolute; width: 268px; background: #fff; border: 1.5px solid #c9d2e3; border-radius: 9px;
          box-shadow: 0 2px 8px rgba(26,34,51,.08); overflow: hidden; }}
.qcard.front {{ border-color: #1a2233; box-shadow: 0 4px 14px rgba(26,34,51,.14); }}
.qhead {{ display: flex; align-items: center; gap: 6px; font-size: 12.5px; font-weight: 700; color: #5a6478;
          background: #f1f3f7; padding: 6px 10px; border-bottom: 1px solid #dfe3ea; }}
.qtext {{ font-size: 16px; font-weight: 700; line-height: 1.35; padding: 12px 14px; height: 116px; }}
.qcard:not(.front) .qtext {{ color: #b5bccb; }}
.panel {{ background: #f7f8fb; border: 1.5px solid #dfe3ea; border-radius: 14px; width: 370px; height: 350px;
          padding: 14px 16px; display: flex; flex-direction: column; gap: 11px; }}
.ptitle {{ display: flex; align-items: center; gap: 10px; font-size: 14.5px; font-weight: 700; color: #5a6478;
           line-height: 1.25; }}
.mcard {{ background: #fff; border: 1.5px solid #dfe3ea; border-radius: 10px; overflow: hidden; }}
.mhead {{ font-size: 14.5px; font-weight: 800; padding: 6px 12px; background: #f1f3f7;
          border-bottom: 1px solid #dfe3ea; }}
.grow {{ display: flex; align-items: center; gap: 10px; font-size: 14px; padding: 3px 12px; white-space: nowrap; }}
.badge {{ display: inline-flex; align-items: center; justify-content: center; width: 38px; height: 23px;
          border-radius: 12px; font-size: 13px; font-weight: 800; flex: none; }}
.result {{ background: #fff; border: 1.5px solid #c9d2e3; border-radius: 14px; width: 380px; height: 350px;
           overflow: hidden; display: flex; flex-direction: column; }}
.result h3 {{ display: flex; align-items: center; gap: 10px; font-size: 21px; font-weight: 800; padding: 14px 20px;
              background: #f1f3f7; border-bottom: 1.5px solid #dfe3ea; }}
.rlist {{ padding: 10px 18px; flex: 1; }}
.rrow {{ display: flex; align-items: center; gap: 10px; font-size: 15px; padding: 4px 2px; }}
.rrow.mark {{ color: {BLUE}; font-weight: 800; }}
.rrow.dots {{ padding: 0 0 0 14px; color: #9aa3b5; font-size: 15px; line-height: 1.1; }}
.rk {{ width: 22px; color: #9aa3b5; font-weight: 700; text-align: right; }}
.rn {{ width: 200px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.bar {{ flex: 1; height: 8px; background: #edf0f5; border-radius: 4px; overflow: hidden; }}
.bar i {{ display: block; height: 100%; background: #9fb4df; border-radius: 4px; }}
.rrow.mark .bar i {{ background: {BLUE}; }}
.rs {{ width: 40px; text-align: right; font-weight: 700; }}
.foot {{ font-size: 12.5px; color: #8a93a6; padding: 0 20px 14px; }}
</style></head><body><div class="fig">

<div class="step"><div class="dark">
  <h3>{icon("db", 22, "#fff")} {TASK}</h3>
  <div class="sub">5,183 docs · no queries, no labels</div>
  {corpus_html}
</div><div class="label"><b>1</b>Your documents</div></div>

<div class="hub">{FAN}</div>
<div class="hub"><div class="circle">{icon("pen", 32, ORANGE, 2.2)}</div>
  <div class="hubname">Query writer<br><span>DeepSeek-V4-Flash</span></div></div>
<div class="hub">{ARROW}</div>

<div class="step"><div class="blue"><div class="stack">{cards}</div></div>
  <div class="label"><b>2</b>Write queries from them</div></div>

<div class="hub">{ARROW}</div>

<div class="step"><div class="panel">
  <div class="ptitle"><div class="circle" style="width:40px;height:40px;border-width:1.5px;flex:none">
    {icon("scale", 21, BLUE, 2)}</div>Each model retrieves its top 10;<br>the judge (JEV-27B) grades each 0–5</div>
  {model_html}
</div><div class="label"><b>3</b>Retrieve and judge</div></div>

<div class="hub">{ARROW}</div>

<div class="step"><div class="result">
  <h3>{icon("rank", 22, BLUE)} Model ranking</h3>
  <div class="rlist">{rank_rows}</div>
  <div class="foot">score: nDCG@10 with the judge's grades, mean of 100 queries</div>
</div><div class="label"><b>4</b>Rank the models</div></div>

</div></body></html>"""

(FIGURES / "pipeline.html").write_text(HTML)
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 2000, "height": 460}, device_scale_factor=2)
    page.set_content(HTML, wait_until="networkidle")
    box = page.locator("body").bounding_box()
    width, height = box["width"], box["height"]
    page.set_viewport_size({"width": int(width) + 1, "height": int(height) + 1})
    page.pdf(path=str(FIGURES / "pipeline.pdf"), width=f"{width}px", height=f"{height}px", print_background=True,
             page_ranges="1")
    page.screenshot(path=str(FIGURES / "pipeline.png"), clip={"x": 0, "y": 0, "width": width, "height": height})
    browser.close()
print("wrote", FIGURES / "pipeline.pdf")
