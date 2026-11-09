"""Figure: how MTEB Gym works, end to end, on real data from SciFact (generated queries, JEV-27B, pointwise).

Drawn as HTML (organization logos from Hugging Face, in figures/logos) and printed to PDF with headless Chromium.
Row 1: (1) the corpus, (2) the query writer's prompt and the query it wrote, (3) each model's retrieval.
Row 2: (4) the judge's output for two documents, (5) the grades and the score, (6) the task's ranking."""

import base64
import glob
import json

from huggingface_hub import hf_hub_download
from playwright.sync_api import sync_playwright

from common import GENERATOR, GYM_RUNS, JUDGE, agreement, judged, ndcg, ratings, records, runs_folder
from mteb_gym import corpus
from plots import BLUE, FIGURES, ORANGE

TASK, QID = "SciFact", "q50"
ROWS = ["Qwen/Qwen3-Embedding-8B", "ibm-granite/granite-embedding-english-r2", "lightonai/DenseOn",
        "sentence-transformers/all-MiniLM-L6-v2", "mteb/baseline-bm25s"]
STRONG, WEAK = ROWS[0], ROWS[3]
FILLER = ["16737163", "7227763", "13312471", "2374637"]
SCALE = ["#eef2f9", "#cbd7ee", "#9fb4df", "#6b8bcd", "#3a64b8", BLUE]  # grade 0..5

R = records()
rec = R[(JUDGE, "pointwise", GENERATOR)][TASK]
cfg = rec["config"]
_, _, top, grades, _ = judged(rec)
docs = corpus.load(TASK).docs
queries = {}
for line in open(hf_hub_download(GYM_RUNS, "tables/queries/queries.jsonl", repo_type="dataset")):
    q = json.loads(line)
    if q["task"] == TASK and q["query_set"] == rec["config"]["query_set"].rsplit("-", 1)[0]:
        queries[q["qid"]] = q
query = queries[QID]
scores = {}
for line in open(hf_hub_download(GYM_RUNS, f"tables/predictions/{TASK}.jsonl", repo_type="dataset")):
    p = json.loads(line)
    if p["query_set"] == cfg["query_set"].rsplit("-", 1)[0] and p["qid"] == QID:
        scores[p["model"]] = dict(zip(p["doc_ids"], p["scores"]))
probs = {}
folder = runs_folder(f"grades/{TASK}/autotrust_JEV-27B-*.jsonl")
for path in glob.glob(f"{folder}/grades/{TASK}/autotrust_JEV-27B-*.jsonl"):
    for line in open(path):
        if f'"qid": "{QID}"' in line:
            g = json.loads(line)
            if g["query_set"] == cfg["query_set"]:
                probs[g["doc_id"]] = g["probabilities"]
board = sorted(ratings(rec).items(), key=lambda x: -x[1])
place = {m: i + 1 for i, (m, _) in enumerate(board)}
per_query = {m: 100 * ndcg(top[m][QID], grades[QID], grades[QID].values(), 10) for m in ROWS}
rho = agreement(ratings(rec), TASK)


def img(name, size, radius=6):
    data = base64.b64encode((FIGURES / "logos" / f"{name}.png").read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}" width="{size}" height="{size}" style="border-radius:{radius}px">'


def mlogo(m, size=20):
    return img(m.split("/")[0], size)


def title(d, n=28):
    text = docs[d] if isinstance(docs[d], str) else json.dumps(docs[d])
    return text[:n].rsplit(" ", 1)[0] + "…" if len(text) > n else text


def short(m):
    return m.split("/")[-1].replace("baseline-bm25s", "BM25").replace("-embedding-english-r2", "-embedding-r2")


ICON = {
    "file": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/>',
    "check": '<path d="M20 6 9 17l-5-5"/>',
}


def icon(name, size=18, color="currentColor", width=2):
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" '
            f'stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round">{ICON[name]}</svg>')


def head(n, name, logo_html="", sub=""):
    return (f'<div class="ph"><span class="num">{n}</span><span class="pt">{name}</span>'
            f'<span class="who">{logo_html}{sub}</span></div>')


seeds = query["seed_doc_ids"]
corpus_rows = [(seeds[0], 1), (FILLER[0], 0), (seeds[1], 2), (FILLER[1], 0), (FILLER[2], 0), (seeds[2], 3),
               (FILLER[3], 0)]
corpus_html = "".join(
    f'<div class="doc {"on" if s else ""}">{icon("file", 15, "#fff" if s else "#7d8597", 1.8)}<span>{title(d)}</span>'
    f'{f"<b>{s}</b>" if s else ""}</div>' for d, s in corpus_rows)
prompt_docs = "".join(f'<div class="pd"><b>[{i + 1}]</b> {title(d, 34)}</div>' for i, d in enumerate(seeds))


def retrieved(m):
    return "".join(f'<div class="rrow"><span class="rk">{i + 1}</span><span class="rt">{title(d, 34)}</span>'
                   f'<span class="sc">{scores[m][d]:.3f}</span></div>' for i, d in enumerate(top[m][QID][:5]))


tabs = "".join(f'<span class="tab {"on" if m == STRONG else ""}">{mlogo(m, 18)}</span>' for m in ROWS) + \
    '<span class="tab more">+23</span>'


def judge_case(m, rank):
    d = top[m][QID][rank]
    bars = "".join(f'<div class="bcol"><div class="bar" style="height:{3 + 42 * probs[d][str(g)]:.0f}px;'
                   f'background:{SCALE[g]}"></div><span>{g}</span></div>' for g in range(6))
    return (f'<div class="jcase"><div class="jdoc">{mlogo(m, 15)}<span>{short(m)}, rank {rank + 1}</span></div>'
            f'<div class="jtitle">{title(d, 38)}</div><div class="jrow"><div class="dist">{bars}</div>'
            f'<div class="eg"><span>grade</span><b>{grades[QID][d]:.2f}</b></div></div></div>')


def squares(m):
    return "".join(f'<span class="sq" style="background:{SCALE[min(int(round(grades[QID].get(d, 0))), 5)]}"></span>'
                   for d in top[m][QID][:10])


grid = "".join(f'<div class="grow">{mlogo(m, 18)}<span class="gname">{short(m)}</span>'
               f'<span class="cells">{squares(m)}</span><span class="nd">{per_query[m]:.1f}</span></div>' for m in ROWS)
legend = "".join(f'<span class="lg"><span class="sq" style="background:{c}"></span>{i}</span>'
                 for i, c in enumerate(SCALE))

rank_rows, last = "", 0
for r in sorted({1, 2, 3, place[STRONG], place[WEAK], place[ROWS[4]]}):
    if r > last + 1:
        rank_rows += '<div class="krow dots">⋮</div>'
    m, s = board[r - 1]
    rank_rows += (f'<div class="krow"><span class="kk">{r}</span>{mlogo(m, 20)}<span class="kn">{short(m)}</span>'
                  f'<span class="ks">{s:.1f}</span></div>')
    last = r

ARROW = ('<div class="arr"><svg width="34" height="18" viewBox="0 0 34 18"><path d="M2 9h26M23 3l6 6-6 6" '
         'stroke="#1a2233" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg></div>')
W1, W2, W3, GAP = 300, 440, 440, 52
TOTAL = W1 + W2 + W3 + 2 * GAP
X_FROM, X_TO = W1 + W2 + 2 * GAP + W3 // 2, W1 // 2
LINK = (f'<svg width="{TOTAL}" height="44" viewBox="0 0 {TOTAL} 44" style="display:block">'
        f'<path d="M{X_FROM} 0 V16 Q{X_FROM} 22 {X_FROM - 6} 22 H{X_TO + 6} Q{X_TO} 22 {X_TO} 28 V40" '
        f'stroke="#1a2233" stroke-width="2" fill="none"/><path d="M{X_TO - 6} 33 L{X_TO} 41 L{X_TO + 6} 33" '
        f'stroke="#1a2233" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>')

HTML = f"""<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Nunito:wght@500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
* {{ box-sizing: border-box; margin: 0; }}
body {{ font-family: Nunito, sans-serif; color: #1a2233; background: #fff; display: inline-block; padding: 14px; }}
.row {{ display: flex; align-items: stretch; }}
.arr {{ display: flex; align-items: center; justify-content: center; width: {GAP}px; }}
.panel {{ border: 1.5px solid #c9d2e3; border-radius: 14px; background: #fff; height: 352px; overflow: hidden;
          display: flex; flex-direction: column; }}
.ph {{ display: flex; align-items: center; gap: 9px; padding: 9px 14px; background: #f1f3f7;
       border-bottom: 1.5px solid #dfe3ea; }}
.num {{ width: 22px; height: 22px; border-radius: 50%; background: #1a2233; color: #fff; font-size: 12.5px;
        font-weight: 800; display: inline-flex; align-items: center; justify-content: center; flex: none; }}
.pt {{ font-size: 17px; font-weight: 800; }}
.who {{ margin-left: auto; display: flex; align-items: center; gap: 7px; font-size: 13px; font-weight: 700;
        color: #3d4556; }}
.pb {{ padding: 9px 14px; flex: 1; display: flex; flex-direction: column; gap: 7px; font-size: 13.5px; }}
.cap {{ font-size: 11.5px; font-weight: 800; color: #8a93a6; text-transform: uppercase; letter-spacing: .5px; }}
.dark {{ background: #111; color: #fff; border: none; }}
.dark .ph {{ background: #111; border-bottom: 1px solid #2c3140; }}
.dark .num {{ background: #fff; color: #111; }}
.dark .who {{ color: #9aa3b5; }}
.doc {{ display: flex; align-items: center; gap: 8px; font-size: 13.5px; color: #8a93a6; padding: 3px 8px;
        border-radius: 7px; white-space: nowrap; }}
.doc b {{ margin-left: auto; font-size: 11.5px; background: {ORANGE}; color: #fff; border-radius: 9px; padding: 0 7px; }}
.doc.on {{ color: #fff; background: rgba(255,90,10,.26); font-weight: 700; }}
.task {{ margin-top: auto; font-size: 12.5px; color: #c9cfdb; border-top: 1px solid #2c3140; padding-top: 8px;
         line-height: 1.35; }}
.task span {{ color: #8a93a6; }}
.prompt {{ font-family: 'JetBrains Mono', monospace; font-size: 11.3px; line-height: 1.5; background: #f7f8fb;
           border: 1px solid #dfe3ea; border-radius: 8px; padding: 7px 10px; color: #3d4556; }}
.prompt .k {{ color: {BLUE}; font-weight: 500; }}
.pd {{ white-space: nowrap; overflow: hidden; text-overflow: ellipsis; padding-left: 10px; }}
.pd b {{ color: {ORANGE}; }}
.qout {{ border: 1.6px solid #1a2233; border-radius: 9px; padding: 7px 11px; font-size: 14.5px; font-weight: 800;
         line-height: 1.33; }}
.checks {{ display: flex; gap: 14px; align-items: center; font-size: 12.5px; font-weight: 700; color: #3d4556; }}
.checks span {{ display: inline-flex; align-items: center; gap: 4px; }}
.chip {{ margin-left: auto; background: #1a2233; color: #fff !important; font-size: 12.5px; font-weight: 800;
         padding: 3px 10px; border-radius: 12px; }}
.tabs {{ display: flex; gap: 6px; align-items: center; }}
.tab {{ display: inline-flex; align-items: center; justify-content: center; width: 36px; height: 30px;
        border: 1.4px solid #dfe3ea; border-radius: 8px; background: #fff; }}
.tab.on {{ border-color: #1a2233; box-shadow: inset 0 -3px 0 {BLUE}; }}
.tab.more {{ font-size: 12px; font-weight: 800; color: #5a6478; width: 42px; }}
.list {{ border: 1px solid #dfe3ea; border-radius: 9px; overflow: hidden; }}
.lhead {{ display: flex; align-items: center; gap: 7px; font-size: 13px; font-weight: 800; padding: 5px 10px;
          background: #f7f8fb; border-bottom: 1px solid #dfe3ea; }}
.lhead span {{ margin-left: auto; font-size: 11.5px; color: #8a93a6; font-weight: 700; }}
.rrow {{ display: flex; gap: 8px; align-items: center; font-size: 13px; padding: 2.5px 10px; white-space: nowrap; }}
.rk {{ width: 14px; color: #8a93a6; font-weight: 700; }}
.rt {{ flex: 1; overflow: hidden; text-overflow: ellipsis; }}
.sc {{ font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #5a6478; }}
.jcase {{ border: 1px solid #dfe3ea; border-radius: 9px; padding: 6px 10px; }}
.jdoc {{ display: flex; align-items: center; gap: 7px; font-size: 12px; font-weight: 700; color: #5a6478; }}
.jtitle {{ font-size: 13px; font-weight: 700; margin: 2px 0 3px; white-space: nowrap; }}
.jrow {{ display: flex; align-items: flex-end; gap: 12px; }}
.dist {{ display: flex; align-items: flex-end; gap: 5px; height: 60px; }}
.bcol {{ display: flex; flex-direction: column; align-items: center; gap: 2px; font-size: 11px; color: #8a93a6;
         font-weight: 700; }}
.bar {{ width: 22px; border-radius: 4px 4px 0 0; }}
.eg {{ display: flex; flex-direction: column; font-size: 11.5px; color: #8a93a6; font-weight: 700; margin-bottom: 14px; }}
.eg b {{ font-size: 23px; color: #1a2233; font-weight: 800; line-height: 1.05; }}
.note {{ font-size: 12.3px; color: #5a6478; line-height: 1.35; }}
.note b {{ color: #1a2233; }}
.grow {{ display: flex; align-items: center; gap: 8px; padding: 3px 0; }}
.gname {{ font-size: 13.5px; font-weight: 700; width: 150px; white-space: nowrap; }}
.cells {{ display: flex; gap: 3px; }}
.sq {{ width: 15px; height: 15px; border-radius: 3px; display: inline-block; }}
.nd {{ margin-left: auto; font-weight: 800; font-size: 14px; font-variant-numeric: tabular-nums; }}
.formula {{ font-size: 12.5px; color: #3d4556; background: #f7f8fb; border: 1px solid #dfe3ea; border-radius: 8px;
            padding: 7px 10px; line-height: 1.4; margin-top: auto; }}
.formula b {{ color: #1a2233; }}
.legend {{ display: flex; align-items: center; gap: 7px; font-size: 12px; color: #5a6478; font-weight: 700; }}
.lg {{ display: inline-flex; align-items: center; gap: 3px; }}
.lg .sq {{ width: 12px; height: 12px; }}
.krow {{ display: flex; align-items: center; gap: 9px; font-size: 14px; padding: 4px 0; }}
.krow.dots {{ color: #9aa3b5; padding: 0 0 0 6px; line-height: 1; }}
.kk {{ width: 20px; text-align: right; color: #8a93a6; font-weight: 700; }}
.kn {{ flex: 1; font-weight: 700; white-space: nowrap; }}
.ks {{ font-weight: 800; color: {BLUE}; font-variant-numeric: tabular-nums; }}
.eval {{ margin-top: auto; display: flex; align-items: center; gap: 10px; border-top: 1px solid #e6e9ef;
         padding-top: 8px; font-size: 12.5px; color: #3d4556; font-weight: 600; line-height: 1.3; }}
.eval b {{ font-size: 21px; color: {BLUE}; font-weight: 800; white-space: nowrap; }}
</style></head><body>

<div class="row">
  <div class="panel dark" style="width:{W1}px">
    {head(1, "Your documents", "", TASK)}
    <div class="pb">
      <div class="cap" style="color:#7d8597">5,183 docs · no queries · no labels</div>
      <div>{corpus_html}</div>
      <div class="task"><span>Task:</span> {cfg["task_instruction"]}</div>
    </div>
  </div>
  {ARROW}
  <div class="panel" style="width:{W2}px">
    {head(2, "Write queries", img("deepseek-ai", 20), "DeepSeek-V4-Flash")}
    <div class="pb">
      <div class="prompt"><span class="k">Task:</span> {cfg["task_instruction"]}<br>
        <span class="k">Documents:</span>{prompt_docs}
        <span class="k">Write</span> ONE query the task describes, not a restatement of any shown document.</div>
      <div class="qout">{query["text"]}</div>
      <div class="checks"><span>{icon("check", 15, BLUE, 2.6)}3–4,000 characters</span>
        <span>{icon("check", 15, BLUE, 2.6)}contains words</span><span class="chip">× 100 queries</span></div>
    </div>
  </div>
  {ARROW}
  <div class="panel" style="width:{W3}px">
    {head(3, "Retrieve", "", "28 models, run as in mteb")}
    <div class="pb">
      <div class="tabs">{tabs}</div>
      <div class="list"><div class="lhead">{mlogo(STRONG, 16)}{short(STRONG)}<span>top 10 · similarity</span></div>
        {retrieved(STRONG)}<div class="rrow" style="color:#8a93a6"><span class="rk"></span>… ranks 6–10</div></div>
      <div class="note">The pool for this query: every document any model returned, <b>{len(grades[QID])}</b> in all.</div>
    </div>
  </div>
</div>
{LINK}
<div class="row">
  <div class="panel" style="width:{W1}px">
    {head(4, "Judge", img("autotrust", 20), "JEV-27B")}
    <div class="pb">
      <div class="note">For every document in the pool: a probability for each grade 0–5. The grade is their mean.</div>
      {judge_case(STRONG, 1)}
      {judge_case(WEAK, 0)}
    </div>
  </div>
  {ARROW}
  <div class="panel" style="width:{W2}px">
    {head(5, "Score", "", "nDCG@10 with the judge's grades")}
    <div class="pb">
      <div class="legend"><span>each model's top 10, by grade</span>{legend}<span style="margin-left:auto">nDCG@10</span></div>
      {grid}
      <div class="formula"><b>nDCG@10</b> = DCG of the model's 10 grades ÷ DCG of the 10 best grades in the pool.
        A model's score is its mean over the <b>100 queries</b>.</div>
    </div>
  </div>
  {ARROW}
  <div class="panel" style="width:{W3}px">
    {head(6, "Rank the models", "", TASK)}
    <div class="pb">
      {rank_rows}
      <div class="eval"><b>ρ = {rho:.2f}</b><span>Spearman with MTEB's ranking from human relevance labels on {TASK}</span></div>
    </div>
  </div>
</div>
</body></html>"""

(FIGURES / "pipeline.html").write_text(HTML)
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 2000, "height": 900}, device_scale_factor=2)
    page.set_content(HTML, wait_until="networkidle")
    size = page.locator("body").bounding_box()
    width, height = size["width"], size["height"]
    page.set_viewport_size({"width": int(width) + 1, "height": int(height) + 1})
    page.pdf(path=str(FIGURES / "pipeline.pdf"), width=f"{width}px", height=f"{height}px", print_background=True,
             page_ranges="1")
    page.screenshot(path=str(FIGURES / "pipeline.png"), clip={"x": 0, "y": 0, "width": width, "height": height})
    browser.close()
print("wrote", FIGURES / "pipeline.pdf")
