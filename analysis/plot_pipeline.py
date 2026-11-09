"""Figure: how MTEB Gym works, step by step, on real data from SciFact (generated queries, JEV-27B).

Drawn as HTML and printed to PDF with headless Chromium (playwright), so text and lines stay vector. Flat on purpose:
white boxes, thin borders, one accent color.
1. Documents: the task's corpus and description; the three documents sampled for query q50 are marked.
2. Write queries: the query writer turns them into one query; repeated to 100 queries.
3. Retrieve: every model returns its top 10 for every query.
4. Judge: the judge grades each retrieved (query, document) pair 0-5.
5. Score and rank: nDCG@10 with the judge's grades, per query, averaged over the 100 queries."""

import json

from huggingface_hub import hf_hub_download
from playwright.sync_api import sync_playwright

from common import GENERATOR, GYM_RUNS, JUDGE, judged, ndcg, ratings, records
from mteb_gym import corpus
from plots import BLUE, FIGURES

TASK, QID = "SciFact", "q50"
EXAMPLE = ["Qwen/Qwen3-Embedding-8B", "sentence-transformers/all-MiniLM-L6-v2"]
FILLER = ["16737163", "7227763", "13312471"]  # unsampled documents shown in the corpus box

R = records()
rec = R[(JUDGE, "pointwise", GENERATOR)][TASK]
models, _, top, grades, _ = judged(rec)
docs = corpus.load(TASK).docs
query = None
for line in open(hf_hub_download(GYM_RUNS, "tables/queries/queries.jsonl", repo_type="dataset")):
    q = json.loads(line)
    if q["task"] == TASK and q["qid"] == QID and q["generator"] != "original":
        query = q
board = sorted(ratings(rec).items(), key=lambda x: -x[1])
place = {m: i + 1 for i, (m, _) in enumerate(board)}
per_query = {m: 100 * ndcg(top[m][QID], grades[QID], grades[QID].values(), 10) for m in EXAMPLE}


def title(d, n=30):
    text = docs[d] if isinstance(docs[d], str) else json.dumps(docs[d])
    return text[:n].rsplit(" ", 1)[0] + "…" if len(text) > n else text


def short(m):
    return m.split("/")[-1].replace("baseline-bm25s", "BM25").replace("-embedding-english-r2", "-embedding-r2")


seeds = query["seed_doc_ids"]
corpus_rows = [(seeds[0], True), (FILLER[0], False), (seeds[1], True), (FILLER[1], False), (seeds[2], True),
               (FILLER[2], False)]
corpus_html = "".join(f'<div class="doc {"on" if s else ""}"><span class="dot"></span>{title(d)}</div>'
                      for d, s in corpus_rows)

retrieve_html = "".join(f'<div class="mrow"><span>{short(m)}</span><span class="muted">top 10</span></div>'
                        for m in [EXAMPLE[0], "ibm-granite/granite-embedding-english-r2", "mteb/baseline-bm25s"])
retrieve_html += '<div class="mrow muted">… 24 more models</div>'
retrieve_html += f'<div class="mrow"><span>{short(EXAMPLE[1])}</span><span class="muted">top 10</span></div>'

judge_html = ""
for m in EXAMPLE:
    rows = "".join(f'<tr><td class="rk">{i + 1}</td><td>{title(d, 27)}</td><td class="g">{grades[QID][d]:.1f}</td></tr>'
                   for i, d in enumerate(top[m][QID][:3]))
    judge_html += (f'<div class="sub">{short(m)}</div><table>{rows}'
                   '<tr><td></td><td class="muted">… ranks 4–10</td><td></td></tr></table>')

rank_html, last = "", 0
for r in sorted({1, 2, 3, place[EXAMPLE[0]], place[EXAMPLE[1]]}):
    if r > last + 1:
        rank_html += '<tr><td class="rk">⋮</td><td></td><td></td></tr>'
    m, s = board[r - 1]
    rank_html += (f'<tr class="{"hl" if m in EXAMPLE else ""}"><td class="rk">{r}</td><td>{short(m)}</td>'
                  f'<td class="g">{s:.1f}</td></tr>')
    last = r

ARROW = ('<div class="arrow"><svg width="34" height="16" viewBox="0 0 34 16"><path d="M1 8h28M23 2l7 6-7 6" '
         'stroke="#1a2233" stroke-width="1.8" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg></div>')


def box(n, name, what, body, out):
    return (f'<div class="box"><div class="head"><span class="n">{n}</span>{name}</div><div class="what">{what}</div>'
            f'<div class="body">{body}</div><div class="out"><span>→</span>{out}</div></div>')


HTML = f"""<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
* {{ box-sizing: border-box; margin: 0; }}
body {{ font-family: Inter, sans-serif; color: #1a2233; background: #fff; display: inline-block; padding: 12px; }}
.fig {{ display: flex; align-items: stretch; }}
.box {{ width: 292px; border: 1.4px solid #c5ccd8; border-radius: 10px; display: flex; flex-direction: column; }}
.head {{ display: flex; align-items: center; gap: 9px; font-size: 17px; font-weight: 700; padding: 12px 14px 4px; }}
.n {{ width: 22px; height: 22px; border-radius: 50%; border: 1.6px solid {BLUE}; color: {BLUE}; font-size: 12.5px;
      display: inline-flex; align-items: center; justify-content: center; font-weight: 700; flex: none; }}
.what {{ font-size: 12.5px; color: #5a6478; padding: 0 14px 10px 45px; line-height: 1.35; min-height: 44px; }}
.body {{ flex: 1; padding: 10px 14px; border-top: 1px solid #e3e7ee; font-size: 13px; }}
.out {{ display: flex; gap: 6px; font-size: 12.5px; font-weight: 600; padding: 9px 14px; border-top: 1px solid #e3e7ee;
        line-height: 1.35; }}
.out span {{ color: {BLUE}; }}
.arrow {{ display: flex; align-items: center; padding: 0 8px; }}
.label {{ font-size: 11.5px; font-weight: 600; color: #8a93a6; text-transform: uppercase; letter-spacing: .5px;
          margin-bottom: 6px; }}
.doc {{ display: flex; align-items: center; gap: 8px; padding: 3.5px 0; color: #8a93a6; white-space: nowrap; }}
.doc .dot {{ width: 8px; height: 8px; border-radius: 50%; border: 1.4px solid #c5ccd8; flex: none; }}
.doc.on {{ color: #1a2233; font-weight: 600; }}
.doc.on .dot {{ background: {BLUE}; border-color: {BLUE}; }}
.task {{ margin-top: 10px; font-size: 12.5px; line-height: 1.35; }}
.query {{ border: 1.4px solid #1a2233; border-radius: 8px; padding: 10px 12px; font-size: 14px; font-weight: 600;
          line-height: 1.38; margin: 4px 0 10px; }}
.small {{ font-size: 12.5px; color: #5a6478; line-height: 1.4; }}
.mrow {{ display: flex; justify-content: space-between; padding: 5px 0; border-bottom: 1px solid #eef1f5; }}
.mrow:last-child {{ border-bottom: none; }}
.muted {{ color: #8a93a6; }}
.sub {{ font-weight: 700; margin: 2px 0; }}
table {{ width: 100%; border-collapse: collapse; margin-bottom: 8px; }}
td {{ padding: 2.5px 0; white-space: nowrap; }}
td.rk {{ width: 20px; color: #8a93a6; }}
td.g {{ text-align: right; font-weight: 700; font-variant-numeric: tabular-nums; }}
tr.hl td {{ color: {BLUE}; font-weight: 700; }}
.formula {{ font-size: 13px; line-height: 1.5; margin-bottom: 10px; }}
.formula b {{ color: {BLUE}; }}
</style></head><body><div class="fig">
{box(1, "Documents", "A task's corpus and its description. No queries, no relevance labels.",
     f'<div class="label">{TASK} · 5,183 documents</div>{corpus_html}'
     f'<div class="task"><span class="muted">Task:</span> {rec["config"]["task_instruction"]}.</div>',
     "Sample 3 documents")}
{ARROW}
{box(2, "Write queries", "An LLM writes one query from the 3 documents and the task description.",
     f'<div class="label">Query writer: DeepSeek-V4-Flash</div><div class="query">{query["text"]}</div>'
     '<div class="small">Too short or too long queries are dropped.</div>',
     "100 queries per task")}
{ARROW}
{box(3, "Retrieve", "Every embedding model, run as in mteb, returns its top 10 documents for each query.",
     f'<div class="label">28 models</div>{retrieve_html}',
     f"Pool: every document any model returned ({len(grades[QID])} for this query)")}
{ARROW}
{box(4, "Judge", "An LLM grades every (query, document) pair in the pool from 0 to 5.",
     f'<div class="label">Judge: JEV-27B</div>{judge_html}',
     f"{rec['diagnostics']['n_grades']:,} grades for {TASK}")}
{ARROW}
{box(5, "Score and rank", "Each model gets nDCG@10 with the judge's grades as relevance, averaged over the queries.",
     '<div class="label">This query</div><table>' + "".join(
         f'<tr><td></td><td>{short(m)}</td><td class="g">{per_query[m]:.1f}</td></tr>' for m in EXAMPLE) + '</table>'
     f'<div class="label">Mean over 100 queries</div><table>{rank_html}</table>',
     "A ranking of the models for this task")}
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
