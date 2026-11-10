"""Figure 1: MTEB Gym on one real SciFact query, sized for the top of page 1 (HTML printed to PDF/PNG with Chromium).

Given a corpus, an LLM writes queries from sampled documents; every embedding model retrieves its top 10 and an LLM
judge grades each result 0-5; the models are ranked by nDCG@10 with those grades, which matches MTEB's ranking.
Two rows, so that text prints at 7pt or more at full text width. Icons: Microsoft Fluent Emoji (MIT) and Lucide
(ISC); logos from Hugging Face."""

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



BLUE, MUTED = "#1f5bd6", "#5a6f99"
GRAIN = ("url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='160' height='160'>"
         "<filter id='n'><feTurbulence type='fractalNoise' baseFrequency='.85' numOctaves='2' stitchTiles='stitch'/>"
         "<feColorMatrix values='0 0 0 0 1  0 0 0 0 1  0 0 0 0 1  0 0 0 .07 0'/></filter>"
         "<rect width='100%' height='100%' filter='url(%23n)'/></svg>\")")


def robot(badge, x, y):
    """An LLM step's actor, as SWE-Sweep draws its agent: one robot, with what it does as a small badge."""
    return (f'<div class="bot" style="left:{x}px;top:{y}px">{emoji("robot", 80)}'
            f'<span>{emoji(badge, 38)}</span></div>')


def head(n, text, sub):
    return f'<div class="hd"><i>{n}</i><div>{text}<small>{sub}</small></div></div>'


def qcard(q, main=False):
    return (f'<div class="q{" main" if main else ""}">{emoji("magnifying_glass_tilted_right", 34)}'
            f'<span>{q["text"]}</span></div>')


def grade(g):
    bg, fg = ("#e3ecfd", BLUE) if g >= 2 else ("#fde4df", "#d1342b")
    return f'<b class="g" style="background:{bg};color:{fg}">{g:.1f}</b>'


def meter(g):
    """A grade as five segments, filled to the grade."""
    color = BLUE if g >= 2 else "#d1342b"
    return "".join(f'<i style="background:{color};opacity:{max(0, min(1, g - k)) * .85 + .15:.2f}"></i>' for k in range(5))


def results(m, tag):
    rows = "".join(f'<div class="r">{line("file-text", 26, "#8fb4f0")}<span class="dt">{title(d, 26)}</span>{grade(grades[QID][d])}</div>'
                   for i, d in enumerate(top[m][QID][:3]))
    return (f'<div class="list"><div class="lh"><i class="av">{tag}</i>Model {tag}<small>grade 0–5</small></div>{rows}'
            f'<div class="more">… 7 more</div></div>')


def dashed(path):
    return f'<path d="{path}" marker-end="url(#h)"/>'


FILES = "".join(f'<div class="file{" seed" if d == SEED else ""}">{line("file-text", 32, "#9fc0f5")}<span>{title(d, 26)}</span></div>'
                for d in [SEED, *[d for d in list(docs)[:4] if d != SEED][:2]])
MEDALS = ["1st_place_medal", "2nd_place_medal", "3rd_place_medal"]
SHOWN = 7


BOARD = (f'<div class="tr th"><span class="ch">{emoji("person_lifting_weights", 64)}<small>Gym</small></span><span></span>'
         f'<span class="ch">{logo("mteb", 58)}<small>MTEB</small></span></div>'
         + "".join(f'<div class="tr"><span><i class="rk gy">{i + 1}</i></span>'
                   f'<span class="nm">{logo(m.split("/")[0], 40)}<em>{short(m)}<small>gym score {score[m]:.1f}</small></em></span>'
                   f'<span><i class="rk mt">{mteb.index(m) + 1}</i></span></div>'
                   for i, m in enumerate(gym[:SHOWN])))

def scatter(w=448, h=450):
    """Each model's rank by the gym against its rank by MTEB's labels; the top 3 circled."""
    pad_l, pad_b, pad_t, pad_r = 44, 40, 18, 18
    x = lambda r: pad_l + (len(models) - r) / (len(models) - 1) * (w - pad_l - pad_r)
    y = lambda r: pad_t + (r - 1) / (len(models) - 1) * (h - pad_t - pad_b)
    dots = "".join(f'<circle cx="{x(mteb.index(m) + 1):.1f}" cy="{y(gym.index(m) + 1):.1f}" r="7" fill="{BLUE}"/>'
                   + (f'<circle cx="{x(mteb.index(m) + 1):.1f}" cy="{y(gym.index(m) + 1):.1f}" r="13" fill="none" '
                      f'stroke="{ORANGE}" stroke-width="3.5"/>' if gym.index(m) < 3 else "") for m in models)
    ticks = "".join(f'<text x="{x(r):.1f}" y="{h - 12}" text-anchor="middle">{r}</text>'
                    f'<text x="{pad_l - 12}" y="{y(r) + 7:.1f}" text-anchor="end">{r}</text>' for r in (1, 14, 28))
    return (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" font-family="Nunito" font-size="22" font-weight="800" fill="{MUTED}">'
            f'<line x1="{x(28)}" y1="{y(28)}" x2="{x(1)}" y2="{y(1)}" stroke="#b8c5dd" stroke-width="2.5" stroke-dasharray="8 7"/>'
            f'<line x1="{pad_l}" y1="{h - pad_b}" x2="{w - pad_r}" y2="{h - pad_b}" stroke="#d3dcea" stroke-width="2"/>'
            f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{h - pad_b}" stroke="#d3dcea" stroke-width="2"/>'
            f'{ticks}{dots}</svg>'
            f'<div style="white-space:nowrap;display:flex;justify-content:space-between;font-size:21px;font-weight:800;color:{MUTED};padding-left:{pad_l}px">'
            f'<span>rank by MTEB &#8594;</span><span>rank by gym &#8593;</span></div>')


SCATTER = scatter()
SAME3 = ('<div class="same3"><div>Top 3 by both</div><ol>'
         + "".join(f'<li><i>{i + 1}</i>{short(m)}</li>' for i, m in enumerate(gym[:3])) + '</ol></div>')
W, H = 2120, 1000
ARROWS = f"""<svg class="lines" width="{W}" height="{H}" viewBox="0 0 {W} {H}" fill="none" stroke="{NAVY}" stroke-width="3.5"
  stroke-dasharray="10 9" stroke-linecap="round">
  <defs><marker id="h" viewBox="0 0 12 12" refX="9" refY="6" markerWidth="8" markerHeight="8" orient="auto">
    <path d="M2 2L10 6L2 10" fill="none" stroke="{NAVY}" stroke-width="2" stroke-dasharray="none"/></marker></defs>
  {dashed("M560 168 C 580 150, 596 156, 618 168")}{dashed("M560 280 C 580 320, 596 340, 618 346")}
  <path d="M960 474 L 960 530" stroke-dasharray="none" stroke-width="4.5" marker-end="url(#h)"/>
  {dashed("M1384 716 C 1368 690, 1330 680, 1300 680")}{dashed("M1384 828 C 1368 856, 1330 866, 1300 866")}
  <path d="M1458 772 L 1474 772" stroke-dasharray="none" stroke-width="4.5" marker-end="url(#h)"/>
</svg>"""

HTML = f"""<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Nunito:wght@600;700;800;900&display=swap" rel="stylesheet">
<style>
* {{ box-sizing: border-box; margin: 0; }}
body {{ font-family: Nunito, sans-serif; color: {NAVY}; background: #fff; width: {W}px; height: {H}px; position: relative;
        overflow: hidden; }}
body > div {{ position: absolute; }}
.lines {{ position: absolute; left: 0; top: 0; z-index: 1; }}
.panel {{ border-radius: 22px; background: {PANEL}; background-image: {GRAIN}; padding: 24px 30px; }}
.hd {{ display: flex; align-items: flex-start; gap: 14px; font-size: 36px; font-weight: 900; line-height: 1.1; }}
.hd i {{ flex: none; font-style: normal; width: 46px; height: 46px; border-radius: 50%; background: {NAVY}; color: #fff;
         font-size: 26px; display: flex; align-items: center; justify-content: center; }}
.hd small {{ display: block; font-size: 26px; font-weight: 800; color: {MUTED}; margin-top: 4px; }}
.corpus {{ left: 30px; top: 30px; width: 500px; height: 352px; border-radius: 18px; background: {NAVY};
           background-image: {GRAIN}; color: #fff; padding: 24px 30px; box-shadow: 0 8px 20px rgba(18,58,122,.18); }}
.corpus .hd small {{ color: #b9cdf0; }}
.corpus .hd i {{ background: #fff; color: {NAVY}; }}
.files {{ margin-top: 18px; border-top: 1.5px solid rgba(255,255,255,.25); padding-top: 8px; }}
.file {{ display: flex; align-items: center; gap: 14px; font-size: 29px; font-weight: 700; padding: 11px 10px; margin: 0 -10px;
         white-space: nowrap; }}
.file svg {{ flex: none; }}
.file span {{ overflow: hidden; text-overflow: ellipsis; }}
.file.seed {{ background: rgba(143,180,240,.2); border-radius: 10px; }}
.bot {{ z-index: 3; width: 128px; height: 128px; border-radius: 50%; background: #fff; border: 3.5px solid {NAVY};
        display: flex; align-items: center; justify-content: center;
        box-shadow: 0 0 0 10px rgba(18,58,122,.10), 0 8px 18px rgba(18,58,122,.2); }}
.bot span {{ position: absolute; right: -12px; bottom: -8px; width: 56px; height: 56px; border-radius: 50%; background: #fff;
             border: 3px solid {NAVY}; display: flex; align-items: center; justify-content: center; }}
.queries {{ left: 600px; top: 30px; width: 720px; height: 444px; }}
.q {{ display: flex; gap: 14px; align-items: flex-start; font-size: 30px; font-weight: 800; line-height: 1.25; background: #fff;
      border-radius: 14px; padding: 16px 20px; border: 1.5px solid #d3dcea; box-shadow: 0 8px 20px rgba(18,58,122,.12);
      margin-top: 18px; }}
.q img {{ flex: none; margin-top: 2px; }}
.q.main {{ border: 3.5px solid {NAVY}; margin-left: 28px; }}
.via {{ left: 986px; top: 474px; font-size: 26px; font-weight: 900; line-height: 1.2; }}
.via small {{ display: block; font-size: 24px; color: {MUTED}; font-weight: 800; }}
.judging {{ left: 600px; top: 540px; width: 720px; height: 430px; }}
.lists {{ display: flex; gap: 26px; margin-top: 20px; position: relative; }}
.list {{ flex: 1; background: #fff; border-radius: 14px; padding: 12px 18px 6px; border: 1.5px solid #d3dcea;
         box-shadow: 0 8px 20px rgba(18,58,122,.12); min-width: 0; }}
.lh {{ display: flex; align-items: center; gap: 10px; font-size: 29px; font-weight: 900; padding-bottom: 8px;
       border-bottom: 1.5px solid #e3e8f0; }}
.lh small {{ margin-left: auto; font-size: 22px; color: {MUTED}; font-weight: 800; }}
.av {{ font-style: normal; width: 34px; height: 34px; border-radius: 50%; background: {NAVY}; color: #fff; font-size: 19px;
       font-weight: 900; display: flex; align-items: center; justify-content: center; flex: none; }}
.r {{ display: flex; align-items: center; gap: 10px; padding: 10px 0; border-bottom: 1px solid #f0f2f6; font-size: 25px; font-weight: 800; }}
.r svg {{ flex: none; }}
.r span {{ flex: 1; min-width: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.g {{ flex: none; width: 62px; text-align: center; border-radius: 9px; padding: 2px 0; font-size: 26px; font-weight: 900; }}
.more {{ font-size: 21px; font-weight: 800; color: #8a9bbd; padding: 6px 0 4px 36px; }}
.vs {{ position: absolute; left: 50%; top: 50%; transform: translate(-50%, -50%); z-index: 3; width: 48px; height: 48px;
       border-radius: 50%; background: {NAVY}; color: #fff; font-size: 19px; font-weight: 900; display: flex;
       align-items: center; justify-content: center; box-shadow: 0 0 0 5px {PANEL}; }}
.board {{ left: 1484px; top: 30px; width: 608px; height: 940px; background: #fff; border-radius: 22px; border: 1.5px solid #d3dcea;
          box-shadow: 0 8px 20px rgba(18,58,122,.10); padding: 24px 24px; display: flex; flex-direction: column; }}
.board .hd {{ font-size: 34px; }}
.table {{ margin-top: 22px; }}
.table {{ display: flex; flex-direction: column; gap: 8px; }}
.tr {{ display: grid; grid-template-columns: 92px 1fr 92px; align-items: center; height: 82px; border-radius: 14px;
       background: #f3f6fb; font-size: 25px; font-weight: 900; }}
.tr > :first-child, .tr > :last-child {{ display: flex; justify-content: center; align-items: center; gap: 6px; }}
.tr.th {{ height: 104px; background: none; }}
.rk {{ font-style: normal; font-size: 38px; font-weight: 900; font-variant-numeric: tabular-nums; }}
.rk.gy {{ color: {NAVY}; }}
.rk.mt {{ color: {ORANGE}; }}
.nm {{ display: flex; align-items: center; gap: 12px; min-width: 0; padding-left: 4px; }}
.nm img {{ flex: none; }}
.nm em {{ font-style: normal; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; line-height: 1.15; }}
.nm small {{ display: block; font-size: 19px; font-weight: 800; color: {MUTED}; }}
.r em {{ font-style: normal; width: 22px; flex: none; color: #8a9bbd; font-weight: 900; font-size: 24px; }}
.bar {{ flex: 1; height: 14px; border-radius: 7px; background: #edf1f7; overflow: hidden; }}
.bar i {{ display: block; height: 100%; border-radius: 7px; }}
.ch {{ flex-direction: column; gap: 4px !important; }}
.ch small {{ font-size: 21px; font-weight: 900; line-height: 1.05; text-align: center; color: {MUTED}; }}
.dt {{ flex: 1; min-width: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-size: 22px; }}
.rest {{ font-size: 22px; font-weight: 800; color: #8a9bbd; padding: 16px 0 0 52px; }}
.call {{ left: 30px; top: 406px; width: 500px; height: 564px; border-radius: 22px; background: #fff; border: 1.5px solid #d3dcea;
         box-shadow: 0 8px 20px rgba(18,58,122,.10); padding: 22px 26px 14px; display: flex; flex-direction: column; }}
.call .t {{ white-space: nowrap; display: flex; gap: 16px; align-items: baseline; justify-content: space-between; font-size: 30px; font-weight: 900; }}
.call .t b {{ font-size: 40px; color: {BLUE}; }}
.call svg {{ margin-top: 6px; flex: none; display: block; width: 448px; height: 450px; }}
.plot {{ position: relative; }}
.same3 {{ position: absolute; right: 16px; bottom: 80px; background: #fff; border: 2.5px solid {ORANGE}; border-radius: 12px;
          padding: 6px 12px 8px; }}
.same3 > div {{ font-size: 18px; font-weight: 900; color: {ORANGE}; margin-bottom: 4px; }}
.same3 ol {{ list-style: none; padding: 0; display: flex; flex-direction: column; gap: 2px; }}
.same3 li {{ display: flex; align-items: center; gap: 6px; font-size: 17px; font-weight: 800; }}
.same3 li i {{ font-style: normal; width: 16px; color: {MUTED}; font-weight: 900; }}
</style></head><body>
{ARROWS}
<div class="corpus">{head(1, f"{TASK} corpus", "no queries, no relevance labels")}<div class="files">{FILES}</div></div>
{robot("writing_hand", 482, 160)}
<div class="panel queries">{head(2, "An LLM writes queries", "from sampled documents, 100 per task")}
  {qcard(query, main=True)}{qcard(queries[OTHER_QIDS[0]])}</div>
<div class="via">28 embedding models<small>retrieve their top 10</small></div>
<div class="panel judging">{head(3, "A judge grades the results", "pointwise: each result 0 to 5<br>pairwise: which top 10 is better, A or B")}
  <div class="lists">{results(STRONG, "A")}<div class="vs">vs</div>{results(WEAK, "B")}</div></div>
{robot("balance_scale", 1330, 708)}
<div class="board">{head(4, "Gym ranking", "by nDCG@10 with the grades")}
  <div class="table">{BOARD}</div>
  <div class="rest">… {len(models) - SHOWN} more</div></div>
<div class="call"><div class="t">Agrees with MTEB<b>ρ = {RHO:.2f}</b></div><div class="plot">{SCATTER}{SAME3}</div></div>
</body></html>"""

if __name__ == "__main__":
    from plots import render
    render(HTML, "figure1", W, H)
