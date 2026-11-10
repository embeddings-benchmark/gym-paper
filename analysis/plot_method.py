"""Method figure: MTEB Gym's mechanism with the paper's notation, laid out as a method diagram (input, a dashed group of
two stages, then two panels), on real SciFact data.

Query generation and the retrieval pool form the first group; the decision-model judge (pointwise grades, pairwise
verdicts in both orders) the second panel; scoring (nDCG@10 over the pool, Bradley-Terry for verdicts) the third."""

import numpy as np

from plot_figure1 import QID, STRONG, WEAK, RHO, docs, grades, gym, models, pair, probs, query, score, short, top
from plots import render

INK, MUTED, LINE = "#1d2433", "#5b6478", "#8f98ab"
ORANGE, BLUE, PINK, PURPLE, GREEN = "#e3925a", "#5d8fd0", "#d47a95", "#8d7fd0", "#5faa76"
TINT = {"orange": ("#fdeee2", "#e3a274"), "pink": ("#fbe6ec", "#d98aa0"), "purple": ("#eeeafb", "#9a8bd6")}
DOC_COLORS = [ORANGE, BLUE, GREEN, PURPLE, PINK, "#c9a43c"]


def page(color, size=30):
    return (f'<svg width="{size * .78:.0f}" height="{size}" viewBox="0 0 22 28" fill="#fff" stroke="{color}" stroke-width="1.9" '
            f'stroke-linejoin="round"><path d="M2 2h12l6 6v18H2z"/><path d="M14 2v6h6" fill="none"/>'
            f'<path d="M6 13h10M6 17h10M6 21h7" stroke-width="1.5"/></svg>')


def stack(colors, size=30):
    """Pages fanned like a stack of documents."""
    return ('<span class="stack">' + "".join(f'<span style="left:{i * 7}px;top:{i * 5}px">{page(c, size)}</span>'
                                             for i, c in enumerate(colors)) + "</span>")


def goal(kind, text):
    bg, border = TINT[kind]
    return f'<div class="goal" style="background:{bg};border-color:{border}">{text}</div>'


def mark(ok):
    color, path = (GREEN, "M4 9l3.5 3.5L14 5") if ok else ("#d9534f", "M5 5l8 8M13 5l-8 8")
    return (f'<svg width="20" height="20" viewBox="0 0 18 18"><rect x=".9" y=".9" width="16.2" height="16.2" rx="3" fill="#fff" '
            f'stroke="{color}" stroke-width="1.6"/><path d="{path}" fill="none" stroke="{color}" stroke-width="2.2" '
            f'stroke-linecap="round" stroke-linejoin="round"/></svg>')


def frac(num, den):
    return f'<span class="frac"><span>{num}</span><span>{den}</span></span>'


def arrow(w=34):
    return (f'<svg width="{w}" height="18" viewBox="0 0 {w} 18" fill="none" stroke="{INK}" stroke-width="2" '
            f'stroke-linecap="round" stroke-linejoin="round"><path d="M2 9H{w - 3}M{w - 9} 3l6 6-6 6"/></svg>')


def brace(h, flip=False):
    d = (f"M2 2 C 12 2, 12 6, 12 {h / 2 - 8} C 12 {h / 2 - 2}, 16 {h / 2}, 20 {h / 2} C 16 {h / 2}, 12 {h / 2 + 2}, 12 {h / 2 + 8} "
         f"C 12 {h - 6}, 12 {h - 2}, 2 {h - 2}")
    return (f'<svg width="22" height="{h}" viewBox="0 0 22 {h}" fill="none" stroke="{INK}" stroke-width="1.8" '
            f'style="{"transform:scaleX(-1)" if flip else ""}"><path d="{d}"/></svg>')


# Pool: which of six pooled documents each of four models puts in its top 10 (real lists on the query).
SHOWN = [STRONG, gym[2], WEAK, "mteb/baseline-bm25s"]
lists = {m: set(top[m][QID][:10]) for m in SHOWN}
POOL = {d for m in models for d in top[m][QID][:10]}
cols = sorted(set.union(*lists.values()), key=lambda d: (-sum(d in lists[m] for m in SHOWN), -grades[QID][d]))[:6]
GRID = (f'<div class="grid"><span></span>' + "".join(f'<span>{page(c, 24)}</span>' for c in DOC_COLORS)
        + "".join(f'<em>{short(m)}</em>' + "".join(f'<span>{mark(d in lists[m])}</span>' for d in cols) for m in SHOWN)
        + "</div>")

# Pointwise: the judge's grade distribution for one pooled document, and Model A's ten graded documents.
BARS = "".join(f'<div class="bar"><i style="height:{3 + 64 * p / max(probs):.0f}px;background:{PINK}"></i><span>{k}</span></div>'
               for k, p in enumerate(probs))
EXPECTED = sum(k * p for k, p in enumerate(probs))
A_GRADES = [grades[QID].get(d, 0) for d in top[STRONG][QID][:10]]
B_GRADES = [grades[QID].get(d, 0) for d in top[WEAK][QID][:10]]
IDEAL = sorted(grades[QID].values(), reverse=True)[:10]
dcg = lambda gs: sum(g / np.log2(r + 2) for r, g in enumerate(gs))
NDCG_A, NDCG_B = dcg(A_GRADES) / dcg(IDEAL), dcg(B_GRADES) / dcg(IDEAL)


def gradebars(gs, color):
    return ('<span class="gb">' + "".join(f'<i style="height:{2 + 34 * g / 5:.0f}px;background:{color}"></i>' for g in gs)
            + "</span>")


# Pairwise: the verdict in each presentation order, as the probability that model A's list is better.
first = pair["probabilities"][0]["System A"] + pair["probabilities"][0]["Tie"] / 2  # A shown as System A
second = pair["probabilities"][1]["System B"] + pair["probabilities"][1]["Tie"] / 2  # A shown as System B

# The resulting ranking: gym scores of the top 4 and of model B.
RANKED = gym[:4] + [WEAK]
lo = min(score[m] for m in RANKED) - 8
RANKBARS = "".join(f'<div class="rb"><em>{gym.index(m) + 1}</em><span>{short(m)}</span>'
                   f'<i style="width:{(score[m] - lo) / (max(score.values()) - lo) * 150:.0f}px;'
                   f'background:{BLUE if m != WEAK else "#c4cbd8"}"></i><b>{score[m]:.1f}</b></div>'
                   + ('<div class="rb gap">&#8942;</div>' if i == 3 else "") for i, m in enumerate(RANKED))

W, H = 1840, 668
HTML = f"""<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=STIX+Two+Text:ital,wght@0,400;0,600;0,700;1,400;1,600&display=swap" rel="stylesheet">
<style>
* {{ box-sizing: border-box; margin: 0; }}
body {{ font-family: 'STIX Two Text', serif; color: {INK}; background: #fff; width: {W}px; height: {H}px; padding: 10px 12px;
        display: flex; align-items: center; font-size: 21px; line-height: 1.2; }}
i, .it {{ font-style: italic; }}
.input {{ display: flex; flex-direction: column; align-items: center; gap: 8px; width: 120px; text-align: center; font-weight: 700;
          font-size: 22px; }}
.input small {{ display: block; font-weight: 400; font-size: 18px; color: {MUTED}; }}
.stack {{ position: relative; display: inline-block; width: 52px; height: 48px; }}
.stack > span {{ position: absolute; }}
.col {{ display: flex; flex-direction: column; align-self: stretch; }}
.col > h2 {{ font-size: 24px; font-weight: 700; text-align: center; margin-bottom: 6px; white-space: nowrap; }}
.group {{ flex: 1; border: 2px dashed {LINE}; border-radius: 16px; padding: 8px; display: flex; flex-direction: column; gap: 8px; justify-content: space-between; }}
.panel {{ flex: 1; border: 2px solid {LINE}; border-radius: 16px; padding: 10px 12px; display: flex; flex-direction: column; gap: 7px; justify-content: space-between; }}
.stage {{ border: 1.6px solid {LINE}; border-radius: 12px; padding: 6px 10px 8px; display: flex; flex-direction: column; gap: 6px;
          background: #fff; }}
.stage h3, .panel h3 {{ font-size: 21px; font-weight: 700; text-align: center; }}
.goal {{ border: 1.6px solid; border-radius: 9px; padding: 4px 10px; font-style: italic; text-align: center; font-size: 20px;
         line-height: 1.15; }}
.row {{ display: flex; align-items: center; justify-content: center; gap: 8px; }}
.note {{ font-size: 18px; color: {MUTED}; text-align: center; line-height: 1.15; }}
.eq {{ text-align: center; font-size: 21px; white-space: nowrap; }}
.frac {{ display: inline-flex; flex-direction: column; vertical-align: middle; text-align: center; }}
.frac > span:first-child {{ border-bottom: 1.4px solid {INK}; padding: 0 3px 1px; }}
.frac > span:last-child {{ padding: 1px 3px 0; }}
.box {{ border: 1.5px solid {INK}; border-radius: 8px; padding: 3px 8px; white-space: nowrap; }}
.query {{ border: 1.5px solid {INK}; border-radius: 8px; padding: 4px 8px; font-style: italic; font-size: 18px; text-align: center;
          width: 210px; line-height: 1.15; }}
.prompt {{ font-size: 18px; text-align: center; line-height: 1.15; }}
.grid {{ display: grid; grid-template-columns: 190px repeat(6, 30px); gap: 3px 4px; align-items: center; justify-content: center;
         font-size: 18px; }}
.grid em {{ font-style: normal; text-align: right; padding-right: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.grid span {{ display: flex; justify-content: center; }}
.bars {{ display: flex; align-items: flex-end; gap: 4px; height: 86px; border-bottom: 1.4px solid {LINE}; }}
.bar {{ display: flex; flex-direction: column; align-items: center; width: 20px; }}
.bar i {{ width: 16px; border-radius: 2px 2px 0 0; }}
.bar span {{ font-size: 15px; color: {MUTED}; }}
.gb {{ display: inline-flex; align-items: flex-end; gap: 2px; height: 38px; border-bottom: 1.3px solid {LINE}; vertical-align: middle; }}
.gb i {{ width: 9px; border-radius: 1.5px 1.5px 0 0; }}
table {{ border-collapse: collapse; font-size: 19px; margin: 0 auto; }}
td, th {{ padding: 2px 10px; text-align: center; border-bottom: 1px solid #dde1e8; white-space: nowrap; }}
th {{ font-weight: 600; font-size: 17px; color: {MUTED}; }}
tr:last-child td {{ border-bottom: none; }}
.rb {{ display: grid; grid-template-columns: 22px 190px 150px 44px; align-items: center; gap: 6px; font-size: 18px; }}
.rb em {{ font-style: normal; color: {MUTED}; text-align: right; }}
.rb span {{ white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.rb i {{ height: 14px; border-radius: 2px; }}
.rb b {{ font-weight: 400; }}
.rb.gap {{ display: block; padding-left: 26px; line-height: .7; color: {MUTED}; }}
.link {{ display: flex; align-items: center; padding: 0 4px; }}
</style></head><body>

<div class="input">{stack([ORANGE, BLUE, GREEN, PURPLE])}<div>Corpus<small>no queries,<br>no labels</small></div></div>
<div class="link">{brace(560)}</div>

<div class="col" style="width:470px"><h2>Queries and Retrieval</h2><div class="group">
  <div class="stage"><h3>Stage A: Query Generation</h3>
    {goal("orange", "Write queries a user of the corpus could ask")}
    <div class="row"><div class="prompt">Prompt LLM with<br>instruction <i>t</i> (optional)<br>and 3 sampled documents
      <div style="margin-top:4px">{page(ORANGE, 26)} {page(BLUE, 26)} {page(GREEN, 26)}</div></div>{arrow(26)}
      <div class="query">{query["text"]}</div></div>
    <div class="eq"><i>q<sub>j</sub></i> = Write(<i>t</i>, <i>d</i><sub>1</sub>, <i>d</i><sub>2</sub>, <i>d</i><sub>3</sub>),
      &nbsp;<span class="note">3&#8211;4,000 chars, 100 per task</span></div>
  </div>
  <div class="stage"><h3>Stage B: Retrieval Pool</h3>
    {goal("orange", "Pool every model's top 10 for each query")}
    {GRID}
    <div class="note">4 of {len(models)} models; &#10003; = in the model's top 10 for <i>q<sub>j</sub></i></div>
    <div class="eq"><i>D<sub>j</sub></i> = &#8899;<sub><i>m</i></sub> top10<sub><i>m</i></sub>(<i>q<sub>j</sub></i>), &nbsp;
      |<i>D<sub>j</sub></i>| = {len(POOL)}</div><div class="note" style="margin-top:-4px">each pooled document is judged once</div>
  </div>
</div></div>

<div class="link">{brace(560, flip=True)}{arrow(22)}</div>

<div class="col" style="width:520px"><h2>Decision-Model Judge</h2><div class="panel">
  {goal("pink", "Read the judge's probability for each answer, instead of sampling one answer")}
  <h3>Pointwise (default): grade each (<i>q<sub>j</sub></i>, <i>d</i>) in <i>D<sub>j</sub></i></h3>
  <div class="row"><span class="box">(<i>q<sub>j</sub></i>, <i>d</i>)</span>{arrow(26)}<span class="box"><b>judge</b></span>{arrow(26)}
    <div class="bars">{BARS}</div><span class="note" style="text-align:left"><i>p</i>(<i>k</i> | <i>q<sub>j</sub></i>, <i>d</i>)<br>over grades<br>0 to 5</span></div>
  <div class="eq">grade <i>g</i>(<i>q<sub>j</sub></i>, <i>d</i>) = &#8721;<sub><i>k</i></sub> <i>k</i> &#183; <i>p</i>(<i>k</i> | <i>q<sub>j</sub></i>, <i>d</i>) = {EXPECTED:.1f}</div>
  <div class="row" style="gap:16px"><span class="note">grades of the top 10:</span>
    <span class="note">A {gradebars(A_GRADES, BLUE)}</span><span class="note">B {gradebars(B_GRADES, "#c4cbd8")}</span></div>
  <div style="border-top:1.4px dashed {LINE};margin:2px 0"></div>
  <h3>Pairwise: is A's top 10 or B's better?</h3>
  <table><tr><th>order shown</th><th><i>P</i>(A better) + &#189;<i>P</i>(tie)</th></tr>
    <tr><td>A first, then B</td><td>{first:.2f}</td></tr>
    <tr><td>B first, then A</td><td>{second:.2f}</td></tr></table>
  <div class="eq"><i>s<sub>AB</sub></i> = mean over the two orders = {pair["score_a"]:.2f}</div>
  <div class="note">(another query) the swap cancels position bias</div>
</div></div>

<div class="link">{arrow(30)}</div>

<div class="col" style="width:520px"><h2>Gym Score</h2><div class="panel">
  {goal("purple", "Turn the judgments into one score per model, comparable across queries and tasks")}
  <h3>Pointwise: nDCG@10 over the pool</h3>
  <div class="eq">nDCG@10(<i>m</i>, <i>q<sub>j</sub></i>) = {frac("&#8721;<sub><i>r</i>&#8804;10</sub> <i>g</i>(<i>q<sub>j</sub></i>, <i>d<sub>m,r</sub></i>) / log<sub>2</sub>(<i>r</i>+1)", "the same for the 10 best in <i>D<sub>j</sub></i>")}</div>
  <div class="eq" style="font-size:19px">here: A = {dcg(A_GRADES):.1f} / {dcg(IDEAL):.1f} = {NDCG_A:.2f}, &nbsp; B = {dcg(B_GRADES):.1f} / {dcg(IDEAL):.1f} = {NDCG_B:.2f}</div>
  <div class="note">score<sub><i>m</i></sub> = 100 &#215; mean over queries, then over tasks</div>
  <div style="border-top:1.4px dashed {LINE};margin:2px 0"></div>
  <h3>Pairwise: Bradley&#8211;Terry over all <i>s</i></h3>
  <div class="eq"><i>P</i>(<i>m</i> &#8827; <i>n</i>) = {frac("<i>&#960;<sub>m</sub></i>", "<i>&#960;<sub>m</sub></i> + <i>&#960;<sub>n</sub></i>")}, &nbsp;
    Elo<sub><i>m</i></sub> = 400 log<sub>10</sub> <i>&#960;<sub>m</sub></i>, mean 1000</div>
  <div style="border-top:1.4px dashed {LINE};margin:2px 0"></div>
  <div class="note" style="color:{INK};font-size:19px">Resulting gym ranking ({len(models)} models)</div>
  <div style="display:flex;flex-direction:column;gap:3px;align-items:center">{RANKBARS}</div>
  <div class="note">Spearman &#961; = {RHO:.2f} with MTEB's ranking</div>
</div></div>
</body></html>"""

if __name__ == "__main__":
    render(HTML, "method", W, H)
