"""Method figure: how MTEB Gym writes queries, pools results, grades them and scores models, laid out like a method
diagram (an input, a dashed group of two stages, two filled panels), with real SciFact data throughout.

Each document keeps one colour everywhere it appears, and symbols are coloured by what they index (query j, document
i, grade k, model m)."""

import numpy as np

from plot_figure1 import QID, STRONG, WEAK, RHO, grades, gym, models, pair, probs, query, score, short, top
from plots import render

INK, GREY = "#111", "#7a7a7a"
ORANGE, BLUE, GREEN, PURPLE, RED = "#f08a24", "#2f6fdf", "#2fa84f", "#7b4fd0", "#d9302c"
DOCS = [ORANGE, BLUE, GREEN, PURPLE]
J, I, K, M = PURPLE, BLUE, ORANGE, GREEN  # colours of the indices: query j, document i, grade k, model m


def doc(color, h=30):
    return (f'<svg width="{h * .76:.0f}" height="{h}" viewBox="0 0 22 29" fill="#fff" stroke="{color}" stroke-width="1.7" '
            f'stroke-linejoin="round"><path d="M1.5 1.5h13l6 6v20h-19z"/><path d="M14.5 1.5v6h6" fill="none"/>'
            f'<path d="M5 11h12M5 14.5h12M5 18h12M5 21.5h9" stroke-width="1.4"/></svg>')


def screen(inner, w=92, h=70):
    """A small rounded card with two cards stacked behind it, holding a tiny diagram."""
    return (f'<span class="screen" style="width:{w}px;height:{h}px"><span></span><span></span><span>{inner}</span></span>')


def v(sym, color, sub=None, subcolor=None):
    s = f'<i style="color:{color}">{sym}</i>'
    if sub:
        s += f'<sub style="color:{subcolor or color}"><i>{sub}</i></sub>'
    return s


def frac(num, den):
    return f'<span class="frac"><span>{num}</span><span>{den}</span></span>'


def mark(ok, size=26):
    color, path = (GREEN, "M4.5 9.5l3.2 3.2L14 5.5") if ok else (RED, "M5.5 5.5l7 7M12.5 5.5l-7 7")
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 18 18"><rect x="1" y="1" width="16" height="16" rx="3.5" fill="#fff" '
            f'stroke="{INK}" stroke-width="1.4"/><path d="{path}" fill="none" stroke="{color}" stroke-width="2.6" '
            f'stroke-linecap="round" stroke-linejoin="round"/></svg>')


def brace(h, right=False):
    m = h / 2
    d = f"M18 2 C 9 2, 9 8, 9 {m - 10} C 9 {m - 3}, 5 {m}, 1 {m} C 5 {m}, 9 {m + 3}, 9 {m + 10} C 9 {h - 8}, 9 {h - 2}, 18 {h - 2}"
    return (f'<svg width="20" height="{h}" viewBox="0 0 20 {h}" fill="none" stroke="{INK}" stroke-width="1.8" '
            f'style="{"transform:scaleX(-1)" if right else ""}"><path d="{d}"/></svg>')


def arrow(w=30):
    return (f'<svg width="{w}" height="16" viewBox="0 0 {w} 16" fill="none" stroke="{INK}" stroke-width="2.2" '
            f'stroke-linecap="round" stroke-linejoin="round"><path d="M2 8H{w - 3}M{w - 9} 2.5l6 5.5-6 5.5"/></svg>')


def bars(values, colors, h=46, w=10, gap=3, top_value=None):
    top_value = top_value or max(values)
    return (f'<span class="bars" style="height:{h}px;gap:{gap}px">'
            + "".join(f'<i style="height:{max(2, h * x / top_value):.0f}px;width:{w}px;background:{c}"></i>'
                      for x, c in zip(values, colors)) + "</span>")


# Stage B: which of four pooled documents four models retrieve (real top-10 lists for the query).
SHOWN = [STRONG, gym[2], WEAK, "mteb/baseline-bm25s"]
lists = {m: set(top[m][QID][:10]) for m in SHOWN}
POOL = {d for m in models for d in top[m][QID][:10]}
cols = sorted(set.union(*lists.values()), key=lambda d: (-sum(d in lists[m] for m in SHOWN), -grades[QID][d]))
cols = [cols[0], cols[1], next(d for d in cols if d not in lists[WEAK]), next(d for d in cols[::-1] if d in lists[WEAK])]
GRID = ('<div class="grid"><b>Docs</b>' + "".join(f'<span>{doc(c, 25)}</span>' for c in DOCS)
        + "".join(f'<b>{short(m).replace("-Embedding", "")}</b>' + "".join(f'<span>{mark(d in lists[m], 23)}</span>' for d in cols)
                  for m in SHOWN) + "</div>")

# Judge: the grade distribution for one pooled document, and the expected grade.
EXPECTED = sum(k * p for k, p in enumerate(probs))
DIST = "".join(f'<span class="col"><i style="height:{max(2, 82 * p / max(probs)):.0f}px"></i><em>{k}</em></span>'
               for k, p in enumerate(probs))

# Pairwise: P(A's list is better), counting a tie as half, in each order.
first = pair["probabilities"][0]["System A"] + pair["probabilities"][0]["Tie"] / 2
second = pair["probabilities"][1]["System B"] + pair["probabilities"][1]["Tie"] / 2

# Scoring: nDCG@10 of model A and model B on the query, against the pool's ten best grades.
A_G = [grades[QID].get(d, 0) for d in top[STRONG][QID][:10]]
B_G = [grades[QID].get(d, 0) for d in top[WEAK][QID][:10]]
BEST = sorted(grades[QID].values(), reverse=True)[:10]
dcg = lambda gs: sum(g / np.log2(r + 2) for r, g in enumerate(gs))


RANKED = gym[:4]
FINAL = bars([score[m] - 60 for m in RANKED] + [score[WEAK] - 60], [PURPLE, BLUE, ORANGE, GREEN, "#bbb"], h=50, w=13, gap=4)

N_PAIRS = len(models) * (len(models) - 1) // 2
RANKLIST = ("".join(f'<div><b>{i + 1}</b>{short(m).replace("-Embedding", "")}</div>' for i, m in enumerate(gym[:3]))
            + '<div><b>&#8942;</b></div>' + f'<div><b>{gym.index(WEAK) + 1}</b>{short(WEAK)}</div>')


def pairgrid(n=6, cell=15):
    """The upper triangle of a models-by-models grid: one cell per pair of models."""
    cells = "".join(f'<rect x="{c * cell + 1}" y="{r * cell + 1}" width="{cell - 2}" height="{cell - 2}" rx="1.5" '
                    f'fill="{"#fff" if c <= r else "#d9d2f5"}" stroke="{INK if c > r else "#ccc"}" stroke-width="1"/>'
                    for r in range(n) for c in range(n))
    return f'<svg width="{n * cell + 2}" height="{n * cell + 2}" viewBox="0 0 {n * cell + 2} {n * cell + 2}">{cells}</svg>'


def toplist(color, n=3):
    return ('<span class="tl">' + "".join(f'<span>{doc(color, 15)}</span>' for _ in range(n)) + '<em>&#8942;</em></span>')


W, H = 1640, 1052
HTML = f"""<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=STIX+Two+Text:ital,wght@0,400;0,700;1,400&display=swap" rel="stylesheet">
<style>
* {{ box-sizing: border-box; margin: 0; }}
body {{ font-family: 'STIX Two Text', serif; color: {INK}; background: #fff; width: {W}px; height: {H}px; position: relative;
        font-size: 23px; line-height: 1.18; }}
.abs {{ position: absolute; }}
.wire {{ position: absolute; inset: 0; pointer-events: none; }}
.top, .bottom {{ display: flex; align-items: center; }}
.bottom {{ flex: 1; align-items: stretch; }}
.split {{ display: flex; padding-left: 40px; }}
.pairs {{ display: flex; gap: 14px; align-items: stretch; }}
.pairs > .link {{ flex: none; }}
.pairs .stage {{ justify-content: space-between; }}
i {{ font-style: italic; }}
h2 {{ font-size: 28px; font-weight: 700; text-align: center; white-space: nowrap; }}
h3 {{ font-size: 24px; font-weight: 700; text-align: center; white-space: nowrap; }}
.col0 {{ display: flex; flex-direction: column; align-items: center; gap: 6px; font-weight: 700; text-align: center; width: 88px; }}
.col0 small {{ display: block; font-weight: 400; font-size: 20px; }}
.group {{ border: 2.2px dashed {GREY}; border-radius: 20px; padding: 6px 12px 10px; background: #fafafa;
          display: flex; flex-direction: column; gap: 6px; }}
.group > h2, .lane > h2 {{ text-align: left; }}
.group > h2 small {{ font-size: 21px; font-weight: 400; margin-left: 10px; }}
.col0 {{ justify-content: center; }}
.stage {{ background: #f4f4f4; border: 1.6px solid #aaa; border-radius: 14px; padding: 8px 12px; display: flex; flex-direction: column;
          gap: 6px; }}
.lanes {{ flex: 1; display: flex; flex-direction: column; gap: 14px; align-self: stretch; }}
.lane {{ border: 2.4px solid {INK}; border-radius: 20px; padding: 6px 12px 8px; display: flex; flex-direction: column; gap: 6px; }}
.lane > h2 {{ text-align: left; }}
.lane > h2 small {{ font-size: 21px; font-weight: 400; margin-left: 10px; }}
.steps {{ flex: 1; display: flex; align-items: stretch; gap: 0; }}
.step {{ flex: 1; background: rgba(255,255,255,.7); border: 1.6px solid #999; border-radius: 14px; padding: 6px 10px;
         display: flex; flex-direction: column; gap: 6px; text-align: center; justify-content: center; }}
.goal {{ border: 1.6px solid #999; border-radius: 10px; padding: 3px 8px; white-space: nowrap; font-style: italic; text-align: center; font-size: 23px;
         line-height: 1.12; }}
.row {{ display: flex; align-items: center; justify-content: center; gap: 8px; }}
.docs {{ display: flex; flex-direction: column; gap: 4px; }}
.llm {{ border: 1.6px solid {INK}; border-radius: 8px; background: #fff; padding: 1px 9px; font-weight: 700; }}
.q {{ background: #fff; border: 1.6px solid {INK}; border-radius: 8px; padding: 4px 8px; font-style: italic; font-size: 20px;
      width: 250px; text-align: center; line-height: 1.12; }}
.grid {{ display: grid; grid-template-columns: auto repeat(4, 36px); gap: 1px 6px; align-items: center; justify-content: center; }}
.grid b {{ font-weight: 400; text-align: right; padding-right: 6px; white-space: nowrap; font-size: 21px; }}
.grid span {{ display: flex; justify-content: center; }}
.eq {{ font-size: 23px; white-space: nowrap; }}
.small {{ font-size: 20px; }}
.calls {{ font-size: 20px; text-align: right; color: #333; }}
.frac {{ display: inline-flex; flex-direction: column; vertical-align: middle; text-align: center; }}
.frac > span:first-child {{ border-bottom: 1.4px solid {INK}; padding: 0 4px 2px; }}
.frac > span:last-child {{ padding: 2px 4px 0; }}
.screen {{ position: relative; display: inline-block; flex: none; margin-left: 8px; }}
.screen > span {{ position: absolute; inset: 0; border: 1.6px solid {INK}; border-radius: 9px; background: #fff; }}
.screen > span:nth-child(1) {{ transform: translate(-8px, -7px); background: #f4f4f4; }}
.screen > span:nth-child(2) {{ transform: translate(-4px, -3.5px); background: #fafafa; }}
.screen > span:nth-child(3) {{ display: flex; align-items: flex-end; justify-content: center; padding: 6px; gap: 3px; }}
.bars {{ display: inline-flex; align-items: flex-end; }}
.bars i {{ border: 1px solid {INK}; border-radius: 1px; }}
.dist {{ display: flex; align-items: flex-end; gap: 5px; border-bottom: 1.6px solid {INK}; border-left: 1.6px solid {INK};
         padding: 0 5px; height: 70px; }}
.dist .col {{ display: flex; flex-direction: column; align-items: center; width: 18px; position: relative; height: 100%;
              justify-content: flex-end; }}
.dist .col i {{ width: 15px; background: {ORANGE}; border: 1px solid {INK}; }}
.dist .col em {{ position: absolute; bottom: -22px; font-style: normal; font-size: 17px; }}
.tl {{ align-self: center; display: inline-flex; flex-direction: column; align-items: center; gap: 1px; border: 1.4px solid {INK}; border-radius: 6px;
       background: #fff; padding: 3px 4px 0; }}
.tl em {{ font-style: normal; line-height: .6; font-size: 16px; }}
table {{ border-collapse: collapse; margin: 0 auto; font-size: 20px; }}
td {{ padding: 0 6px; white-space: nowrap; text-align: left; }}
.link {{ display: flex; align-items: center; padding: 0 2px; }}
.out {{ display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 8px; text-align: center; }}
.rank {{ background: #fff; border: 1.6px solid {INK}; border-radius: 10px; padding: 6px 10px; font-size: 19px; text-align: left;
         display: flex; flex-direction: column; gap: 2px; width: 100%; }}
.rank div {{ display: flex; gap: 6px; white-space: nowrap; }}
.rank b {{ width: 22px; text-align: right; font-weight: 400; color: #555; flex: none; }}
.num {{ display: inline-flex; align-items: center; justify-content: center; width: 30px; height: 30px; border-radius: 50%;
        background: {INK}; color: #fff; font-size: 20px; margin-right: 10px; vertical-align: 3px; }}
.stage h3 {{ text-align: left; }}
.llmcol {{ display: flex; flex-direction: column; align-items: center; gap: 2px; }}
.instr {{ border: 1.6px dashed {GREY}; border-radius: 8px; padding: 2px 8px; font-size: 17px; line-height: 1.1; text-align: center; background: #fff; }}
.spacer {{ height: 64px; }}
.qstack {{ position: relative; display: flex; flex-direction: column; align-items: center; gap: 4px; margin-left: 8px; }}
.qstack > span {{ position: absolute; left: 0; top: 0; width: 250px; height: calc(100% - 26px); border: 1.6px solid {INK}; border-radius: 8px; background: #f2f2f2; }}
.qstack > span:nth-child(1) {{ transform: translate(-8px, -7px); }}
.qstack > span:nth-child(2) {{ transform: translate(-4px, -3.5px); background: #fafafa; }}
.qstack .q {{ position: relative; }}
.qstack em {{ font-size: 18px; color: #444; }}
.group {{ background: #fff !important; }}
</style></head><body>
<svg class="wire" width="{W}" height="{H}" viewBox="0 0 {W} {H}" fill="none" stroke="{INK}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
  <path d="M112 132 H140 M132 126 L140 132 L132 138"/>
  <path d="M842 132 H870 M862 126 L870 132 L862 138"/>
  <path d="M1250 256 V292 M1244 284 L1250 292 L1256 284"/>
  <path d="M1390 676 H1416 M1408 670 L1416 676 L1408 682"/></svg>
<div class="col0 abs" style="left:4px;top:8px;width:106px;height:248px">{doc(ORANGE, 34)}{doc(BLUE, 34)}<div>Corpus<small>no queries,<br>no labels</small></div></div>

<div class="stage abs" style="left:146px;top:8px;width:694px;height:248px"><h3><span class="num">1</span>Write Queries</h3>
  <div class="goal" style="background:#fbe9df">An LLM writes a query from documents sampled from the corpus</div>
  <div class="row" style="gap:8px;flex:1">
    <div class="docs">{doc(ORANGE)}{doc(BLUE)}{doc(GREEN)}</div>
    <svg width="30" height="104" viewBox="0 0 30 104" fill="none" stroke="{INK}" stroke-width="1.4"><path d="M2 16 L28 52M2 52 L28 52M2 88 L28 52"/></svg>
    <div class="llmcol"><span class="instr">task instruction<br>(optional)</span>
      <svg width="16" height="20" viewBox="0 0 16 20" fill="none" stroke="{INK}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M8 1V18M3 13l5 5 5-5"/></svg>
      <span class="llm">LLM</span><span class="spacer"></span></div>
    {arrow(24)}<div class="qstack"><span></span><span></span><div class="q">{query["text"]}</div><em>1 of 100 per task</em></div></div>
</div>

<div class="stage abs" style="left:876px;top:8px;width:756px;height:248px"><h3><span class="num">2</span>Retrieve</h3>
  <div class="goal" style="background:#fbe9df">Every model returns its top 10 for each query</div>
  {GRID}
  <div class="small" style="text-align:center">&#10003; = in the model's top 10 &#160;(4 of {len(models)} models shown)</div>
</div>

<div class="group abs" style="left:4px;top:296px;width:1386px;height:750px"><h2><span class="num">3</span>Judge the results, in one of two ways</h2></div>

<div class="lane abs" style="background:#fdeef2;left:20px;top:344px;width:1354px;height:328px"><h2>Pointwise<small>grade each document once</small></h2>
  <div class="steps">
    <div class="step"><h3>Pool</h3>
      <div class="goal" style="background:#fbd5e1">Every document<br>any model returned</div>
      <div class="eq">{v("D", J, "j")} = &#8899;<sub><i style="color:{M}">m</i></sub> top10<sub><i style="color:{M}">m</i></sub>({v("q", J, "j")})</div>
      <div class="small">{len(POOL)} documents here</div></div>
    <div class="link">{arrow(22)}</div>
    <div class="step" style="flex:1.45"><h3>Grade</h3>
      <div class="goal" style="background:#fbd5e1">The judge sees the query and one document</div>
      <div class="row">{screen(f'<i style="align-self:center;color:{J}">q<sub>j</sub></i><b style="font-size:21px;align-self:center">+</b>{doc(BLUE, 32)}', 94, 54)}
        {arrow(20)}<span class="llm">Judge</span>{arrow(20)}<div class="dist">{DIST}</div></div>
      <div class="eq" style="margin-top:14px">{v("g", I, "ij")} = &#8721;<sub><i style="color:{K}">k</i></sub> <i style="color:{K}">k</i> &#183; <i>p</i>(<i style="color:{K}">k</i> | {v("q", J, "j")}, {v("d", I, "i")}) = {EXPECTED:.1f}</div></div>
    <div class="link">{arrow(22)}</div>
    <div class="step" style="flex:1.25"><h3>Score</h3>
      <div class="goal" style="background:#fbd5e1">nDCG@10 with the grades as labels</div>
      <div class="eq">nDCG@10 = {frac(f"&#8721;<sub><i>r</i>&#8804;10</sub> {v('g', I, 'r')} / log<sub>2</sub>(<i>r</i>+1)", f"the same for the 10 best in {v('D', J, 'j')}")}</div>
      <div class="row" style="gap:18px"><span class="small">A</span>{bars(A_G, [BLUE] * 10, h=34, w=7, gap=2, top_value=5)}<span class="small">{dcg(A_G) / dcg(BEST):.2f}</span>
        <span class="small">B</span>{bars(B_G, ["#bbb"] * 10, h=34, w=7, gap=2, top_value=5)}<span class="small">{dcg(B_G) / dcg(BEST):.2f}</span></div></div>
  </div>
  <div class="calls">Judge calls per query: one per pooled document ({len(POOL)} here)</div>
</div>

<div class="lane abs" style="background:#f1edfc;left:20px;top:682px;width:1354px;height:352px"><h2>Pairwise<small>compare two models' lists</small></h2>
  <div class="steps">
    <div class="step"><h3>Pair Up</h3>
      <div class="goal" style="background:#e2daf8">Every pair<br>of models</div>
      <div class="row">{pairgrid()}</div>
      <div class="small">{N_PAIRS} pairs of {len(models)} models,<br>or a random subset</div></div>
    <div class="link">{arrow(22)}</div>
    <div class="step" style="flex:1.45"><h3>Compare</h3>
      <div class="goal" style="background:#e2daf8">The judge sees the query and both top 10s</div>
      <div class="row" style="margin-top:12px">{screen(f'<i style="align-self:center;color:{J}">q<sub>j</sub></i>{toplist(BLUE)}{toplist("#999")}', 112, 100)}
        {arrow(20)}<span class="llm">Judge</span>{arrow(20)}
        <table><tr><td>A first:</td><td><i>P</i>(A better) = {first:.2f}</td></tr><tr><td>B first:</td><td><i>P</i>(A better) = {second:.2f}</td></tr></table></div>
      <div class="eq"><i>s</i><sub>AB</sub> = mean of both orders = {pair["score_a"]:.2f}</div>
      <div class="small">each document cut to 4,000 characters</div></div>
    <div class="link">{arrow(22)}</div>
    <div class="step" style="flex:1.25"><h3>Score</h3>
      <div class="goal" style="background:#e2daf8">Bradley&#8211;Terry fit to all pair scores</div>
      <div class="eq"><i>P</i>(<i style="color:{M}">m</i> beats <i style="color:{M}">n</i>) = {frac(f"<i style='color:{M}'>&#960;</i><sub><i style='color:{M}'>m</i></sub>", f"<i style='color:{M}'>&#960;</i><sub><i style='color:{M}'>m</i></sub> + <i style='color:{M}'>&#960;</i><sub><i style='color:{M}'>n</i></sub>")}</div>
      <div class="eq">Elo<sub><i style="color:{M}">m</i></sub> = 400 log<sub>10</sub> <i style="color:{M}">&#960;</i><sub><i style="color:{M}">m</i></sub>, mean 1000</div></div>
  </div>
  <div class="calls">Judge calls per query: two per pair ({2 * N_PAIRS} for all pairs)</div>
</div>

<div class="out abs" style="left:1420px;top:542px;width:216px;height:270px"><h3><span class="num">4</span>Rank</h3><div class="rank">{RANKLIST}</div>
  <div class="small">pointwise:<br>&#961; = {RHO:.2f} with MTEB</div></div>
</body></html>"""

if __name__ == "__main__":
    render(HTML, "method", W, H)
