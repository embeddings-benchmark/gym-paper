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
GRID = ('<div class="grid"><b>Docs</b>' + "".join(f'<span>{doc(c, 28)}</span>' for c in DOCS)
        + "".join(f'<b>{short(m).replace("-Embedding", "")}</b>' + "".join(f'<span>{mark(d in lists[m])}</span>' for d in cols)
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

W, H = 1720, 720
HTML = f"""<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=STIX+Two+Text:ital,wght@0,400;0,700;1,400&display=swap" rel="stylesheet">
<style>
* {{ box-sizing: border-box; margin: 0; }}
body {{ font-family: 'STIX Two Text', serif; color: {INK}; background: #fff; width: {W}px; height: {H}px; padding: 8px 10px;
        display: flex; align-items: center; font-size: 24px; line-height: 1.18; }}
i {{ font-style: italic; }}
h2 {{ font-size: 29px; font-weight: 700; text-align: center; white-space: nowrap; margin-bottom: 6px; }}
h3 {{ font-size: 25px; font-weight: 700; text-align: center; white-space: nowrap; }}
.col0 {{ display: flex; flex-direction: column; align-items: center; gap: 6px; font-weight: 700; text-align: center; width: 92px; }}
.col0 small {{ display: block; font-weight: 400; font-size: 20px; }}
.colx {{ display: flex; flex-direction: column; align-self: stretch; }}
.group {{ flex: 1; border: 2.2px dashed {GREY}; border-radius: 18px; padding: 6px 10px 10px; display: flex; flex-direction: column;
          gap: 4px; }}
.stage {{ background: #f4f4f4; border: 1.6px solid #aaa; border-radius: 14px; padding: 8px 12px; display: flex; flex-direction: column;
          gap: 6px; }}
.panel {{ flex: 1; border: 2.4px solid {INK}; border-radius: 22px; padding: 10px 14px; display: flex; flex-direction: column; gap: 8px;
          text-align: center; }}
.goal {{ border: 1.6px solid #999; border-radius: 12px; padding: 4px 10px; font-style: italic; text-align: center; font-size: 25px;
         line-height: 1.15; }}
.row {{ display: flex; align-items: center; justify-content: center; gap: 10px; }}
.docs {{ display: flex; flex-direction: column; gap: 4px; }}
.llm {{ border: 1.6px solid {INK}; border-radius: 8px; background: #fff; padding: 2px 10px; font-weight: 700; }}
.q {{ background: #fff; border: 1.6px solid {INK}; border-radius: 8px; padding: 4px 8px; font-style: italic; font-size: 20px;
      width: 250px; text-align: center; line-height: 1.12; }}
.grid {{ display: grid; grid-template-columns: auto repeat(4, 38px); gap: 3px 6px; align-items: center; justify-content: center; }}
.grid b {{ font-weight: 400; text-align: right; padding-right: 6px; white-space: nowrap; font-size: 22px; }}
.grid span {{ display: flex; justify-content: center; }}
.eq {{ font-size: 25px; white-space: nowrap; }}
.small {{ font-size: 21px; }}
.frac {{ display: inline-flex; flex-direction: column; vertical-align: middle; text-align: center; }}
.frac > span:first-child {{ border-bottom: 1.4px solid {INK}; padding: 0 4px 2px; }}
.frac > span:last-child {{ padding: 2px 4px 0; }}
.rule {{ border-top: 1.4px solid {INK}; margin: 2px 10px; }}
.screen {{ position: relative; display: inline-block; flex: none; margin-left: 8px; }}
.screen > span {{ position: absolute; inset: 0; border: 1.6px solid {INK}; border-radius: 9px; background: #fff; }}
.screen > span:nth-child(1) {{ transform: translate(-8px, -7px); background: #f4f4f4; }}
.screen > span:nth-child(2) {{ transform: translate(-4px, -3.5px); background: #fafafa; }}
.screen > span:nth-child(3) {{ display: flex; align-items: flex-end; justify-content: center; padding: 6px; gap: 3px; }}
.bars {{ display: inline-flex; align-items: flex-end; }}
.bars i {{ border: 1px solid {INK}; border-radius: 1px; }}
.dist {{ display: flex; align-items: flex-end; gap: 6px; border-bottom: 1.6px solid {INK}; border-left: 1.6px solid {INK};
         padding: 0 6px; height: 96px; position: relative; }}
.dist .col {{ display: flex; flex-direction: column; align-items: center; width: 22px; position: relative; height: 100%;
              justify-content: flex-end; }}
.dist .col i {{ width: 18px; background: {ORANGE}; border: 1px solid {INK}; }}
.dist .col em {{ position: absolute; bottom: -24px; font-style: normal; font-size: 18px; }}
table {{ border-collapse: collapse; margin: 0 auto; font-size: 22px; }}
td {{ padding: 1px 8px; white-space: nowrap; }}
td.n {{ text-align: right; }}
.link {{ display: flex; align-items: center; padding: 0 2px; }}
</style></head><body>

<div class="col0">{doc(ORANGE, 34)}{doc(BLUE, 34)}<div>Corpus<small>no queries,<br>no labels</small></div></div>
<div class="link">{brace(540)}</div>

<div class="colx" style="width:480px"><h2>Queries and Results</h2><div class="group">
  <h3>Stage A: Write Queries</h3>
  <div class="stage">
    <div class="goal" style="background:#fbe9df">Write a query from documents<br>sampled from the corpus</div>
    <div class="row" style="gap:8px">
      <div class="docs">{doc(ORANGE)}{doc(BLUE)}{doc(GREEN)}</div>
      <svg width="34" height="104" viewBox="0 0 34 104" fill="none" stroke="{INK}" stroke-width="1.4">
        <path d="M2 16 L32 52M2 52 L32 52M2 88 L32 52"/></svg>
<span class="llm">LLM</span>
      {arrow(24)}<div class="q">{query["text"]}</div></div>
    <div class="small">plus an optional task instruction;<br>100 queries per task, each 3 to 4,000 characters</div>
  </div>
  <h3 style="margin-top:6px">Stage B: Pool the Results</h3>
  <div class="stage">
    <div class="goal" style="background:#fbe9df">Every model retrieves its top 10;<br>the judge grades their union</div>
    {GRID}
    <div class="eq" style="text-align:center">{v("D", J, "j")} = &#8899;<sub><i style="color:{M}">m</i></sub> top10<sub><i style="color:{M}">m</i></sub>({v("q", J, "j")})
      <span class="small">&nbsp;({len(POOL)} documents here)</span></div>
  </div>
</div></div>

<div class="link">{brace(540, right=True)}</div>
<div class="link">{arrow(26)}</div>

<div class="colx" style="width:540px"><h2>LLM Judge</h2><div class="panel" style="background:#fdeef2">
  <div class="goal" style="background:#fbd5e1">Grade each pooled document from 0 to 5,<br>seeing only the query and that document</div>
  <div class="row" style="gap:10px;margin-top:6px">
    {screen(f'<i style="align-self:center;color:{J}">q<sub>j</sub></i><b style="font-size:22px;align-self:center">+</b>{doc(BLUE, 34)}', 100, 60)}
    {arrow(26)}<span class="llm">LLM</span>{arrow(26)}<span style="white-space:nowrap">a probability for<br>each grade <i style="color:{K}">k</i> = 0&#8211;5</span></div>
  <div class="eq">{v("g", I, "ij")} = &#8721;<sub><i style="color:{K}">k</i></sub> <i style="color:{K}">k</i> &#183; <i>p</i>(<i style="color:{K}">k</i> | {v("q", J, "j")}, {v("d", I, "i")})</div>
  <div class="row" style="gap:22px;margin:2px 0 18px">
    <div class="dist">{DIST}</div>
    <div style="text-align:left">For this query<br>and document:<br>{v("g", I, "ij")} = <b>{EXPECTED:.1f}</b></div></div>
  <div class="rule"></div>
  <div><b>Or pairwise:</b> is A's or B's top 10 better?</div>
  <table><tr><td>A's list shown first:</td><td class="n"><i>P</i>(A better) = {first:.2f}</td></tr>
    <tr><td>B's list shown first:</td><td class="n"><i>P</i>(A better) = {second:.2f}</td></tr></table>
  <div class="small">The judge favours the list it reads first,<br>so each pair is judged in both orders and averaged: <b>{pair["score_a"]:.2f}</b></div>
</div></div>

<div class="link">{arrow(26)}</div>

<div class="colx" style="width:500px"><h2>Gym Ranking</h2><div class="panel" style="background:#f1edfc">
  <div class="goal" style="background:#e2daf8">Score each model by nDCG@10,<br>using the judge's grades as labels</div>
  <div class="eq" style="margin-top:6px">nDCG@10 = {frac(f"&#8721;<sub><i>r</i>&#8804;10</sub> {v('g', I, 'r')} / log<sub>2</sub>(<i>r</i>+1)", f"the same for the 10 best in {v('D', J, 'j')}")}</div>
  <div class="small">{v("g", I, "r")}: grade of the model's result at rank <i>r</i></div>
  <div class="small" style="margin-top:2px">This query, grades of each top 10:</div>
  <div class="row" style="gap:28px">
    <div>{bars(A_G, [BLUE] * 10, h=44, w=9, gap=2, top_value=5)}<div class="small">A: {dcg(A_G) / dcg(BEST):.2f}</div></div>
    <div>{bars(B_G, ["#bbb"] * 10, h=44, w=9, gap=2, top_value=5)}<div class="small">B: {dcg(B_G) / dcg(BEST):.2f}</div></div></div>
  <div class="small">Averaged over queries, then over tasks</div>
  <div class="rule"></div>
  <div class="small"><b>Pairwise:</b> a Bradley&#8211;Terry fit to all comparisons,<br>
    <i>P</i>(<i style="color:{M}">m</i> beats <i style="color:{M}">n</i>) = {frac(f"<i style='color:{M}'>&#960;</i><sub><i style='color:{M}'>m</i></sub>", f"<i style='color:{M}'>&#960;</i><sub><i style='color:{M}'>m</i></sub> + <i style='color:{M}'>&#960;</i><sub><i style='color:{M}'>n</i></sub>")}</div>
  <div class="rule"></div>
  <div class="row" style="gap:16px;margin-top:6px">{screen(FINAL, 110, 74)}
    <span style="text-align:left">Ranking of all {len(models)} models:<br>&#961; = {RHO:.2f} with MTEB's ranking</span></div>
</div></div>
</body></html>"""

if __name__ == "__main__":
    render(HTML, "method", W, H)
