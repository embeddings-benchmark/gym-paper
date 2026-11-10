"""Method figure: the mechanism of each MTEB Gym stage, with the paper's notation, on the SciFact query of Figure 1.

A technical companion to Figure 1 in the style of a method diagram: four outlined panels (query generation, the
retrieval pool, judging, scoring), each opening with its goal in a tinted box, then the mechanism and its equation,
with small diagrams drawn from real data."""

from plot_figure1 import QID, STRONG, WEAK, gym, models, pair, probs, query, short, top, RHO
from plots import render

INK, MUTED, LINE = "#1d2433", "#5b6478", "#9aa3b5"
TINT = {"gen": ("#fdeee2", "#e3a274"), "pool": ("#e6effb", "#7fa3d6"), "judge": ("#fbe6ec", "#d98aa0"),
        "score": ("#eeeafb", "#9a8bd6")}
PALETTE = ["#e8945a", "#6c9bd2", "#74b88a", "#a58fd6", "#d9748f", "#d6b44a", "#5fb3b3", "#9aa3b5"]


def page(color, size=34):
    """A document: a page with a folded corner and text lines, outlined in its colour."""
    return (f'<svg width="{size * .78:.0f}" height="{size}" viewBox="0 0 22 28" fill="#fff" stroke="{color}" stroke-width="1.8" '
            f'stroke-linejoin="round"><path d="M2 2h12l6 6v18H2z"/><path d="M14 2v6h6" fill="none"/>'
            f'<path d="M6 13h10M6 17h10M6 21h7" stroke-width="1.5"/></svg>')


def goal(kind, text):
    bg, border = TINT[kind]
    return f'<div class="goal" style="background:{bg};border-color:{border}">{text}</div>'


def frac(num, den):
    return f'<span class="frac"><span>{num}</span><span>{den}</span></span>'


def arrow(w=36):
    return (f'<svg width="{w}" height="22" viewBox="0 0 {w} 22" fill="none" stroke="{INK}" stroke-width="2.4" '
            f'stroke-linecap="round" stroke-linejoin="round"><path d="M2 11H{w - 3}M{w - 11} 4l8 7-8 7"/></svg>')


def down(h=34):
    return (f'<svg width="22" height="{h}" viewBox="0 0 22 {h}" fill="none" stroke="{INK}" stroke-width="2.4" '
            f'stroke-linecap="round" stroke-linejoin="round"><path d="M11 2V{h - 3}M4 {h - 11}l7 8 7-8"/></svg>')


# The pool on the query: every model's top 10, a colour per document shared by several of the models shown.
SHOWN = [STRONG, gym[2], WEAK, "mteb/baseline-bm25s"]
lists = {m: top[m][QID][:10] for m in SHOWN}
POOL = {d for m in models for d in top[m][QID][:10]}
counts = {}
for m in SHOWN:
    for d in lists[m]:
        counts[d] = counts.get(d, 0) + 1
shared = [d for d in dict.fromkeys(d for m in SHOWN for d in lists[m]) if counts[d] > 1]
color = {d: PALETTE[i % (len(PALETTE) - 1)] for i, d in enumerate(shared)}
ROWS = "".join(f'<div class="row"><em>{short(m)}</em>'
               + "".join(f'<i style="background:{color.get(d, "#fff")};border-color:{color.get(d, LINE)}"></i>' for d in lists[m])
               + "</div>" for m in SHOWN)

BARS = "".join(f'<div class="bar"><i style="height:{4 + 92 * p / max(probs):.0f}px"></i><span>{g}</span></div>'
               for g, p in enumerate(probs))
EXPECTED = sum(g * p for g, p in enumerate(probs))
TOP3 = "".join(f'<li><b>{i + 1}</b>{short(m)}</li>' for i, m in enumerate(gym[:3]))

W, H = 2000, 712
HTML = f"""<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=STIX+Two+Text:ital,wght@0,400;0,600;0,700;1,400;1,600&display=swap" rel="stylesheet">
<style>
* {{ box-sizing: border-box; margin: 0; }}
body {{ font-family: 'STIX Two Text', serif; color: {INK}; background: #fff; width: {W}px; height: {H}px; padding: 14px 18px;
        display: flex; align-items: stretch; gap: 0; font-size: 26px; line-height: 1.25; }}
.col {{ display: flex; flex-direction: column; }}
.col > h2 {{ font-size: 31px; font-weight: 700; text-align: center; margin-bottom: 10px; }}
.panel {{ flex: 1; border: 2px solid {LINE}; border-radius: 18px; padding: 16px 18px; display: flex; flex-direction: column;
          gap: 12px; }}
.goal {{ border: 2px solid; border-radius: 10px; padding: 6px 10px; font-style: italic; text-align: center; font-size: 26px; }}
.link {{ display: flex; align-items: center; padding: 0 4px; padding-top: 44px; }}
.eq {{ text-align: center; font-size: 27px; }}
.eq i {{ font-style: italic; }}
.note {{ text-align: center; font-size: 23px; color: {MUTED}; }}
.frac {{ display: inline-flex; flex-direction: column; vertical-align: middle; text-align: center; }}
.frac > span:first-child {{ border-bottom: 1.6px solid {INK}; padding: 0 4px 2px; }}
.frac > span:last-child {{ padding: 2px 4px 0; }}
.box {{ border: 1.6px solid {LINE}; border-radius: 10px; padding: 6px 10px; background: #fff; white-space: nowrap; }}
.dash {{ border-style: dashed; color: {MUTED}; }}
.center {{ display: flex; flex-direction: column; align-items: center; gap: 6px; }}
.docs {{ display: flex; gap: 8px; align-items: flex-end; }}
.llm {{ border: 2px solid {INK}; border-radius: 10px; padding: 4px 18px; font-weight: 700; }}
.query {{ border: 2px solid {INK}; border-radius: 10px; padding: 8px 12px; font-style: italic; text-align: center; font-size: 24px; }}
.row {{ display: grid; grid-template-columns: 170px repeat(10, 1fr); gap: 4px; align-items: center; }}
.row em {{ font-style: normal; font-size: 21px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; padding-right: 6px; }}
.row i {{ height: 26px; border-radius: 4px; border: 1.6px solid; }}
.dots {{ text-align: center; color: {MUTED}; line-height: .6; font-size: 26px; }}
.stage {{ border: 1.6px dashed {LINE}; border-radius: 12px; padding: 8px 12px 10px; display: flex; flex-direction: column; gap: 6px; }}
.stage h3 {{ font-size: 25px; font-weight: 700; }}
.stage h3 small {{ font-weight: 400; color: {MUTED}; font-size: 22px; }}
.flow {{ display: flex; align-items: center; justify-content: center; gap: 8px; }}
.bars {{ display: flex; align-items: flex-end; gap: 6px; height: 120px; padding-bottom: 2px; border-bottom: 1.6px solid {LINE}; }}
.bar {{ display: flex; flex-direction: column; align-items: center; gap: 2px; width: 26px; }}
.bar i {{ width: 22px; background: {TINT["judge"][1]}; border-radius: 3px 3px 0 0; }}
.bar span {{ font-size: 20px; color: {MUTED}; }}
.mini {{ display: flex; flex-direction: column; gap: 3px; }}
.mini span {{ display: flex; gap: 3px; }}
.mini i {{ width: 15px; height: 15px; border-radius: 3px; border: 1.4px solid {LINE}; }}
.rank {{ list-style: none; padding: 0; display: flex; flex-direction: column; gap: 2px; font-size: 23px; }}
.rank b {{ display: inline-block; width: 26px; color: {MUTED}; }}
</style></head><body>

<div class="col" style="width:410px"><h2>Query generation</h2><div class="panel">
  {goal("gen", "Write queries a user of the corpus could ask")}
  <div class="box dash" style="text-align:center;font-size:23px">task instruction <i>t</i> (optional)</div>
  <div class="center"><div class="docs">{page(PALETTE[0])}{page(PALETTE[1])}{page(PALETTE[2])}</div>
    <div class="note" style="color:{INK}">3 documents <i>d</i><sub>1</sub>, <i>d</i><sub>2</sub>, <i>d</i><sub>3</sub> sampled from the corpus</div>
    {down(28)}<div class="llm">LLM writer</div>{down(28)}</div>
  <div class="query">{query["text"]}</div>
  <div class="eq"><i>q<sub>j</sub></i> = Write(<i>t</i>, <i>d</i><sub>1</sub>, <i>d</i><sub>2</sub>, <i>d</i><sub>3</sub>)</div>
  <div class="note">kept if 3&#8211;4,000 characters; 100 per task</div>
</div></div>

<div class="link">{arrow()}</div>

<div class="col" style="width:400px"><h2>Retrieval pool</h2><div class="panel">
  {goal("pool", "Every model retrieves its top 10 for every query")}
  <div class="note" style="color:{INK};text-align:left">top 10 for <i>q<sub>j</sub></i>, by rank &#8594;</div>
  <div class="mini" style="gap:12px">{ROWS}</div>
  <div class="dots">&#8942;</div>
  <div class="note">4 of {len(models)} models; one colour per document retrieved by several</div>
  <div class="eq" style="margin-top:8px"><i>D<sub>j</sub></i> = &#8899;<sub><i>m</i></sub> top10<sub><i>m</i></sub>(<i>q<sub>j</sub></i>)</div>
  <div class="note">here |<i>D<sub>j</sub></i>| = {len(POOL)} documents, each judged once</div>
</div></div>

<div class="link">{arrow()}</div>

<div class="col" style="width:610px"><h2>Judging</h2><div class="panel">
  {goal("judge", "An LLM judge rates what the models retrieve")}
  <div class="stage"><h3>Pointwise <small>(default): grade each document 0 to 5</small></h3>
    <div class="flow"><span class="box">(<i>q<sub>j</sub></i>, <i>d</i>)</span>{arrow(40)}<span class="llm">judge</span>{arrow(40)}
      <div class="bars">{BARS}</div><span class="note" style="text-align:left">p(grade)</span></div>
    <div class="eq"><i>g</i>(<i>q<sub>j</sub></i>, <i>d</i>) = &#8721;<sub><i>k</i></sub> <i>k</i> &#183; <i>p</i>(<i>k</i> | <i>q<sub>j</sub></i>, <i>d</i>) = {EXPECTED:.1f}</div></div>
  <div class="stage"><h3>Pairwise<small>: which of two top 10s is better</small></h3>
    <div class="flow"><span class="box">(<i>q<sub>j</sub></i>, top10<sub><i>A</i></sub>, top10<sub><i>B</i></sub>)</span>{arrow(40)}
      <span class="llm">judge</span><span class="note" style="text-align:left">both orders,<br>averaged</span></div>
    <div class="eq"><i>s<sub>A</sub></i> = <i>P</i>(<i>A</i> wins) + &#189;<i>P</i>(tie) = {pair["score_a"]:.2f}</div></div>
</div></div>

<div class="link">{arrow()}</div>

<div class="col" style="width:430px"><h2>Scoring</h2><div class="panel">
  {goal("score", "Turn the judgments into one score per model")}
  <div class="stage"><h3>Pointwise</h3>
    <div class="eq">nDCG@10 = {frac("DCG of the model's top 10", "DCG of the 10 best in <i>D<sub>j</sub></i>")}</div>
    <div class="eq" style="font-size:24px">DCG = &#8721;<sub><i>r</i>&#8804;10</sub> <i>g</i>(<i>q<sub>j</sub></i>, <i>d<sub>r</sub></i>) / log<sub>2</sub>(<i>r</i>+1)</div>
    <div class="note">&#215;100, mean over queries, then tasks</div></div>
  <div class="stage"><h3>Pairwise</h3>
    <div class="note" style="color:{INK}">Bradley&#8211;Terry over all <i>s<sub>A</sub></i>, Elo scale</div></div>
  <div class="box" style="margin-top:auto"><ol class="rank">{TOP3}</ol>
    <div class="note" style="text-align:left;margin-top:2px">&#8942; {len(models)} models &#183; &#961; = {RHO:.2f} with MTEB</div></div>
</div></div>
</body></html>"""

if __name__ == "__main__":
    render(HTML, "method", W, H)
