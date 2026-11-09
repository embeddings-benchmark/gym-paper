"""Method figure: each step of MTEB Gym in detail, on the same SciFact query as Figure 1.

Three stages, as in the method section: writing queries, judging (pointwise, the default, and pairwise), and turning
the judgments into a ranking. Data, icons and style come from plot_figure1."""

from plot_figure1 import (GENERATOR, JUDGE, NAVY, ORANGE, PANEL, CREAM, GRAIN, RHO, QID, SEED, STRONG, WEAK, docs, emoji, grades,
                          gym, line, logo, mteb, pair, probs, query, score, short, title, top)
from plots import render

BLUE = "#1f5bd6"


def robot(badge, size=96):
    return (f'<div class="bot" style="width:{size}px;height:{size}px">{emoji("robot", int(size * .62))}'
            f'<span>{emoji(badge, 30)}</span></div>')


def arrow(width=40):
    return (f'<svg width="{width}" height="24" viewBox="0 0 {width} 24" fill="none" stroke="{NAVY}" stroke-width="3.5" '
            f'stroke-linecap="round" stroke-linejoin="round"><path d="M2 12H{width - 4}M{width - 13} 4l9 8-9 8"/></svg>')


def doc(d, n=28):
    return f'<div class="doc">{line("file-text", 26, "#8fb4f0")}<span>{title(d, n)}</span></div>'


def grade(g):
    bg, fg = ("#e3ecfd", BLUE) if g >= 2 else ("#fde4df", "#d1342b")
    return f'<b class="g" style="background:{bg};color:{fg}">{g:.1f}</b>'


def mini(m, tag):
    rows = "".join(f'<div class="mr">{line("file-text", 20, "#8fb4f0")}{grade(grades[QID][d])}</div>' for d in top[m][QID][:3])
    return f'<div class="mini"><div class="mh"><i class="av">{tag}</i>top 10</div>{rows}<div class="dots">…</div></div>'


bars = "".join(f'<div class="bar"><i style="height:{6 + 110 * p / max(probs):.0f}px;opacity:{.3 + .7 * p / max(probs):.2f}"></i>'
               f'<span>{g}</span></div>' for g, p in enumerate(probs))
podium = "".join(f'<div class="pr">{emoji(medal, 34)}{logo(m.split("/")[0], 30)}<span>{short(m)}</span>'
                 f'<b>{score[m]:.1f}</b><em>#{mteb.index(m) + 1}</em></div>'
                 for medal, m in zip(["1st_place_medal", "2nd_place_medal", "3rd_place_medal"], gym[:3]))

STAGE1 = f"""<div class="stage" style="width:560px">
  <div class="sh"><i>1</i>Write queries</div>
  <div class="aim">Write a query someone could ask, from documents in the corpus</div>
  <div class="task">{line("scroll-text", 24, NAVY)}<span>Task description: <b>SciFact</b> verifies scientific claims with paper abstracts</span></div>
  <div class="lbl">3 sampled documents</div>
  {"".join(doc(d) for d in query["seed_doc_ids"][:3])}
  <div class="step">{robot("writing_hand")}{arrow(56)}<div class="q">{emoji("magnifying_glass_tilted_right", 30)}<span>{query["text"]}</span></div></div>
  <div class="note">kept if 3 to 4,000 characters; 100 queries per task</div>
</div>"""

STAGE2 = f"""<div class="stage" style="width:720px">
  <div class="sh"><i>2</i>Judge the results</div>
  <div class="aim">Each model retrieves its top 10, and an LLM judge grades them</div>
  <div class="mode"><small>pointwise (default)</small>
    <div class="row">
      <div class="pair">{emoji("magnifying_glass_tilted_right", 26)}<span>query</span><b>+</b>{doc(SEED, 26)}</div>
      {arrow(36)}{robot("balance_scale", 84)}{arrow(36)}
      <div class="hist">{bars}</div>
    </div>
    <div class="f">grade = Σ g · p(g) = <b>{grades[QID][SEED]:.1f}</b><span>its probability for each grade 0–5</span></div>
  </div>
  <div class="mode"><small>pairwise</small>
    <div class="row">
      {mini(STRONG, "A")}<b class="vs">vs</b>{mini(WEAK, "B")}
      {arrow(36)}{robot("balance_scale", 84)}{arrow(36)}
      <div class="win">{emoji("crown", 34)}<span>A is better</span><em>p = {pair["score_a"]:.2f}</em></div>
    </div>
    <div class="f t">shown in both orders, A first and B first, and averaged</div>
  </div>
</div>"""

STAGE3 = f"""<div class="stage" style="width:520px">
  <div class="sh"><i>3</i>Rank the models</div>
  <div class="aim">Turn the judgments into one score per model</div>
  <div class="eq"><small>pointwise</small>
    <div class="m">nDCG@10 = <span class="frac"><span>Σ<sub>i ≤ 10</sub> grade<sub>i</sub> / log<sub>2</sub>(i + 1)</span><span>same, for the best 10 grades</span></span></div>
    <div class="sub">mean over queries, then over tasks</div></div>
  <div class="eq"><small>pairwise</small>
    <div class="m">Bradley–Terry over all wins</div>
    <div class="sub">one rating per model, on an Elo scale</div></div>
  <div class="board"><div class="bh">{emoji("person_lifting_weights", 34)}Gym ranking<em>{logo("mteb", 24)}MTEB</em></div>{podium}
    <div class="rho">ρ = {RHO:.2f} with MTEB's ranking</div></div>
</div>"""

W, H = 1960, 860
HTML = f"""<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Nunito:wght@600;700;800;900&family=STIX+Two+Text:ital,wght@0,500;1,500&display=swap" rel="stylesheet">
<style>
* {{ box-sizing: border-box; margin: 0; }}
body {{ font-family: Nunito, sans-serif; color: {NAVY}; background: {CREAM}; width: {W}px; height: {H}px; padding: 24px 20px;
        display: flex; align-items: stretch; gap: 10px; }}
.stage {{ flex: none; background: {PANEL}; background-image: {GRAIN}; border-radius: 20px; padding: 22px 24px;
          display: flex; flex-direction: column; gap: 14px; }}
.link {{ flex: none; align-self: center; }}
.sh {{ display: flex; align-items: center; gap: 14px; font-size: 36px; font-weight: 900; }}
.sh i {{ font-style: normal; width: 44px; height: 44px; border-radius: 50%; background: {NAVY}; color: #fff; font-size: 26px;
         display: flex; align-items: center; justify-content: center; }}
.aim {{ font-family: "STIX Two Text", serif; font-style: italic; font-size: 26px; line-height: 1.2; color: #2b4a85;
        border: 2px dashed #9db3da; border-radius: 12px; padding: 8px 14px; background: rgba(255,255,255,.5); }}
.task {{ display: flex; gap: 10px; align-items: flex-start; font-size: 22px; font-weight: 700; line-height: 1.25;
         background: #fff; border-radius: 12px; padding: 10px 14px; border: 1.5px solid #d3dcea; }}
.task svg {{ flex: none; margin-top: 2px; }}
.lbl {{ font-size: 20px; font-weight: 900; letter-spacing: .6px; text-transform: uppercase; color: #6b7fa6; margin-bottom: -6px; }}
.doc {{ display: flex; align-items: center; gap: 10px; font-size: 24px; font-weight: 800; background: #fff; border-radius: 10px;
        padding: 8px 12px; border: 1.5px solid #d3dcea; white-space: nowrap; overflow: hidden; }}
.doc svg {{ flex: none; }}
.doc span {{ overflow: hidden; text-overflow: ellipsis; }}
.step {{ display: flex; align-items: center; gap: 10px; margin-top: 4px; }}
.bot {{ position: relative; flex: none; border-radius: 50%; background: #fff; border: 3px solid {NAVY}; display: flex;
        align-items: center; justify-content: center; box-shadow: 0 0 0 7px rgba(18,58,122,.10); }}
.bot span {{ position: absolute; right: -10px; bottom: -8px; width: 42px; height: 42px; border-radius: 50%; background: #fff;
             border: 2.5px solid {NAVY}; display: flex; align-items: center; justify-content: center; }}
.step svg, .row svg {{ flex: none; }}
.q {{ flex: 1; display: flex; gap: 10px; align-items: flex-start; font-size: 24px; font-weight: 800; line-height: 1.25;
      background: #fff; border: 3px solid {NAVY}; border-radius: 12px; padding: 10px 14px; }}
.q img {{ flex: none; }}
.note {{ margin-top: auto; font-size: 21px; font-weight: 800; color: #4b6390; }}
.mode {{ position: relative; background: #fff; border-radius: 14px; border: 1.5px solid #d3dcea; padding: 22px 18px 14px;
         display: flex; flex-direction: column; gap: 12px; flex: 1; }}
.mode > small, .eq > small {{ position: absolute; top: -13px; left: 16px; background: {NAVY}; color: #fff; font-size: 17px;
         font-weight: 900; letter-spacing: .8px; text-transform: uppercase; border-radius: 999px; padding: 2px 12px; }}
.row {{ display: flex; align-items: center; gap: 8px; }}
.pair {{ width: 236px; display: flex; flex-direction: column; gap: 6px; font-size: 22px; font-weight: 800; }}
.pair > img {{ display: none; }}
.pair > span {{ border: 3px solid {NAVY}; border-radius: 10px; padding: 4px 12px; text-align: center; }}
.pair > b {{ text-align: center; font-size: 22px; line-height: .8; }}
.pair .doc {{ font-size: 21px; padding: 6px 10px; }}
.hist {{ display: flex; align-items: flex-end; gap: 8px; height: 130px; padding-bottom: 28px; border-bottom: 2px solid #d3dcea; }}
.bar {{ position: relative; display: flex; justify-content: center; }}
.bar i {{ width: 26px; border-radius: 6px 6px 0 0; background: {BLUE}; }}
.bar span {{ position: absolute; bottom: -28px; font-size: 20px; font-weight: 900; color: #6b7fa6; }}
.f {{ font-family: "STIX Two Text", serif; font-size: 27px; display: flex; align-items: baseline; gap: 16px; flex-wrap: wrap; }}
.f b {{ font-family: Nunito, sans-serif; font-weight: 900; color: {BLUE}; font-size: 30px; }}
.f.t {{ font-family: Nunito, sans-serif; font-size: 21px; font-weight: 800; color: #4b6390; }}
.f span {{ font-family: Nunito, sans-serif; font-size: 19px; font-weight: 800; color: #6b7fa6; }}
.mini {{ width: 112px; border: 1.5px solid #d3dcea; border-radius: 10px; padding: 6px 8px; }}
.mh {{ display: flex; align-items: center; gap: 6px; font-size: 17px; font-weight: 900; color: #6b7fa6; margin-bottom: 2px; }}
.av {{ font-style: normal; width: 26px; height: 26px; border-radius: 50%; background: {NAVY}; color: #fff; font-size: 15px;
       font-weight: 900; display: flex; align-items: center; justify-content: center; }}
.mr {{ display: flex; align-items: center; justify-content: space-between; padding: 3px 0; }}
.g {{ border-radius: 7px; padding: 1px 0; width: 50px; text-align: center; font-size: 18px; font-weight: 900; }}
.dots {{ text-align: center; color: #8a9bbd; font-weight: 900; line-height: .6; }}
.vs {{ width: 36px; height: 36px; border-radius: 50%; background: {NAVY}; color: #fff; font-size: 15px; display: flex;
       align-items: center; justify-content: center; flex: none; }}
.win {{ display: flex; flex-direction: column; align-items: center; gap: 2px; font-size: 23px; font-weight: 900; color: {BLUE}; }}
.win em {{ font-style: normal; font-size: 19px; color: #6b7fa6; }}
.eq {{ position: relative; background: #fff; border-radius: 14px; border: 1.5px solid #d3dcea; padding: 24px 18px 14px; }}
.m {{ font-family: "STIX Two Text", serif; font-size: 27px; display: flex; align-items: center; gap: 12px; }}
.frac {{ display: inline-flex; flex-direction: column; align-items: center; font-size: 23px; }}
.frac span:first-child {{ border-bottom: 2px solid {NAVY}; padding: 0 4px 3px; }}
.frac span:last-child {{ padding-top: 3px; font-family: Nunito, sans-serif; font-size: 18px; font-weight: 800; color: #4b6390; }}
.sub {{ margin-top: 8px; font-size: 20px; font-weight: 800; color: #6b7fa6; }}
.board {{ margin-top: auto; background: #fff; border-radius: 14px; border: 1.5px solid #d3dcea; padding: 12px 16px; }}
.bh {{ display: flex; align-items: center; gap: 10px; font-size: 26px; font-weight: 900; padding-bottom: 6px;
       border-bottom: 1.5px solid #e3e8f0; }}
.bh em {{ margin-left: auto; display: flex; align-items: center; gap: 6px; font-style: normal; color: {ORANGE}; font-size: 20px; }}
.pr {{ display: flex; align-items: center; gap: 8px; font-size: 21px; font-weight: 800; padding: 6px 0; border-bottom: 1px solid #eef1f6; }}
.pr img {{ flex: none; border-radius: 7px; }}
.pr span {{ flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.pr b {{ color: {BLUE}; font-size: 22px; }}
.pr em {{ font-style: normal; width: 34px; text-align: right; color: {ORANGE}; font-weight: 900; }}
.rho {{ padding-top: 8px; font-size: 20px; font-weight: 900; color: #4b6390; }}
</style></head><body>
{STAGE1}<div class="link">{arrow(34)}</div>{STAGE2}<div class="link">{arrow(34)}</div>{STAGE3}
</body></html>"""

render(HTML, "method", W, H)
