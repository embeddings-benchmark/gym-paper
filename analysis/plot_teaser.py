"""Figure 1 (teaser), after AlpacaFarm's first figure: the use case on top, what MTEB Gym does inside a dashed box,
the actors on the side, and a comparison with MTEB's human-labeled evaluation below.

Numbers: overall agreement with MTEB's ranking (plot_overall.py) and agreement with human votes in MTEB Arena
(plot_humans.py), both from the published records. Icons: Microsoft Fluent Emoji (MIT), in figures/icons."""

import base64

from playwright.sync_api import sync_playwright

from common import GENERATOR, JUDGE, TASKS, overall_agreement, ratings, records
from plots import FIGURES

R = records()
RHO = overall_agreement({t: ratings(R[(JUDGE, "pointwise", GENERATOR)][t]) for t in TASKS})
HUMAN_GYM, HUMAN_MTEB = 0.857, 0.762  # plot_humans.py: Spearman with the MTEB Arena ranking, 8 models
NAVY, CREAM, INK = "#123a7a", "#fbf7ef", "#14213d"


def icon(name, size, style=""):
    data = base64.b64encode((FIGURES / "icons" / name).read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}" width="{size}" height="{size}" style="{style}">'


ARROW = (f'<svg width="44" height="22" viewBox="0 0 44 22"><path d="M3 11h34M29 4l8 7-8 7" stroke="{INK}" '
         f'stroke-width="2.6" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>')
YES = icon("check_mark_button_3d.png", 30)
NO = icon("cross_mark_3d.png", 26)

HTML = f"""<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Nunito:wght@600;700;800;900&display=swap" rel="stylesheet">
<style>
* {{ box-sizing: border-box; margin: 0; }}
body {{ font-family: Nunito, sans-serif; color: {INK}; background: {CREAM}; display: inline-block; padding: 24px 28px; }}
.top {{ display: flex; align-items: center; justify-content: center; gap: 16px; }}
.tb {{ display: flex; flex-direction: column; align-items: center; gap: 6px; background: #fff; border: 2px solid {INK};
       border-radius: 14px; padding: 12px 20px; width: 300px; font-size: 17px; font-weight: 800; text-align: center;
       box-shadow: 0 6px 14px rgba(20,33,61,.08); }}
.tb.gym {{ border-style: dashed; border-color: {NAVY}; background: #eef3fb; }}
.tb span {{ font-size: 14px; font-weight: 700; color: #5d6f93; }}
.funnel {{ display: flex; justify-content: center; margin: 0; height: 30px; }}
.main {{ display: flex; gap: 22px; align-items: stretch; }}
.legend {{ display: flex; flex-direction: column; justify-content: center; gap: 20px; width: 170px; }}
.actor {{ display: flex; align-items: center; gap: 10px; font-size: 16px; font-weight: 800; line-height: 1.15; }}
.actor span {{ display: block; font-size: 13px; font-weight: 700; color: #5d6f93; }}
.box {{ flex: 1; border: 2.5px dashed {NAVY}; border-radius: 18px; padding: 16px 20px 18px; background: #eef3fb; }}
.box h2 {{ text-align: center; font-size: 24px; font-weight: 900; color: {NAVY}; margin-bottom: 12px;
           display: flex; align-items: center; justify-content: center; gap: 10px; }}
.steps {{ display: flex; align-items: center; gap: 12px; }}
.step {{ flex: 1; display: flex; flex-direction: column; align-items: center; gap: 10px; }}
.card {{ background: #fff; border: 2px solid {INK}; border-radius: 14px; width: 100%; height: 118px;
         display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 6px;
         box-shadow: 0 6px 14px rgba(20,33,61,.08); padding: 8px; }}
.card .ics {{ display: flex; align-items: center; gap: 8px; font-size: 26px; font-weight: 900; color: #5d6f93; }}
.card .t {{ font-size: 13.5px; font-weight: 700; color: #3c4a66; text-align: center; line-height: 1.25; }}
.step b {{ font-size: 19px; font-weight: 900; text-align: center; line-height: 1.15; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 20px; background: #fff; font-size: 16.5px;
         box-shadow: 0 6px 14px rgba(20,33,61,.06); }}
th, td {{ border: 1.5px solid {INK}; padding: 9px 12px; text-align: center; }}
th {{ font-weight: 800; background: #f3f6fb; }}
td {{ font-weight: 700; }}
td.name {{ font-weight: 900; text-align: left; white-space: nowrap; }}
td.name div {{ display: flex; align-items: center; gap: 10px; }}
td .big {{ font-size: 20px; font-weight: 900; color: {NAVY}; }}
tr.gym td {{ background: #eef3fb; }}
</style></head><body>

<div class="top">
  <div class="tb">{icon("card_file_box_3d.png", 40)}Your documents<span>no queries, no labels</span></div>{ARROW}
  <div class="tb gym">{icon("flexed_biceps_3d_default.png", 40)}Rank models in MTEB Gym<span>an LLM writes queries and judges</span></div>{ARROW}
  <div class="tb">{icon("rocket_3d.png", 40)}Deploy the best model<span>for your documents</span></div>
</div>

<div class="funnel"><svg width="1000" height="30" viewBox="0 0 1000 30"><path d="M370 0 L120 28 M630 0 L880 28"
  stroke="{NAVY}" stroke-width="2" stroke-dasharray="6 6" fill="none"/></svg></div>

<div class="main">
  <div class="legend">
    <div class="actor">{icon("writing_hand_3d_default.png", 46)}<div>Query writer<span>an LLM</span></div></div>
    <div class="actor">{icon("robot_3d.png", 46)}<div>Embedding models<span>the candidates</span></div></div>
    <div class="actor">{icon("balance_scale_3d.png", 46)}<div>Judge<span>an LLM</span></div></div>
  </div>
  <div class="box">
    <h2>MTEB Gym</h2>
    <div class="steps">
      <div class="step"><div class="card"><div class="ics">{icon("card_file_box_3d.png", 44)}→{icon("writing_hand_3d_default.png", 44)}</div>
        <div class="t">reads documents,<br>writes realistic queries</div></div><b>Write queries</b></div>
      {ARROW}
      <div class="step"><div class="card"><div class="ics">{icon("robot_3d.png", 44)}{icon("robot_3d.png", 44)}{icon("robot_3d.png", 44)}</div>
        <div class="t">every model retrieves<br>its top 10 for each query</div></div><b>Retrieve</b></div>
      {ARROW}
      <div class="step"><div class="card"><div class="ics">{icon("balance_scale_3d.png", 44)}<span style="font-size:20px">0–5</span></div>
        <div class="t">the judge grades<br>every retrieved document</div></div><b>Judge</b></div>
      {ARROW}
      <div class="step"><div class="card"><div class="ics">{icon("trophy_3d.png", 46)}</div>
        <div class="t">by nDCG@10<br>with the judge's grades</div></div><b>Rank</b></div>
    </div>
  </div>
</div>

<table>
  <tr><th style="width:270px"></th><th>Queries</th><th>Relevance labels</th><th>Works on your own corpus</th>
      <th>Agreement with MTEB<br>(28 models, 28 tasks)</th><th>Agreement with human<br>votes (MTEB Arena)</th></tr>
  <tr class="gym"><td class="name"><div>{icon("flexed_biceps_3d_default.png", 30)}MTEB Gym</div></td>
      <td>generated</td><td>none</td><td>{YES}</td><td><span class="big">{RHO:.2f}</span></td>
      <td><span class="big">{HUMAN_GYM:.2f}</span></td></tr>
  <tr><td class="name"><div>{icon("busts_in_silhouette_3d.png", 30)}MTEB (human labels)</div></td>
      <td>written by people</td><td>by people, per query</td><td>{NO}</td><td>reference</td>
      <td><span class="big">{HUMAN_MTEB:.2f}</span></td></tr>
</table>
</body></html>"""

(FIGURES / "teaser.html").write_text(HTML)
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1500, "height": 900}, device_scale_factor=2)
    page.set_content(HTML, wait_until="networkidle")
    size = page.locator("body").bounding_box()
    width, height = size["width"], size["height"]
    page.set_viewport_size({"width": int(width) + 1, "height": int(height) + 1})
    page.pdf(path=str(FIGURES / "teaser.pdf"), width=f"{width}px", height=f"{height}px", print_background=True,
             page_ranges="1")
    page.screenshot(path=str(FIGURES / "teaser.png"), clip={"x": 0, "y": 0, "width": width, "height": height})
    browser.close()
print("wrote", FIGURES / "teaser.pdf")
