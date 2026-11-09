"""Figure 1: MTEB Gym at a glance (HTML printed to PDF/PNG with headless Chromium).

Top: the corpus, MTEB Gym (query writer and judge) and the embedding models, joined by two loops.
Middle: what the judge sees for one real SciFact query: two models' top results, each graded 0-5, and the nDCG@10 that
follows. Bottom, below "Evaluation": the gym's overall ranking of 28 models over 28 tasks against MTEB's, and the
agreement with human votes in MTEB Arena. Icons: Microsoft Fluent Emoji (MIT); logos from Hugging Face."""

import base64
import json

import numpy as np
from huggingface_hub import hf_hub_download
from playwright.sync_api import sync_playwright

from common import GENERATOR, GYM_RUNS, JUDGE, TASKS, judged, labels, ndcg, overall_agreement, ratings, records
from mteb_gym import corpus
from plots import FIGURES

TASK, QID = "SciFact", "q50"
STRONG, WEAK = "Qwen/Qwen3-Embedding-8B", "sentence-transformers/all-MiniLM-L6-v2"
HUMAN_GYM = 0.857  # plot_humans.py
NAVY, CREAM, INK, GREEN = "#123a7a", "#fbf7ef", "#14213d", "#1d8a4e"

R = records()
recs = R[(JUDGE, "pointwise", GENERATOR)]
per_task = {t: ratings(recs[t]) for t in TASKS}
models = sorted(per_task[TASKS[0]])
gym = sorted(models, key=lambda m: -np.mean([per_task[t][m] for t in TASKS]))
mteb = sorted(models, key=lambda m: -np.mean([labels(t)[m] for t in TASKS]))
RHO = overall_agreement(per_task)
_, _, top, grades, _ = judged(recs[TASK])
docs = corpus.load(TASK).docs
QUERY_SET = recs[TASK]["config"]["query_set"].rsplit("-", 1)[0]  # the queries table names the set without its last part
query = next(q for q in map(json.loads, open(hf_hub_download(GYM_RUNS, "tables/queries/queries.jsonl",
                                                             repo_type="dataset")))
             if q["task"] == TASK and q["qid"] == QID and q["query_set"] == QUERY_SET)
task_board = sorted(ratings(recs[TASK]).items(), key=lambda x: -x[1])
task_place = {m: (i + 1, v) for i, (m, v) in enumerate(task_board)}
nd = {m: 100 * ndcg(top[m][QID], grades[QID], grades[QID].values(), 10) for m in (STRONG, WEAK)}


def b64(path, mime):
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def icon(name, size, style=""):
    mime = "image/svg+xml" if name.endswith(".svg") else "image/png"
    return f'<img src="{b64(FIGURES / "icons" / name, mime)}" width="{size}" height="{size}" style="{style}">'


def logo(org, size=30, radius=8):
    return f'<img src="{b64(FIGURES / "logos" / f"{org}.png", "image/png")}" width="{size}" height="{size}" style="border-radius:{radius}px">'


def short(m):
    return m.split("/")[-1]


def title(d, n=34):
    text = docs[d] if isinstance(docs[d], str) else json.dumps(docs[d])
    return text[:n].rsplit(" ", 1)[0] + "…" if len(text) > n else text


def badge(g):
    if g >= 3:
        bg, fg, ring = "#1f5bd6", "#fff", "#c9d8f7"
    elif g >= 2:
        bg, fg, ring = "#8fb0ec", "#fff", "#e1e9fa"
    else:
        bg, fg, ring = "#f2d4cf", "#b3261e", "#fbeae7"
    return f'<span class="gb" style="background:{bg};color:{fg};box-shadow:0 0 0 4px {ring}">{g:.1f}</span>'


def result_card(m, x, y, r):
    rows = "".join(f'<div class="rl"><span class="rt">{title(d)}</span>{badge(grades[QID][d])}</div>'
                   for d in top[m][QID][:3])
    return (f'<div class="rc" style="left:{x}px;top:{y}px;transform:rotate({r}deg)"><div class="rh">'
            f'{logo(m.split("/")[0], 24, 6)}<span>{short(m)}</span><em>nDCG@10 {nd[m]:.0f}</em></div>{rows}</div>')


def loop(word, top_label, bottom_label, flip):
    up = "M8 58 C 70 6, 150 6, 212 50" if not flip else "M212 58 C 150 6, 70 6, 8 50"
    dn = "M212 112 C 150 164, 70 164, 8 120" if not flip else "M8 112 C 70 164, 150 164, 212 120"

    def head(x, y, d):
        return f'<path d="M{x - 15 * d} {y - 13} L{x} {y} L{x - 18 * d} {y + 5}"/>'
    heads = head(212, 50, 1) + head(8, 120, -1) if not flip else head(8, 50, -1) + head(212, 120, 1)
    return (f'<div class="loop"><div class="lt">{top_label}</div><svg width="220" height="170" viewBox="0 0 220 170" '
            f'fill="none" stroke="{INK}" stroke-width="5" stroke-linecap="round" stroke-linejoin="round">'
            f'<path d="{up}"/><path d="{dn}"/>{heads}</svg><div class="lw">{word}</div><div class="lb">{bottom_label}</div></div>')


def term(name, sub, cmd, order):
    rows = "".join(f'<div><span class="g">{i + 1}.</span> {short(m)}</div>' for i, m in enumerate(order[:3]))
    return (f'<div class="term"><div class="tn">{name}<small>{sub}</small></div><div class="to">'
            f'<div class="dim">$ {cmd}</div>{rows}<div class="dim">  … 25 more</div></div></div>')


LOGOS = "".join(logo(o, 46, 10) for o in ["Qwen", "ibm-granite", "lightonai", "sentence-transformers", "microsoft",
                                           "mteb", "BAAI", "nomic-ai", "Snowflake"])
DOWN = (f'<svg width="30" height="56" viewBox="0 0 30 56"><path d="M15 2v44M6 38l9 10 9-10" stroke="{INK}" '
        f'stroke-width="3" stroke-dasharray="3 6" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>')
GRAIN = ("url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='160' height='160'>"
         "<filter id='n'><feTurbulence type='fractalNoise' baseFrequency='.85' numOctaves='2' stitchTiles='stitch'/>"
         "<feColorMatrix values='0 0 0 0 1  0 0 0 0 1  0 0 0 0 1  0 0 0 .08 0'/></filter>"
         "<rect width='100%' height='100%' filter='url(%23n)'/></svg>\")")

HTML = f"""<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,600;0,700;0,800;0,900;1,600;1,800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
* {{ box-sizing: border-box; margin: 0; }}
body {{ font-family: Inter, sans-serif; color: {INK}; background: {CREAM}; display: inline-block; padding: 22px 26px; width: 1900px; }}
.top {{ display: flex; align-items: center; justify-content: space-between; }}
.box {{ border: 3px solid {INK}; border-radius: 12px; height: 300px; display: flex; flex-direction: column;
        overflow: hidden; box-shadow: 5px 5px 0 rgba(20,33,61,.12); }}
.box .hd {{ font-size: 30px; font-weight: 900; text-align: center; padding: 10px; border-bottom: 3px solid {INK}; }}
.black {{ background: #111; background-image: {GRAIN}; color: #fff; width: 430px; }}
.black .hd {{ border-bottom-color: #fff; }}
.icons {{ flex: 1; display: flex; justify-content: space-around; align-items: center; padding: 8px 14px 14px; }}
.icn {{ display: flex; flex-direction: column; align-items: center; gap: 10px; font-size: 25px; font-weight: 900; }}
.icn small {{ font-size: 15px; font-weight: 700; color: #a7adb8; margin-top: -8px; }}
.gym {{ background: #fff8ec; width: 470px; align-items: center; padding: 18px 22px; gap: 4px; }}
.gym h2 {{ font-size: 40px; font-weight: 900; display: flex; align-items: center; gap: 12px; }}
.roles {{ display: flex; flex-direction: column; gap: 12px; width: 100%; margin-top: 14px; }}
.role {{ display: flex; align-items: center; gap: 14px; background: #fff; border: 2px solid #e6dcc8; border-radius: 12px;
         padding: 8px 14px; }}
.role b {{ display: block; font-size: 20px; font-weight: 900; }}
.role span {{ display: flex; align-items: center; gap: 7px; font-size: 15.5px; font-weight: 700; color: #4b5873; }}
.white {{ background: #fff; width: 430px; }}
.grid {{ display: grid; grid-template-columns: repeat(3, 46px); gap: 12px; }}
.loop {{ position: relative; width: 230px; height: 300px; display: flex; align-items: center; justify-content: center; }}
.lt, .lb {{ position: absolute; width: 100%; text-align: center; font-size: 18px; font-style: italic; font-weight: 600;
            line-height: 1.2; color: #2c3a58; }}
.lt {{ top: 14px; }} .lb {{ bottom: 14px; }}
.lw {{ position: absolute; font-size: 36px; font-weight: 800; font-style: italic; }}
.zoom {{ display: flex; align-items: center; gap: 26px; margin-top: 8px; }}
.soft {{ position: relative; flex: 1; height: 290px; border-radius: 22px; background: #e7eef9; background-image: {GRAIN}; }}
.soft .tag {{ position: absolute; left: 22px; top: 16px; font-size: 15px; font-weight: 800; letter-spacing: .6px;
              color: {NAVY}; text-transform: uppercase; display: flex; align-items: center; gap: 8px; }}
.qc {{ position: absolute; left: 22px; top: 58px; width: 330px; background: #fff; border: 2.5px solid {INK};
       border-radius: 14px; padding: 14px 16px; font-size: 19px; font-weight: 800; line-height: 1.35;
       box-shadow: 0 10px 22px rgba(20,33,61,.14); transform: rotate(-2deg); }}
.qc small {{ display: flex; align-items: center; gap: 7px; font-size: 13px; color: #5d6f93; font-weight: 800;
             margin-bottom: 6px; letter-spacing: .4px; }}
.rc {{ position: absolute; width: 420px; background: #fff; border: 2px solid {INK}; border-radius: 14px;
       box-shadow: 0 10px 22px rgba(20,33,61,.14); padding-bottom: 6px; }}
.rh {{ display: flex; align-items: center; gap: 9px; font-size: 17px; font-weight: 900; padding: 9px 14px;
       border-bottom: 1.5px solid #edf1f7; margin-bottom: 4px; }}
.rh em {{ margin-left: auto; font-style: normal; font-size: 14px; font-weight: 800; color: #4b5873; }}
.rl {{ display: flex; align-items: center; gap: 12px; padding: 6px 16px; font-size: 15px; font-weight: 600; }}
.rt {{ flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; color: #3c4a66; }}
.gb {{ display: inline-flex; align-items: center; justify-content: center; width: 44px; height: 27px; border-radius: 14px;
       font-size: 14.5px; font-weight: 900; flex: none; }}
.judgebadge {{ position: absolute; right: 26px; top: 14px; display: flex; align-items: center; gap: 10px;
               font-size: 16px; font-weight: 900; color: {NAVY}; }}
.judgebadge .c {{ width: 74px; height: 74px; border-radius: 50%; background: #fff; border: 3px solid {NAVY};
                  display: flex; align-items: center; justify-content: center; box-shadow: 0 6px 14px rgba(18,58,122,.18); }}
.out {{ position: absolute; left: 1450px; top: 96px; width: 360px; background: #fff; border: 2px solid {INK};
        border-radius: 14px; box-shadow: 0 10px 22px rgba(20,33,61,.14); padding: 10px 16px; }}
.oh {{ display: flex; align-items: center; gap: 9px; font-size: 17px; font-weight: 900; padding-bottom: 8px;
       border-bottom: 1.5px solid #edf1f7; margin-bottom: 4px; }}
.or {{ display: flex; align-items: center; gap: 10px; font-size: 16px; font-weight: 700; padding: 5px 0; }}
.or.dim {{ color: #9aa6bf; padding: 0 0 0 6px; line-height: .9; }}
.or .k {{ width: 24px; text-align: right; color: #9aa6bf; font-weight: 900; }}
.or .n {{ flex: 1; white-space: nowrap; }}
.or b {{ color: #1f5bd6; font-weight: 900; }}
.ofoot {{ font-size: 13px; font-weight: 700; color: #5d6f93; margin-top: 4px; }}
.arr2 {{ position: absolute; left: 1388px; top: 150px; font-size: 44px; font-weight: 300; color: {INK}; }}
.sep {{ position: relative; margin: 64px 90px 0; border-top: 2px solid #b8bec9; }}
.sep span {{ position: absolute; left: 50%; transform: translateX(-50%); font-size: 26px; font-weight: 800;
             font-style: italic; color: #3c4a66; background: {CREAM}; padding: 0 16px; white-space: nowrap; }}
.sep .a {{ top: -40px; }} .sep .b {{ top: 8px; }}
.eval {{ display: flex; align-items: center; gap: 22px; margin-top: 58px; }}
.term {{ flex: 1; display: flex; background: #161616; border-radius: 12px; overflow: hidden; height: 168px; }}
.tn {{ width: 230px; display: flex; flex-direction: column; justify-content: center; align-items: center; color: #fff;
       font-size: 29px; font-weight: 900; border-right: 1.5px solid #333; text-align: center; line-height: 1.1; }}
.tn small {{ font-size: 15px; font-weight: 700; color: #a7adb8; margin-top: 6px; }}
.to {{ font-family: 'JetBrains Mono', monospace; font-size: 19px; color: #e5e7eb; padding: 16px 24px;
       display: flex; flex-direction: column; justify-content: center; gap: 4px; }}
.to .g {{ color: #4ade80; }} .to .dim {{ color: #8a93a6; }}
.mid {{ width: 210px; display: flex; flex-direction: column; align-items: center; gap: 6px; }}
.rho {{ font-size: 40px; font-weight: 900; color: {NAVY}; white-space: nowrap; }}
.mid small {{ font-size: 15px; font-weight: 700; color: #4b5873; text-align: center; line-height: 1.25; }}
.human {{ margin-top: 6px; display: inline-flex; align-items: center; gap: 8px; background: #fff; border: 2px solid {INK};
          border-radius: 999px; padding: 6px 14px; font-size: 15px; font-weight: 800; white-space: nowrap; }}
.human b {{ color: {NAVY}; font-size: 18px; }}
</style></head><body>

<div class="top">
  <div class="box black"><div class="hd">Your corpus</div><div class="icons">
    <div class="icn">{icon("card_file_box_3d.png", 104)}Documents<small>no labels</small></div>
    <div class="icn">{icon("label_3d.png", 96)}Task<small>one sentence</small></div></div></div>
  {loop("Write", "sample<br>documents", "realistic<br>queries", flip=True)}
  <div class="box gym"><h2>{icon("flexed_biceps_3d_default.png", 54)}MTEB Gym</h2>
    <div class="roles">
      <div class="role">{icon("writing_hand_3d_default.png", 56)}<div><b>Query writer</b><span>{logo("deepseek-ai", 20, 5)}DeepSeek-V4-Flash</span></div></div>
      <div class="role">{icon("balance_scale_3d.png", 56)}<div><b>Judge</b><span>{logo("autotrust", 20, 5)}JEV-27B · grades 0–5</span></div></div>
    </div></div>
  {loop("Retrieve", "100 queries<br>per task", "each model's<br>top 10", flip=False)}
  <div class="box white"><div class="hd">Embedding models</div><div class="icons">
    <div class="icn">{icon("robot_3d.png", 104)}28 models<small style="color:#5d6f93">as in mteb</small></div>
    <div class="grid">{LOGOS}</div></div></div>
</div>

<div style="display:flex;justify-content:center;margin:2px 0 -2px">{DOWN}</div>

<div class="zoom"><div class="soft">
  <div class="tag">{icon("magnifying_glass_tilted_right_flat.svg", 20)}One query on {TASK}, judged</div>
  <div class="qc"><small>{icon("writing_hand_flat_default.svg", 16)}GENERATED QUERY</small>{query["text"]}</div>
  {result_card(STRONG, 420, 54, -1.5)}
  {result_card(WEAK, 870, 92, 1.8)}
  <div class="out"><div class="oh">{icon("trophy_3d.png", 28)}Mean over 100 queries</div>
    <div class="or"><span class="k">{task_place[STRONG][0]}</span>{logo("Qwen", 22, 5)}<span class="n">{short(STRONG)}</span><b>{task_place[STRONG][1]:.1f}</b></div>
    <div class="or dim">⋮</div>
    <div class="or"><span class="k">{task_place[WEAK][0]}</span>{logo("sentence-transformers", 22, 5)}<span class="n">{short(WEAK)}</span><b>{task_place[WEAK][1]:.1f}</b></div>
    <div class="ofoot">{TASK} ranking of 28 models</div></div>
  <div class="arr2">→</div>
  <div class="judgebadge">JEV-27B<div class="c">{icon("balance_scale_3d.png", 50)}</div></div>
</div></div>

<div class="sep"><span class="a">Inference</span><span class="b">Evaluation</span></div>

<div class="eval">
  {term("MTEB Gym", "no queries, no labels", "mteb-gym rank   # 28 tasks", gym)}
  <div class="mid"><div class="rho">ρ = {RHO:.2f}</div>{icon("check_mark_button_3d.png", 40)}
    <small>same ranking of<br>28 models</small>
    <div class="human">{icon("busts_in_silhouette_3d.png", 22)}human votes <b>{HUMAN_GYM:.2f}</b></div></div>
  {term("MTEB", "human relevance labels", "mteb leaderboard   # same tasks", mteb)}
</div>
</body></html>"""

(FIGURES / "figure1.html").write_text(HTML)
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1960, "height": 1200}, device_scale_factor=2)
    page.set_content(HTML, wait_until="networkidle")
    size = page.locator("body").bounding_box()
    width, height = size["width"], size["height"]
    page.set_viewport_size({"width": int(width) + 1, "height": int(height) + 1})
    page.pdf(path=str(FIGURES / "figure1.pdf"), width=f"{width}px", height=f"{height}px", print_background=True,
             page_ranges="1")
    page.screenshot(path=str(FIGURES / "figure1.png"), clip={"x": 0, "y": 0, "width": width, "height": height})
    browser.close()
print("wrote", FIGURES / "figure1.pdf")
