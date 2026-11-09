"""Figure: overview of MTEB Gym, drawn as a picture of the system (HTML printed to PDF with headless Chromium).

Top: the corpus, MTEB Gym with its two LLMs (query writer, judge), and the embedding models, joined by two loops.
Bottom, below "Evaluation": the gym's overall ranking against MTEB's ranking from human labels (28 models, 28 tasks)."""

import base64

import numpy as np
from playwright.sync_api import sync_playwright

from common import GENERATOR, JUDGE, TASKS, labels, overall_agreement, ratings, records
from plots import BLUE, FIGURES

R = records()
per_task = {t: ratings(R[(JUDGE, "pointwise", GENERATOR)][t]) for t in TASKS}
models = sorted(per_task[TASKS[0]])
gym = sorted(models, key=lambda m: -np.mean([per_task[t][m] for t in TASKS]))
mteb = sorted(models, key=lambda m: -np.mean([labels(t)[m] for t in TASKS]))
rho = overall_agreement(per_task)


def img(name, size, radius=8):
    data = base64.b64encode((FIGURES / "logos" / f"{name}.png").read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}" width="{size}" height="{size}" style="border-radius:{radius}px">'


def short(m):
    return m.split("/")[-1]


def ranking(order):
    return "".join(f'<div><span class="p">{i + 1}.</span> {short(m)}</div>' for i, m in enumerate(order[:3]))


DOCS_ICON = """<svg width="96" height="96" viewBox="0 0 96 96" fill="none" stroke="#fff" stroke-width="4"
 stroke-linejoin="round"><path d="M30 14h30l14 14v48H30z" fill="#2b2b2b"/><path d="M22 22h30l14 14v48H22z"
 fill="#3a3a3a"/><path d="M14 30h30l14 14v46H14z" fill="#e9e3d3" stroke="#e9e3d3"/><path d="M44 30v14h14" stroke="#b9b09a"/>
 <path d="M22 56h26M22 64h26M22 72h18" stroke="#8d846e" stroke-linecap="round"/></svg>"""
TASK_ICON = """<svg width="84" height="96" viewBox="0 0 84 96" fill="none" stroke="#fff" stroke-width="7"
 stroke-linejoin="round" stroke-linecap="round"><rect x="10" y="16" width="64" height="74" rx="8"/>
 <path d="M30 10h24v14H30z" fill="#111"/><path d="M24 44h36M24 58h36M24 72h22"/></svg>"""
GYM_ICON = """<svg width="120" height="76" viewBox="0 0 120 76" fill="none" stroke="#1a2233" stroke-width="6"
 stroke-linecap="round" stroke-linejoin="round"><rect x="8" y="22" width="12" height="32" rx="3" fill="#1a2233"/>
 <rect x="22" y="12" width="14" height="52" rx="4" fill="#fff"/><path d="M36 38h48"/>
 <rect x="84" y="12" width="14" height="52" rx="4" fill="#fff"/><rect x="100" y="22" width="12" height="32" rx="3"
 fill="#1a2233"/></svg>"""


def loop(label, top, bottom, flip=False):
    upper = "M10 52 C 60 0, 140 0, 190 44" if not flip else "M190 52 C 140 0, 60 0, 10 44"
    lower = "M190 108 C 140 160, 60 160, 10 116" if not flip else "M10 108 C 60 160, 140 160, 190 116"
    def head(x, y, dx):
        return f'<path d="M{x - 14 * dx} {y - 12} L{x} {y} L{x - 16 * dx} {y + 6}"/>'
    heads = (head(190, 44, 1) + head(10, 116, -1)) if not flip else (head(10, 44, -1) + head(190, 116, 1))
    return f"""<div class="loop"><div class="lt">{top}</div>
<svg width="200" height="160" viewBox="0 0 200 160" fill="none" stroke="#1a2233" stroke-width="5"
 stroke-linecap="round" stroke-linejoin="round"><path d="{upper}"/><path d="{lower}"/>{heads}</svg>
<div class="ll">{label}</div><div class="lb">{bottom}</div></div>"""


LOGOS = "".join(img(o, 46) for o in ["Qwen", "ibm-granite", "lightonai", "sentence-transformers", "microsoft", "mteb"])

HTML = f"""<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,500;0,600;0,700;0,800;1,500;1,700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
* {{ box-sizing: border-box; margin: 0; }}
body {{ font-family: Inter, sans-serif; color: #1a2233; background: #fff; display: inline-block; padding: 16px; }}
.top {{ display: flex; align-items: center; }}
.box {{ border: 2.5px solid #111; border-radius: 10px; width: 400px; height: 290px; overflow: hidden;
        display: flex; flex-direction: column; box-shadow: 3px 3px 0 #d6d6d6; }}
.box .hd {{ font-size: 30px; font-weight: 800; text-align: center; padding: 10px; border-bottom: 2.5px solid #111; }}
.black {{ background: #111; color: #fff; }}
.black .hd {{ border-bottom-color: #fff; }}
.icons {{ flex: 1; display: flex; justify-content: space-around; align-items: center; padding: 6px 18px 12px; }}
.icn {{ display: flex; flex-direction: column; align-items: center; gap: 14px; font-size: 26px; font-weight: 800; }}
.icn small {{ display: block; font-size: 16px; font-weight: 600; opacity: .7; margin-top: -10px; }}
.center {{ background: #fbf6ee; width: 360px; height: 290px; align-items: center; justify-content: center; gap: 4px;
           padding: 16px; }}
.center h2 {{ font-size: 36px; font-weight: 800; margin-top: 2px; }}
.roles {{ display: flex; flex-direction: column; gap: 9px; margin-top: 12px; width: 100%; }}
.role {{ display: flex; align-items: center; gap: 12px; font-size: 18px; line-height: 1.2; }}
.role b {{ display: block; font-size: 19px; }}
.role span {{ color: #3d4556; }}
.loop {{ position: relative; width: 210px; height: 290px; display: flex; flex-direction: column;
         align-items: center; justify-content: center; }}
.lt, .lb {{ font-size: 19px; font-style: italic; font-weight: 500; text-align: center; line-height: 1.2;
            position: absolute; width: 100%; }}
.lt {{ top: 22px; }} .lb {{ bottom: 22px; }}
.ll {{ position: absolute; top: 120px; font-size: 36px; font-weight: 700; font-style: italic; }}
.grid {{ display: grid; grid-template-columns: repeat(3, 46px); gap: 14px; }}
.more {{ font-size: 18px; font-weight: 700; opacity: .65; }}
.sep {{ position: relative; margin: 62px 120px 6px; border-top: 1.5px solid #b8bec9; }}
.sep span {{ position: absolute; left: 50%; transform: translateX(-50%); font-size: 26px; font-weight: 700;
             font-style: italic; color: #3d4556; background: #fff; padding: 0 14px; }}
.sep .a {{ top: -40px; }} .sep .b {{ top: 6px; }}
.down {{ display: flex; justify-content: space-between; padding: 0 180px; margin-top: -2px; height: 40px; }}
.eval {{ display: flex; align-items: center; gap: 22px; margin-top: 52px; }}
.term {{ flex: 1; display: flex; background: #1b1b1b; border-radius: 10px; overflow: hidden; height: 128px; }}
.term .nm {{ width: 210px; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center;
             font-size: 27px; font-weight: 800; color: #fff; border-right: 1.5px solid #3a3a3a; line-height: 1.15; }}
.term .nm small {{ display: block; font-size: 15px; font-weight: 600; color: #a7adb8; margin-top: 4px; }}
.term .out {{ font-family: 'JetBrains Mono', monospace; font-size: 18px; color: #e5e7eb; padding: 18px 22px;
              display: flex; flex-direction: column; justify-content: center; gap: 4px; }}
.term .out .p {{ color: #4ade80; }}
.term .out .dim {{ color: #8a93a6; }}
.mid {{ display: flex; flex-direction: column; align-items: center; gap: 4px; width: 170px; }}
.mid .rho {{ white-space: nowrap; font-size: 34px; font-weight: 800; color: {BLUE}; }}
.mid .sub {{ font-size: 15px; font-weight: 600; color: #5a6478; text-align: center; line-height: 1.2; }}
</style></head><body>

<div class="top">
  <div class="box black">
    <div class="hd">Your corpus</div>
    <div class="icons">
      <div class="icn">{DOCS_ICON}Documents<small>no labels</small></div>
      <div class="icn">{TASK_ICON}Task<small>one sentence</small></div>
    </div>
  </div>

  {loop("Write", "sample 3<br>documents", "documents and<br>task description", flip=True)}

  <div class="box center">
    {GYM_ICON}
    <h2>MTEB Gym</h2>
    <div class="roles">
      <div class="role">{img("deepseek-ai", 40)}<div><b>Query writer</b><span>writes realistic queries</span></div></div>
      <div class="role">{img("autotrust", 40)}<div><b>Judge</b><span>grades each result 0–5</span></div></div>
    </div>
  </div>

  {loop("Retrieve", "100 queries<br>per task", "each model's<br>top 10")}

  <div class="box">
    <div class="hd">Embedding models</div>
    <div class="icons">
      <div class="icn"><div class="grid">{LOGOS}</div><span class="more">+ 22 more</span></div>
    </div>
  </div>
</div>

<div class="sep"><span class="a">Inference</span><span class="b">Evaluation (agreement with MTEB)</span></div>

<div class="eval">
  <div class="term"><div class="nm">MTEB Gym<small>judge's grades</small></div>
    <div class="out"><div class="dim">$ mteb-gym rank  # 28 tasks</div>{ranking(gym)}</div></div>
  <div class="mid"><div class="rho">ρ = {rho:.2f}</div><svg width="34" height="34" viewBox="0 0 34 34">
    <circle cx="17" cy="17" r="16" fill="#22a05a"/><path d="M9 17l5.5 5.5L25 12" stroke="#fff" stroke-width="3.5"
    fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg><div class="sub">28 models<br>28 tasks</div></div>
  <div class="term"><div class="nm">MTEB<small>human relevance labels</small></div>
    <div class="out"><div class="dim">$ mteb leaderboard  # same 28 tasks</div>{ranking(mteb)}</div></div>
</div>
</body></html>"""

(FIGURES / "overview.html").write_text(HTML)
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 2200, "height": 900}, device_scale_factor=2)
    page.set_content(HTML, wait_until="networkidle")
    size = page.locator("body").bounding_box()
    width, height = size["width"], size["height"]
    page.set_viewport_size({"width": int(width) + 1, "height": int(height) + 1})
    page.pdf(path=str(FIGURES / "overview.pdf"), width=f"{width}px", height=f"{height}px", print_background=True,
             page_ranges="1")
    page.screenshot(path=str(FIGURES / "overview.png"), clip={"x": 0, "y": 0, "width": width, "height": height})
    browser.close()
print("wrote", FIGURES / "overview.pdf")
