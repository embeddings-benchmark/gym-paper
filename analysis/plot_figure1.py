"""Figure 1: MTEB Gym in four steps, drawn on real SciFact data (HTML printed to PDF/PNG with headless Chromium).

Given a corpus, an LLM writes queries from sampled documents; every embedding model retrieves its top 10 and an LLM
judge grades each result 0-5; the models are ranked by nDCG@10 with those grades, which matches MTEB's ranking.
Icons: Microsoft Fluent Emoji (MIT) and Lucide (ISC); logos from Hugging Face."""

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


def actor(action, x, y):
    """An LLM step's actor, as SWE-Sweep draws its agent: one robot, with what it does as a small badge."""
    return (f'<div class="actor" style="left:{x}px;top:{y}px">{emoji("robot", 70)}'
            f'<span class="act">{line(action, 22, "#fff")}</span></div>')


def qcard(q, x, y, r, main=False):
    return (f'<div class="qcard{" main" if main else ""}" style="left:{x}px;top:{y}px;transform:rotate({r}deg)">'
            f'{emoji("magnifying_glass_tilted_right", 28)}<span>{q["text"]}</span></div>')


CHECK = ('<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="3.4" '
         'stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>')


def scoreboard():
    a = pair["score_a"]
    return (f'<div class="score"><span class="sw"><i class="av">A</i>wins</span><b>{a:.2f}</b>'
            f'<div class="sb"><i style="width:{100 * a:.0f}%"></i></div><b class="lo">{1 - a:.2f}</b>'
            f'<span class="sw"><i class="av lo">B</i></span><small>pairwise</small></div>')


def pill(g):
    bg, fg = ("#e3ecfd", "#1f5bd6") if g >= 2 else ("#fde4df", "#d1342b")
    return f'<b class="pill" style="background:{bg};color:{fg}">{g:.1f}</b>'


def judging():
    """Pointwise judging, the default: the judge grades every result in each model's top 10."""
    def card(m, tag, x, win=False):
        rows = "".join(f'<div class="gr">{line("file-text", 20, "#8fb4f0")}<span>{title(d, 40)}</span>{pill(grades[QID][d])}</div>'
                       for i, d in enumerate(top[m][QID][:3]))
        return (f'<div class="lcard{" won" if win else ""}" style="left:{x}px"><div class="lh"><i class="av">{tag[-1]}</i>{tag}<small>top 10</small></div>'
                f'{rows}<div class="gmore">… 7 more</div></div>')
    return (f'<div class="qstrip">{emoji("magnifying_glass_tilted_right", 26)}<span>{query["text"]}</span></div>'
            + card(STRONG, "Model A", 1166, pair["score_a"] > .5) + '<div class="vsb">vs</div>' + card(WEAK, "Model B", 1474) + SCORE)


def dashed(path):
    return f'<path d="{path}" marker-end="url(#h)"/>'


SCORE = scoreboard()
orgs = ["Qwen", "Snowflake", "microsoft", "google", "nomic-ai"]
CROWD = "".join(logo(o, 38, "crowd") for o in orgs) + f'<span class="more">+{len(models) - len(orgs)}</span>'
FILES = "".join(f'<div class="file{" seed" if d == SEED else ""}">{line("file-text", 28, "#9fc0f5")}<span>{title(d, 22)}</span>'
                f'</div>' for d in [SEED, *[d for d in list(docs)[:4] if d != SEED][:3]])
MEDALS = ["1st_place_medal", "2nd_place_medal", "3rd_place_medal"]
BOARD = "".join(f'<div class="row"><span class="k">{emoji(MEDALS[i], 34)}</span>'
                f'{logo(m.split("/")[0], 30)}<span class="nm">{short(m)}</span><b>{score[m]:.1f}</b><span class="mt">#{mteb.index(m) + 1}</span></div>'
                for i, m in enumerate(gym[:3])) + '<div class="row rest">… 25 more</div>'
GRAIN = ("url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='160' height='160'>"
         "<filter id='n'><feTurbulence type='fractalNoise' baseFrequency='.85' numOctaves='2' stitchTiles='stitch'/>"
         "<feColorMatrix values='0 0 0 0 1  0 0 0 0 1  0 0 0 0 1  0 0 0 .07 0'/></filter>"
         "<rect width='100%' height='100%' filter='url(%23n)'/></svg>\")")
W, H = 2470, 560
ARROWS = f"""<svg class="lines" width="{W}" height="{H}" viewBox="0 0 {W} {H}" fill="none" stroke="{NAVY}" stroke-width="3"
  stroke-dasharray="9 8" stroke-linecap="round">
  <defs><marker id="h" viewBox="0 0 12 12" refX="9" refY="6" markerWidth="8" markerHeight="8" orient="auto">
    <path d="M2 2L10 6L2 10" fill="none" stroke="{NAVY}" stroke-width="2" stroke-dasharray="none"/></marker></defs>
  {dashed("M418 172 C 440 114, 470 96, 500 96")}{dashed("M444 224 L 500 224")}{dashed("M418 276 C 440 334, 470 352, 500 352")}
  
  {dashed("M1806 186 C 1798 176, 1790 172, 1774 172")}{dashed("M1806 266 C 1798 300, 1790 310, 1774 310")}
  <path d="M1044 224 L 1124 224" stroke-dasharray="none" stroke-width="4" marker-end="url(#h)"/>
  <path d="M1900 224 L 1918 224" stroke-dasharray="none" stroke-width="4" marker-end="url(#h)"/>
</svg>"""

HTML = f"""<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Nunito:wght@600;700;800;900&display=swap" rel="stylesheet">
<style>
* {{ box-sizing: border-box; margin: 0; }}
body {{ font-family: Nunito, sans-serif; color: {NAVY}; background: {CREAM}; width: {W}px; height: {H}px; position: relative;
        overflow: hidden; }}
.lines {{ position: absolute; left: 0; top: 0; z-index: 1; }}
.corpus {{ position: absolute; left: 30px; top: 44px; width: 380px; height: 360px; border-radius: 12px; background: {NAVY};
           background-image: {GRAIN}; color: #fff; padding: 18px 26px; box-shadow: 0 8px 20px rgba(18,58,122,.18); }}
.corpus h3 {{ display: flex; align-items: center; gap: 14px; font-size: 30px; font-weight: 800; padding-bottom: 12px;
              border-bottom: 1.5px solid rgba(255,255,255,.25); margin-bottom: 10px; }}
.file svg {{ flex: none; }}
.file span {{ overflow: hidden; text-overflow: ellipsis; }}
.file {{ display: flex; align-items: center; gap: 14px; font-size: 23px; font-weight: 700; padding: 11px 0; white-space: nowrap; }}
.panel {{ position: absolute; top: 24px; height: 404px; border-radius: 18px; background: {PANEL}; background-image: {GRAIN}; }}
.actor {{ position: absolute; z-index: 3; width: 112px; height: 112px; border-radius: 50%; background: #fff;
          border: 3px solid {NAVY}; display: flex; align-items: center; justify-content: center;
          box-shadow: 0 0 0 9px rgba(18,58,122,.10), 0 8px 18px rgba(18,58,122,.2); }}
.badge {{ position: absolute; right: -6px; bottom: -2px; background: #fff; border-radius: 10px; padding: 3px; line-height: 0;
          box-shadow: 0 2px 6px rgba(18,58,122,.3); }}
.who {{ position: absolute; top: -42px; left: 50%; transform: translateX(-50%); white-space: nowrap; font-size: 18px;
        font-weight: 900; background: #fff; color: {NAVY}; border: 2.5px solid {NAVY}; padding: 2px 12px; border-radius: 999px; }}
.qcard {{ position: absolute; z-index: 2; width: 466px; background: #fff; border-radius: 12px; padding: 14px 18px;
          display: flex; gap: 12px; align-items: flex-start; font-size: 22px; font-weight: 800; line-height: 1.3;
          box-shadow: 0 8px 20px rgba(18,58,122,.14); border: 1.5px solid #d3dcea; }}
.qcard img, .qcard svg, .qstrip svg {{ flex: none; margin-top: 1px; }}
.qcard.main {{ border: 3px solid {NAVY}; font-size: 21px; }}
.jcard {{ position: absolute; z-index: 2; top: 46px; width: 280px; height: 364px; background: #fff; border-radius: 14px;
          padding: 18px 20px; box-shadow: 0 8px 20px rgba(18,58,122,.14); border: 1.5px solid #d3dcea;
          display: flex; flex-direction: column; }}
.jcard h4 {{ font-size: 26px; font-weight: 900; line-height: 1; }}
.jcard > small {{ font-size: 16px; font-weight: 700; color: #4b6390; margin: 4px 0 14px; }}
.doc {{ display: flex; gap: 10px; align-items: flex-start; font-size: 17px; font-weight: 800; line-height: 1.25;
        border: 1.5px solid #e1e7f0; border-radius: 10px; padding: 10px 12px; }}
.doc img {{ flex: none; }}
.note {{ font-size: 14px; font-weight: 800; color: #8a9bbd; text-align: center; margin-top: 7px; }}
.dots {{ display: flex; justify-content: space-between; margin-top: 16px; }}
.dots span {{ width: 34px; height: 34px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
              font-size: 16px; font-weight: 900; box-shadow: inset 0 0 0 1.5px rgba(31,91,214,.35); }}
.res {{ margin-top: auto; display: flex; align-items: center; gap: 12px; border-top: 1.5px solid #eef1f6; padding-top: 12px; }}
.res b {{ font-size: 44px; font-weight: 900; color: #1f5bd6; line-height: 1; }}
.res span {{ font-size: 15px; font-weight: 800; color: #4b6390; line-height: 1.2; }}
.entry {{ display: flex; align-items: center; gap: 7px; border: 1.5px solid #e1e7f0; border-radius: 10px; padding: 9px 9px;
          font-size: 15px; font-weight: 900; white-space: nowrap; }}
.entry span {{ flex: 1; overflow: hidden; text-overflow: ellipsis; }}
.entry.win {{ border: 2.5px solid #1f5bd6; background: #f3f7ff; }}
.entry {{ position: relative; }}
.entry > img:last-child:not(:first-child) {{ position: absolute; right: -11px; top: -11px; }}
.vs {{ text-align: center; font-size: 20px; font-weight: 900; color: #8a9bbd; margin: 6px 0; }}
.jcard h4 {{ display: flex; align-items: center; gap: 9px; }}
.jcard.pw {{ width: 276px; }}
.jcard.pb {{ width: 404px; }}
.hist {{ display: flex; justify-content: space-between; align-items: flex-end; height: 92px; margin: 14px 4px 30px;
         border-bottom: 2px solid #d3dcea; }}
.bar {{ width: 30px; display: flex; flex-direction: column; align-items: center; justify-content: flex-end; position: relative; }}
.bar i {{ width: 30px; border-radius: 6px 6px 0 0; background: #1f5bd6; }}
.bar span {{ position: absolute; bottom: -24px; font-size: 16px; font-weight: 900; color: #6b7fa6; }}
.pw .res {{ margin-top: auto; }}
.arena {{ display: flex; align-items: stretch; gap: 30px; position: relative; }}
.side {{ position: relative; flex: 1; border: 1.5px solid #e1e7f0; border-radius: 12px; padding: 8px 10px 10px;
         display: flex; flex-direction: column; align-items: center; gap: 5px; }}
.side.won {{ border: 2.5px solid #1f5bd6; background: #f3f7ff; }}
.side .win {{ position: absolute; right: -10px; top: -12px; line-height: 0; }}
.tag {{ font-size: 13px; font-weight: 900; letter-spacing: .6px; text-transform: uppercase; color: #8a9bbd; }}
.name {{ font-size: 15px; font-weight: 900; text-align: center; line-height: 1.15; min-height: 36px; display: flex; align-items: center; }}
.hit {{ display: flex; align-items: center; gap: 6px; width: 100%; }}
.hit em {{ font-style: normal; font-size: 12px; font-weight: 900; color: #8a9bbd; width: 10px; }}
.hit i {{ height: 8px; border-radius: 4px; background: #d6deeb; }}
.arena .vs {{ position: absolute; left: 50%; top: 50%; transform: translate(-50%, -50%); z-index: 2; width: 46px; height: 46px;
              border-radius: 50%; background: {ORANGE}; color: #fff; font-size: 17px; font-weight: 900; margin: 0;
              display: flex; align-items: center; justify-content: center; box-shadow: 0 0 0 4px #fff; }}
.split {{ display: flex; height: 38px; border-radius: 10px; overflow: hidden; margin-top: auto; font-size: 19px; font-weight: 900; }}
.split div {{ display: flex; align-items: center; gap: 8px; padding: 0 12px; }}
.split div:first-child {{ background: #1f5bd6; color: #fff; }}
.split div:last-child {{ flex: 1; justify-content: flex-end; background: #f6d4d1; color: #b3261e; }}
.split b {{ font-weight: 900; opacity: .8; }}
.pb .note {{ margin-top: 6px; }}
.io {{ position: absolute; z-index: 2; width: 258px; height: 160px; background: #fff; border-radius: 14px; padding: 14px 16px;
        border: 1.5px solid #d3dcea; box-shadow: 0 8px 20px rgba(18,58,122,.12); display: flex; flex-direction: column; gap: 8px; }}
.io.out {{ width: 230px; }}
.chip {{ font-size: 14px; font-weight: 900; letter-spacing: .8px; color: #8a9bbd; display: flex; flex-direction: column; gap: 1px; }}
.chip span {{ font-size: 18px; letter-spacing: 0; color: {NAVY}; }}
.ln {{ display: flex; align-items: center; gap: 8px; font-size: 17px; font-weight: 800; white-space: nowrap; }}
.ln img {{ flex: none; }}
.ln span {{ overflow: hidden; text-overflow: ellipsis; }}
.ln b {{ color: {ORANGE}; font-weight: 900; }}
.owl {{ position: absolute; z-index: 3; left: 1484px; top: 112px; width: 172px; display: flex; flex-direction: column;
        align-items: center; }}
.ring {{ position: relative; width: 168px; height: 168px; border-radius: 50%; background: #fff; border: 3px solid {NAVY};
         display: flex; align-items: center; justify-content: center;
         box-shadow: 0 0 0 10px rgba(18,58,122,.10), 0 10px 22px rgba(18,58,122,.22); }}
.ring .badge {{ right: 2px; bottom: 4px; }}
.owl .who {{ position: static; transform: none; margin-top: 18px; text-align: center; line-height: 1.1; padding: 5px 16px; }}
.owl .who small {{ display: block; font-size: 14px; color: {ORANGE}; font-weight: 900; }}
.dec {{ display: flex; align-items: flex-end; justify-content: space-between; margin-top: auto; }}
.dec > b {{ font-size: 46px; margin-left: 14px; font-weight: 900; color: #1f5bd6; line-height: .9; }}
.out .hist {{ display: flex; align-items: flex-end; gap: 5px; height: 60px; margin: 0 0 18px; border: none; }}
.out .hist i {{ position: relative; width: 15px; border-radius: 4px 4px 0 0; background: #1f5bd6; }}
.out .hist span {{ position: absolute; bottom: -20px; left: 50%; transform: translateX(-50%); font-size: 13px; font-weight: 900;
                   color: #6b7fa6; font-style: normal; }}
.out .hist i span {{ opacity: 1; }}
.pick {{ display: flex; align-items: center; gap: 8px; white-space: nowrap; font-size: 22px; font-weight: 900; padding-bottom: 4px; }}
.out .split {{ height: 12px; border-radius: 6px; background: #f6c9c4; margin-top: 6px; display: block; }}
.out .split i {{ display: block; height: 100%; border-radius: 6px; background: #1f5bd6; }}
.say {{ position: absolute; z-index: 2; left: 1690px; width: 250px; background: #fff; border-radius: 22px; padding: 16px 22px;
         border: 3px solid {NAVY}; box-shadow: 0 8px 20px rgba(18,58,122,.14); }}
.say::before {{ content: ""; position: absolute; left: -22px; bottom: 14px; border: 12px solid transparent;
                border-right: 18px solid {NAVY}; border-left: 0; }}
.say::after {{ content: ""; position: absolute; left: -16px; bottom: 17px; border: 9px solid transparent;
               border-right: 14px solid #fff; border-left: 0; }}
.say.low::before {{ bottom: auto; top: 14px; }}
.say.low::after {{ bottom: auto; top: 17px; }}
.say .big {{ white-space: nowrap; display: flex; align-items: baseline; justify-content: space-between; font-size: 25px; font-weight: 900; }}
.say .big b {{ font-size: 40px; color: #1f5bd6; line-height: 1; }}
.say small {{ display: block; font-size: 15px; font-weight: 800; color: #6b7fa6; margin-top: 4px; }}
.qstrip {{ position: absolute; z-index: 2; left: 1166px; top: 44px; width: 600px; display: flex; gap: 10px; align-items: center;
           font-size: 19px; font-weight: 800; line-height: 1.25; background: #fff; border: 3px solid {NAVY}; border-radius: 12px;
           padding: 10px 14px; box-shadow: 0 8px 20px rgba(18,58,122,.12); }}
.qstrip img {{ flex: none; }}
.lcard {{ position: absolute; z-index: 2; top: 128px; width: 292px; background: #fff; border-radius: 14px; padding: 12px 16px 8px;
          border: 1.5px solid #d3dcea; box-shadow: 0 8px 20px rgba(18,58,122,.12); }}
.lh {{ display: flex; align-items: center; gap: 10px; font-size: 22px; font-weight: 900; padding-bottom: 8px;
       border-bottom: 1.5px solid #e3e8f0; }}
.lh small {{ margin-left: auto; font-size: 15px; color: #6b7fa6; font-weight: 800; }}
.gr {{ display: flex; align-items: center; gap: 8px; padding: 7px 0; border-bottom: 1px solid #f0f2f6; font-size: 16px; font-weight: 800; }}
.gr em {{ font-style: normal; color: #8a9bbd; font-weight: 900; width: 12px; flex: none; }}
.gr span {{ flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.pill {{ flex: none; width: 40px; text-align: right; font-size: 19px; font-weight: 900; color: {NAVY}; font-variant-numeric: tabular-nums; }}
.pill.lo {{ color: #9aa8c2; }}
.gmore {{ font-size: 14px; font-weight: 800; color: #8a9bbd; padding: 6px 0 2px 20px; }}
.act {{ position: absolute; right: -8px; bottom: -4px; width: 40px; height: 40px; border-radius: 50%; background: {ORANGE};
        display: flex; align-items: center; justify-content: center; box-shadow: 0 0 0 3px #fff; }}
.via {{ position: absolute; left: 1036px; top: 150px; width: 100px; text-align: center; font-size: 17px; font-weight: 900;
        line-height: 1.15; }}
.via small {{ display: block; position: absolute; top: 102px; width: 100%; font-size: 15px; color: #6b7fa6; }}
.qi {{ flex: none; width: 36px; height: 36px; border-radius: 50%; background: #e3ecfd; display: flex; align-items: center;
       justify-content: center; }}
.qstrip .qi {{ width: 32px; height: 32px; }}
.qcard svg, .qstrip svg {{ margin: 0 !important; }}
.av {{ width: 28px; height: 28px; border-radius: 50%; background: {NAVY}; color: #fff; font-style: normal; font-size: 15px; font-weight: 900;
       display: flex; align-items: center; justify-content: center; }}
.gr svg {{ flex: none; }}
.pill {{ width: 46px !important; text-align: center !important; border-radius: 8px; padding: 3px 0; font-size: 17px !important; }}
.file em {{ margin-left: auto; font-style: normal; font-size: 14px; font-weight: 900; background: {ORANGE}; color: #fff;
            border-radius: 999px; padding: 2px 10px; }}
.vsb {{ position: absolute; z-index: 3; left: 1446px; top: 194px; width: 40px; height: 40px; border-radius: 50%;
        background: {NAVY}; color: #fff; font-size: 16px; font-weight: 900; display: flex; align-items: center;
        justify-content: center; box-shadow: 0 0 0 4px {PANEL}; }}
.row img {{ flex: none; }}
.file.seed {{ background: rgba(143,180,240,.18); border-radius: 8px; margin: 0 -10px; padding-left: 10px; padding-right: 10px; }}

.crown b {{ font-weight: 900; opacity: .85; }}
.crown {{ position: absolute; top: -17px; right: 14px; display: flex; align-items: center; gap: 6px; background: #1f5bd6;
          color: #fff; font-size: 16px; font-weight: 900; border-radius: 999px; padding: 4px 14px 4px 10px;
          box-shadow: 0 3px 8px rgba(31,91,214,.3); }}
.score {{ position: absolute; z-index: 2; left: 1166px; top: 352px; width: 600px; height: 58px; background: #fff;
          border-radius: 14px; border: 1.5px solid #d3dcea; box-shadow: 0 8px 20px rgba(18,58,122,.12);
          display: flex; align-items: center; gap: 12px; padding: 0 18px; }}
.score .sw {{ display: flex; align-items: center; gap: 8px; font-size: 19px; font-weight: 900; }}
.score > b {{ font-size: 24px; font-weight: 900; color: #1f5bd6; }}
.score > b.lo {{ color: #d1342b; }}
.sb {{ flex: 1; height: 12px; border-radius: 6px; background: #fde4df; overflow: hidden; }}
.sb i {{ display: block; height: 100%; background: #1f5bd6; border-radius: 6px; }}
.av.lo {{ background: #8a9bbd; }}
.score small {{ position: absolute; top: -11px; left: 18px; background: {NAVY}; color: #fff; font-size: 12px; font-weight: 900;
                letter-spacing: .8px; text-transform: uppercase; border-radius: 999px; padding: 1px 10px; }}
.stack {{ display: flex; }}
.stack img {{ margin-left: -14px; border: 3px solid #fff; background: #fff; border-radius: 50% !important; }}
.stack img:first-child {{ margin-left: 0; }}
.actor img {{ border-radius: 14px; }}
.crowd {{ margin-left: -12px; border: 2.5px solid #fff; background: #fff; border-radius: 50% !important; object-fit: contain; padding: 4px; box-shadow: 0 2px 6px rgba(18,58,122,.2); }}
.models {{ position: absolute; left: 1000px; top: 300px; display: flex; align-items: center; padding-left: 10px; z-index: 3; }}
.more {{ margin-left: 6px; font-size: 18px; font-weight: 900; background: {CREAM}; border-radius: 8px; padding: 0 4px; }}
.board {{ position: absolute; left: 1926px; top: 44px; width: 520px; height: 360px; background: #fff; border-radius: 14px;
          border: 1.5px solid #d3dcea; box-shadow: 0 8px 20px rgba(18,58,122,.10); padding: 14px 20px; }}
.board h3 {{ display: flex; align-items: center; gap: 10px; font-size: 30px; font-weight: 900; padding-bottom: 10px;
             border-bottom: 1.5px solid #e3e8f0; margin-bottom: 4px; }}
.board h3 small {{ margin-left: auto; display: flex; align-items: center; gap: 8px; font-size: 22px; color: {ORANGE}; font-weight: 900; }}
.row {{ display: flex; align-items: center; gap: 9px; font-size: 21px; font-weight: 800; padding: 7px 0;
        border-bottom: 1px solid #eef1f6; height: 68px; }}
.row.rest {{ border: none; height: 44px; color: #8a9bbd; font-size: 19px; padding-left: 44px; }}
.row .k {{ width: 34px; line-height: 0; }}
.row .k:not(:has(img)) {{ line-height: 1; }}
.row .nm {{ flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.row b {{ font-weight: 900; color: #1f5bd6; font-size: 25px; }}
.row .mt {{ width: 46px; text-align: right; color: {ORANGE}; font-weight: 900; font-size: 23px; }}
.cap {{ position: absolute; top: 452px; text-align: center; font-size: 30px; font-weight: 800; line-height: 1.2; }}
.cap small {{ display: block; font-size: 21px; font-weight: 700; color: #4b6390; margin-top: 4px; }}
</style></head><body>
{ARROWS}
<div class="corpus"><h3>{line("folder-open", 32, "#8fb4f0")}{TASK} corpus</h3>{FILES}</div>
{actor("pen-line", 358, 168)}
<div class="panel" style="left:470px;width:560px"></div>
{qcard(queries[OTHER_QIDS[0]], 510, 54, -2)}
{qcard(query, 540, 176, 1, main=True)}
{qcard(queries[OTHER_QIDS[1]], 500, 312, -1)}
<div class="via">28 embedding<br>models<small>top 10 each</small></div>
<div class="panel" style="left:1140px;width:700px"></div>
{judging()}
{actor("scale", 1786, 168)}
<div class="board"><h3>{line("trophy", 30, NAVY)}Ranking<small>{logo("mteb", 30)}MTEB</small></h3>{BOARD}</div>
<div class="cap" style="left:30px;width:380px">Given a corpus<small>no queries, no relevance labels</small></div>
<div class="cap" style="left:470px;width:560px">an LLM writes realistic queries<small>100 per task, from sampled documents</small></div>
<div class="cap" style="left:1140px;width:700px">each model retrieves, a judge grades<small>each result graded 0–5 · or A vs B, which top 10 is better</small></div>
<div class="cap" style="left:1926px;width:520px">then ranks the models<small>same top 3 as MTEB · ρ = {RHO:.2f}</small></div>
</body></html>"""

(FIGURES / "figure1.html").write_text(HTML)
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": W, "height": H}, device_scale_factor=2)
    page.set_content(HTML, wait_until="networkidle")
    page.pdf(path=str(FIGURES / "figure1.pdf"), width=f"{W}px", height=f"{H}px", print_background=True, page_ranges="1")
    page.screenshot(path=str(FIGURES / "figure1.png"))
    browser.close()
print("wrote", FIGURES / "figure1.pdf")
