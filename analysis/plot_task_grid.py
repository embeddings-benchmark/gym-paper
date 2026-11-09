"""Supporting figure: agreement with MTEB on each of the 28 tasks, as a grid of tiles.

Each tile is a task; its number is the Spearman correlation between the gym's ranking of the 28 models (JEV-27B,
pointwise, generated queries) and MTEB's ranking on the task's labels. Tasks below .6 carry the cause found in the
failure analysis (sections/failure_analysis.tex)."""

from common import GENERATOR, JUDGE, TASKS, agreement, overall_agreement, ratings, records
from plots import CREAM, GRAIN, NAMES, NAVY, ORANGE, line_icon, render

R = records()
per_task = {t: ratings(R[(JUDGE, "pointwise", GENERATOR)][t]) for t in TASKS}
rho = {t: agreement(per_task[t], t) for t in TASKS}
MEAN, OVERALL = sum(rho.values()) / len(rho), overall_agreement(per_task)

OTHER = ["SciFact", "NFCorpus", "FinQARetrieval", "AppsRetrieval", "AILACasedocs", "CUREv1"]
GROUPS = [("MTEB(eng, v2)", [t for t in TASKS if not t.startswith("Bright") and t not in OTHER]),
          ("BRIGHT", [t for t in TASKS if t.startswith("Bright")]),
          ("Other domains", OTHER)]
CAUSES = {  # failure_analysis.tex
    "relevance": ("scale", "judge and labels define<br>relevance differently",
                  ["BrightPonyRetrieval", "BrightAopsRetrieval", "ClimateFEVERHardNegatives"]),
    "tied": ("equal", "models nearly<br>tied on the labels", ["FEVERHardNegatives"]),
    "queries": ("pen-line", "generated queries<br>differ from the originals", ["HotpotQAHardNegatives", "BrightEconomicsRetrieval"]),
}
cause = {t: icon for icon, _, ts in CAUSES.values() for t in ts}


def fill(r):
    if r >= .8:
        return NAVY, "#fff"
    if r >= .6:
        return "#2f63c4", "#fff"
    if r >= .4:
        return "#a9c0ea", NAVY
    return "#dde3ec", NAVY


def tile(t):
    bg, fg = fill(rho[t])
    mark = f'<span class="mark">{line_icon(cause[t], 22, "#fff", 2.4)}</span>' if t in cause else ""
    value = f"{rho[t]:.2f}".replace("0.", ".").replace("-.", "−.")
    return f'<div class="tile" style="background:{bg};color:{fg}">{mark}<b>{value}</b><span>{NAMES[t]}</span></div>'


rows = "".join(f'<div class="row"><div class="lbl">{g}<small>{len(ts)} tasks</small></div>'
               f'{"".join(tile(t) for t in sorted(ts, key=lambda t: -rho[t]))}</div>' for g, ts in GROUPS)
shades = "".join(f'<div class="key"><i style="background:{fill(r)[0]}"></i>{text}</div>'
                 for r, text in [(.9, "≥ .8"), (.7, ".6 – .8"), (.5, ".4 – .6"), (0, "< .4")])
causes = "".join(f'<div class="key"><span class="mark">{line_icon(icon, 20, "#fff", 2.4)}</span>{text}</div>'
                 for icon, text, _ in CAUSES.values())

W, H = 1890, 532
HTML = f"""<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Nunito:wght@700;800;900&display=swap" rel="stylesheet">
<style>
* {{ box-sizing: border-box; margin: 0; }}
body {{ font-family: Nunito, sans-serif; color: {NAVY}; background: {CREAM}; width: {W}px; height: {H}px; padding: 26px 30px; }}
.row {{ display: flex; gap: 10px; margin-bottom: 12px; align-items: stretch; }}
.lbl {{ width: 200px; white-space: nowrap; flex: none; display: flex; flex-direction: column; justify-content: center; font-size: 25px; font-weight: 900;
        line-height: 1.1; }}
.lbl small {{ font-size: 17px; font-weight: 800; color: #4b6390; margin-top: 3px; }}
.tile {{ position: relative; width: 125px; height: 104px; border-radius: 10px; background-image: {GRAIN} !important;
         display: flex; flex-direction: column; justify-content: center; align-items: center; padding: 6px;
         box-shadow: 0 3px 8px rgba(18,58,122,.10); }}
.tile b {{ font-size: 32px; font-weight: 900; line-height: 1; }}
.tile span {{ font-size: 15px; font-weight: 800; text-align: center; line-height: 1.12; margin-top: 6px; opacity: .92; }}
.mark {{ display: inline-flex; width: 36px; height: 36px; border-radius: 50%; background: {ORANGE}; align-items: center;
         justify-content: center; box-shadow: 0 0 0 3px {CREAM}; flex: none; }}
.tile .mark {{ position: absolute; top: -10px; right: -10px; }}
.foot {{ display: flex; gap: 22px; margin-top: 22px; align-items: stretch; }}
.call {{ background: {ORANGE}; color: #fff; border-radius: 16px; padding: 14px 24px; display: flex; flex-direction: column;
         justify-content: center; }}
.call b {{ font-size: 21px; font-weight: 900; letter-spacing: 1.2px; }}
.call span {{ font-size: 27px; font-weight: 900; line-height: 1.2; }}
.legend {{ flex: none; background: #fff; border-radius: 16px; border: 1.5px solid #e6e1d6; padding: 14px 24px;
           display: grid; grid-template-columns: repeat(2, auto); gap: 10px 28px; align-content: center; }}
.legend.c {{ flex: 1; grid-template-columns: repeat(3, auto); }}
.key {{ display: flex; align-items: center; gap: 12px; font-size: 19px; font-weight: 800; line-height: 1.15; }}
.key i {{ width: 40px; height: 30px; border-radius: 7px; flex: none; }}
</style></head><body>
{rows}
<div class="foot">
  <div class="call"><b>28 TASKS</b><span>mean per task {MEAN:.2f}<br>overall ranking ρ = {OVERALL:.2f}</span></div>
  <div class="legend">{shades}</div>
  <div class="legend c">{causes}</div>
</div>
</body></html>"""

render(HTML, "task_grid", W, H)
