"""Shared look for the paper's figures: large sans-serif text, thick marks, two colors, labels in the plot.

Blue is the gym with generated queries (what it uses); orange is the second series, such as the original queries."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import ROOT

FIGURES = ROOT / "figures"
BLUE, ORANGE, INK, GRAY, LIGHT = "#154aa8", "#ff5a0a", "#1a2233", "#8492a6", "#dfe2e8"  # SWE-Sweep
BLUE_TINT, BLUE_PALE, ORANGE_PALE = "#9fb4df", "#eef2fa", "#ffe6d6"
MAIN = BLUE  # the gym with generated queries
OWN = dict(color=ORANGE, marker="s", label="Original queries")
GEN = dict(color=BLUE, marker="o", label="Generated queries")
FONTSIZE = 18

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
    "font.size": FONTSIZE,
    "axes.titlesize": FONTSIZE + 1,
    "axes.labelsize": FONTSIZE,
    "xtick.labelsize": FONTSIZE - 1,
    "ytick.labelsize": FONTSIZE - 1,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": INK,
    "axes.linewidth": 1.2,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "legend.frameon": False,
    "lines.linewidth": 4,
    "lines.markersize": 13,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,
})


def save(fig, name):
    FIGURES.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"{name}.{ext}")
    print("wrote", FIGURES / f"{name}.pdf")


def tag(ax, x, y, text, color, **kw):
    """A label written in the plot, in the color of what it names."""
    ax.annotate(text, (x, y), color=color, fontsize=kw.pop("fontsize", FONTSIZE - 1), **kw)


NAMES = {
    "ArguAna": "ArguAna", "ClimateFEVERHardNegatives": "Climate-FEVER", "FiQA2018": "FiQA-2018", "SCIDOCS": "SCIDOCS",
    "TRECCOVID": "TREC-COVID", "HotpotQAHardNegatives": "HotpotQA", "Touche2020Retrieval.v3": "Touché-2020",
    "FEVERHardNegatives": "FEVER", "CQADupstackGamingRetrieval": "CQADupStack Gaming",
    "CQADupstackUnixRetrieval": "CQADupStack Unix",
    "BrightBiologyRetrieval": "Biology", "BrightEarthScienceRetrieval": "Earth Science",
    "BrightEconomicsRetrieval": "Economics", "BrightPsychologyRetrieval": "Psychology",
    "BrightRoboticsRetrieval": "Robotics", "BrightStackoverflowRetrieval": "Stack Overflow",
    "BrightSustainableLivingRetrieval": "Sustainable Living", "BrightPonyRetrieval": "Pony",
    "BrightLeetcodeRetrieval": "LeetCode", "BrightAopsRetrieval": "AoPS",
    "BrightTheoremQAQuestionsRetrieval": "TheoremQA Questions", "BrightTheoremQATheoremsRetrieval": "TheoremQA Theorems",
    "SciFact": "SciFact", "NFCorpus": "NFCorpus", "FinQARetrieval": "FinQA", "AppsRetrieval": "APPS",
    "AILACasedocs": "AILA Casedocs", "CUREv1": "CURE",
}


# Illustrated figures (Figure 1 and the tile figures) are HTML printed to PDF and PNG with headless Chromium.
NAVY, CREAM, PANEL = "#123a7a", "#fbf8f1", "#e3ebf7"
GRAIN = ("url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='160' height='160'>"
         "<filter id='n'><feTurbulence type='fractalNoise' baseFrequency='.85' numOctaves='2' stitchTiles='stitch'/>"
         "<feColorMatrix values='0 0 0 0 1  0 0 0 0 1  0 0 0 0 1  0 0 0 .07 0'/></filter>"
         "<rect width='100%' height='100%' filter='url(%23n)'/></svg>\")")


def line_icon(name, size, color, width=2):
    """A Lucide icon (figures/icons/line) as inline SVG."""
    svg = (FIGURES / "icons" / "line" / f"{name}.svg").read_text().split("-->", 1)[-1]
    svg = svg.replace("currentColor", color).replace('stroke-width="2"', f'stroke-width="{width}"')
    return svg.replace('width="24"', f'width="{size}"').replace('height="24"', f'height="{size}"')


def render(html, name, width, height):
    from playwright.sync_api import sync_playwright
    (FIGURES / f"{name}.html").write_text(html)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=2)
        page.set_content(html, wait_until="networkidle")
        page.pdf(path=str(FIGURES / f"{name}.pdf"), width=f"{width}px", height=f"{height}px", print_background=True,
                 page_ranges="1")
        page.screenshot(path=str(FIGURES / f"{name}.png"))
        browser.close()
    print("wrote", FIGURES / f"{name}.pdf")
