"""Shared look for the paper's figures: large sans-serif text, thick marks, two colors, labels in the plot.

Blue circles are always the tasks' own queries; red triangles are always generated queries."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import ROOT

FIGURES = ROOT / "figures"
BLUE, RED, INK, GRAY = "#0077B6", "#E90909", "#111111", "#9a9a9a"
OWN = dict(color=BLUE, marker="o", label="Tasks' own queries")
GEN = dict(color=RED, marker="v", label="Generated queries")
FONTSIZE = 18

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
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
    "lines.linewidth": 5,
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
