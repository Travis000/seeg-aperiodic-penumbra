# -*- coding: utf-8 -*-
"""
fig1_hero.py — Figure 1 hero panel A: a vertical study-timeline (left) beside the
anatomical illustration (right), enriched with the paper's core finding (the
aperiodic-exponent gradient / inhibitory penumbra) drawn onto the real anatomy.

The illustration itself is the Gemini-generated raster (Fig1A_illustration_raw.png);
the timeline, exponent-coloured contacts, colour-bar and callouts are vector overlays.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe
from pathlib import Path

ILLU = Path(__file__).resolve().parents[3] / "output" / "2_Figures" / "Fig1A_illustration_raw.png"

C_PRE, C_RFTC, C_POST, C_OUT = "#5B7FA6", "#B5784E", "#5B7FA6", "#6E9B73"
EC, LC = "#555555", "#1A1A1A"
# anatomy landmark pixel coords in the raw illustration
LES = np.array([704, 893])     # RF-TC lesion / SOZ
ENT = np.array([1047, 714])    # SEEG skull entry
DISC = (360, 205)              # a scalp electrode


def draw_timeline(ax, letter="A"):
    """Vertical study-timeline (left column of hero A)."""
    ax.set_xlim(0, 3.0); ax.set_ylim(0, 10); ax.axis("off")
    if letter:
        ax.text(-0.02, 1.0, letter, transform=ax.transAxes, fontsize=15,
                fontweight="bold", va="top", ha="left")
    ax.add_patch(FancyArrowPatch((0.55, 9.2), (0.55, 0.7), arrowstyle="-|>",
                 mutation_scale=15, lw=2.2, color="#CFC9C2", shrinkA=0, shrinkB=0))
    mil = [(8.4, C_PRE, "Pre-operative", "baseline"),
           (6.0, C_RFTC, "SEEG-guided RF-TC", "month 0"),
           (3.6, C_POST, "Post-operative", "6–8 months"),
           (1.2, C_OUT, "Outcome (Engel)", "12–16 months")]
    for y, col, title, tp in mil:
        ax.scatter(0.55, y, s=300, color=col, edgecolors="white", linewidths=2.4, zorder=5)
        ax.text(0.86, y + 0.20, title, ha="left", va="bottom", fontsize=13,
                fontweight="bold", color=col)
        ax.text(0.86, y - 0.24, tp, ha="left", va="top", fontsize=10.5,
                fontstyle="italic", color="#8A8A8A")
    ax.text(0.86, 6.0 - 0.74, "1,618 SEEG contacts", ha="left", va="top",
            fontsize=9.5, color=C_RFTC)
    ax.plot([0.24, 0.16, 0.16, 0.24], [8.4, 8.4, 3.6, 3.6], color="#A8A8A8", lw=1.1)
    ax.text(0.02, (8.4 + 3.6) / 2, "Paired scalp EEG  (N = 16)", ha="center", va="center",
            rotation=90, fontsize=9.5, fontweight="bold", color="#5A5A5A")


def draw_illustration(ax):
    """Anatomical illustration + exponent-gradient overlay (right column of hero A)."""
    raw = mpimg.imread(ILLU); H, W = raw.shape[0], raw.shape[1]
    ax.imshow(raw, extent=[0, W, H, 0]); ax.axis("off")
    ax.set_xlim(-10, W + 520); ax.set_ylim(H + 40, -200)

    # exponent-coloured SEEG contacts along the electrode (SOZ red/low -> distal blue/high)
    cmap = plt.get_cmap("RdBu")
    for f in np.linspace(0.10, 0.92, 7):
        p = LES + f * (ENT - LES)
        ax.scatter(*p, s=80, color=cmap(f), edgecolors="white", linewidths=1.3, zorder=8)

    # mini colour-bar (inset)
    cax = ax.inset_axes([0.985, 0.30, 0.02, 0.34])
    cax.imshow(np.linspace(1, 0, 100).reshape(-1, 1), cmap="RdBu", aspect="auto")
    cax.set_xticks([]); cax.set_yticks([])
    cax.text(1.6, 1.0, "high\n(inhibition)", transform=cax.transAxes, ha="left", va="top",
             fontsize=8, color="#2166AC")
    cax.text(1.6, 0.0, "low\n(excitation)", transform=cax.transAxes, ha="left", va="bottom",
             fontsize=8, color="#B2182B")
    cax.text(-0.4, 0.5, "Aperiodic exponent", transform=cax.transAxes, ha="center", va="center",
             rotation=90, fontsize=8.6, fontweight="bold", color="#333")

    def lab(text, xy, xytext, ha, col=LC):
        ax.annotate(text, xy=xy, xytext=xytext, ha=ha, va="center", fontsize=11.5, color=col,
                    linespacing=1.2, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none"),
                    arrowprops=dict(arrowstyle="-", color=EC, lw=1.5, shrinkA=6, shrinkB=3))
        ax.scatter(*xy, s=20, color=EC, zorder=9,
                   path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])

    lab("Scalp EEG", DISC, (360, -135), "center")
    lab("SEEG depth electrodes\n(~2 mm burr holes)", tuple(ENT), (W + 12, 470), "left")
    lab("Inhibitory penumbra\nexponent ↑ with distance",
        tuple((LES + 0.55 * (ENT - LES)).astype(int)), (W + 12, 775), "left", col="#2166AC")
    lab("RF-TC lesion = SOZ\nlowest exponent", tuple(LES), (W + 12, 1075), "left", col="#B2182B")


def draw_mechanism(ax, letter="E"):
    """Low-key dotted-flow mechanism cascade (matches the timeline's visual language)."""
    ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis("off")
    if letter:
        ax.text(-0.005, 1.10, letter, transform=ax.transAxes, fontsize=15,
                fontweight="bold", va="top", ha="left")
    nodes = [(1.1, C_RFTC, "Low-exponent SOZ", "excitation ↑"),
             (3.9, "#6B6B6B", "RF-TC ablation", "remove SOZ"),
             (6.7, C_POST, "Inhibitory penumbra", "exponent ↑"),
             (9.0, C_OUT, "Seizure freedom", "Engel I–II")]
    for i in range(3):
        ax.add_patch(FancyArrowPatch((nodes[i][0] + 0.55, 1.5), (nodes[i + 1][0] - 0.55, 1.5),
                     arrowstyle="-|>", mutation_scale=14, lw=1.8, color="#C2C2C2", zorder=1))
    for x, col, title, sub in nodes:
        ax.scatter(x, 1.5, s=240, color=col, edgecolors="white", linewidths=2.2, zorder=5)
        ax.text(x, 2.22, title, ha="center", va="bottom", fontsize=11, fontweight="bold", color=col)
        ax.text(x, 1.16, sub, ha="center", va="top", fontsize=9, fontstyle="italic", color="#777")


if __name__ == "__main__":
    import matplotlib; matplotlib.use("Agg")
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "DejaVu Sans"], "pdf.fonttype": 42})
    fig = plt.figure(figsize=(14, 8.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.31, 1.0], wspace=0.0,
                          left=0.02, right=0.99, top=0.97, bottom=0.03)
    draw_timeline(fig.add_subplot(gs[0, 0]))
    draw_illustration(fig.add_subplot(gs[0, 1]))
    out = ILLU.parent / "Fig1A_hero_preview.png"
    fig.savefig(out, dpi=140, facecolor="white", bbox_inches="tight")
    print("saved", out)
