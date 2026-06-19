"""Composite reviewer figure: empirical cross-species typical dN/dS-vs-dS (panel a)
next to the s/mu robustness grids (panels b, c) in ONE shared (dS, dN/dS) frame.

Panel (a) is the measured per-species typical-pair (dS, dN/dS) cloud from
``typical_dnds_vs_ds_across_species.py`` (read from species_summary.csv). Panels
(b, c) are the two ``smu_robustness_typical_grid.py`` panels (s/mu refit while
sweeping the assumed typical-pair dN/dS=alpha0 and dS). Because all three share
the same x (typical dS, log) and y (typical dN/dS, linear) axes, the red dashed
"across-species range" box in (b, c) is literally the bounding box of the cloud in
(a): the reader sees the data, then how that region maps onto the inferred s/mu.

The s/mu grids are reused directly from ``smu_robustness_typical_grid`` (imported,
which recomputes them); panel (a) is redrawn from the summary CSV. No analysis is
duplicated here.

Output: figures/typical_dnds_smu_composite.{pdf,png}
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parents[1]
SUMMARY = REPO / "outputs" / "typical_dnds_across_species" / "species_summary.csv"
SMU_SCRIPT = REPO / "paper_figures" / "smu_robustness_typical_grid.py"
OUT = REPO / "figures" / "typical_dnds_smu_composite"
VULGATUS = "Bacteroides_vulgatus_57955"

# Shared axis frame (matches the smu grid axes exactly).
XLIM = (1e-3, 1e-1)      # typical-pair dS (log)
YLIM = (0.02, 0.30)      # typical-pair dN/dS (linear)


def _load_smu():
    """Import the smu robustness script (runs its compute) and return its module."""
    spec = importlib.util.spec_from_file_location("smu_robust", SMU_SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _panel_label(ax, letter: str) -> None:
    ax.text(-0.02, 1.04, f"({letter})", transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="bottom", ha="right")


def draw_scatter(ax, summary: pd.DataFrame, box: tuple, show_box: bool = True) -> None:
    """Panel (a): per-species typical (dS, dN/dS) cloud + 16-84 inter-pair spans.

    ``show_box`` draws the dashed across-species range rectangle (used in the
    composite to tie the eye to the s/mu panels); turn it off for a standalone
    panel where there is nothing to tie it to.
    """
    from scipy.stats import spearmanr, pearsonr
    generic = summary[summary["category"].isin(["typical", "within_clade"])].copy()
    cloud = generic[generic["species_name"] != VULGATUS]
    vw = summary[(summary["species_name"] == VULGATUS) & (summary["category"] == "within_clade")]
    vc = summary[(summary["species_name"] == VULGATUS) & (summary["category"] == "cross_clade")]

    def spans(g, ecolor, z):
        ax.hlines(g["dNdS"], g["dS_p16"], g["dS_p84"], color=ecolor, lw=0.8, alpha=0.9, zorder=z)
        ax.vlines(g["dS"], g["dNdS_p16"], g["dNdS_p84"], color=ecolor, lw=0.8, alpha=0.9, zorder=z)

    spans(cloud, "#9ecae1", 2)
    ax.scatter(cloud["dS"], cloud["dNdS"], s=26, color="#0072B2", linewidths=0,
               alpha=0.9, zorder=3, label="species (typical pairs)")
    if len(vc):
        spans(vc, "#fdae6b", 2)
        ax.scatter(vc["dS"], vc["dNdS"], s=70, marker="D", color="#D55E00",
                   linewidths=0, zorder=5, label="P. vulgatus x dorei (cross-clade)")
    if len(vw):
        spans(vw, "#fdae6b", 2)
        ax.scatter(vw["dS"], vw["dNdS"], s=70, marker="D", facecolor="none",
                   edgecolor="#D55E00", linewidths=1.6, zorder=6,
                   label="P. vulgatus within-clade")
    # same across-species box as panels (b,c), to tie the eye across panels
    if show_box:
        lo_ds, hi_ds, lo_dn, hi_dn = box
        ax.add_patch(Rectangle((lo_ds, lo_dn), hi_ds - lo_ds, hi_dn - lo_dn, fill=False,
                               edgecolor="red", linewidth=1.4, linestyle="--", zorder=4,
                               label="across-species range"))
    # correlation across the generic cloud (rank-based rho is scale-free, so it is
    # identical whether (a) is drawn linear or log; report linear Pearson too).
    xx = generic["dS"].to_numpy(float); yy = generic["dNdS"].to_numpy(float)
    ok = (xx > 0) & (yy > 0) & np.isfinite(xx) & np.isfinite(yy)
    rho, p_rho = spearmanr(xx[ok], yy[ok])
    r_log, _ = pearsonr(np.log10(xx[ok]), np.log10(yy[ok]))
    ax.text(0.03, 0.045,
            f"Spearman $\\rho$={rho:.2f}\n(p={p_rho:.1g})\nlog-log r={r_log:.2f}",
            transform=ax.transAxes, fontsize=7.5, va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6", alpha=0.9))
    ax.set_xscale("log")
    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
    ax.set_xlabel(r"typical-pair $d_S$")
    ax.set_ylabel(r"typical-pair $d_N/d_S$  ($\alpha_0$)")
    ax.set_title("measured per-species typical dN/dS", fontsize=10)
    ax.legend(loc="upper right", fontsize=6.5, frameon=True)


def draw_grid(ax, m, Z, title: str):
    """Panels (b,c): log10(s/mu) heatmap + contours + across-species box + Fig-3 star."""
    X, Y = np.meshgrid(m.TYP_DS_GRID, m.TYP_DNDS_GRID)
    vmin = np.nanmin(np.log10([m.smu_A, m.smu_B]))
    vmax = np.nanmax(np.log10([m.smu_A, m.smu_B]))
    clevels = np.round(np.arange(np.floor(vmin * 20) / 20, vmax + 0.05, 0.05), 2)
    c = ax.pcolormesh(X, Y, np.log10(Z), shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    cs = ax.contour(X, Y, np.log10(Z), levels=clevels, colors="white", linewidths=0.8)
    ax.clabel(cs, fmt=lambda v: f"{10 ** v / 1e3:.1f}k", fontsize=6)
    ax.add_patch(Rectangle(
        (m.ds_box_lo, m.dnds_box_lo), m.ds_box_hi - m.ds_box_lo, m.dnds_box_hi - m.dnds_box_lo,
        fill=False, edgecolor="red", linewidth=1.4, linestyle="--", zorder=4,
        label="across-species range"))
    ax.scatter([m.real_typ_ds], [m.real_typ_dnds], marker="*", s=180, color="red",
               edgecolors="white", linewidths=0.8, zorder=5, label="Fig 3 fit")
    ax.set_xscale("log")
    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
    ax.set_xlabel(r"typical-pair $d_S$")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper right", fontsize=6.5, frameon=True)
    return c


def main() -> None:
    summary = pd.read_csv(SUMMARY)
    m = _load_smu()
    box = (m.ds_box_lo, m.ds_box_hi, m.dnds_box_lo, m.dnds_box_hi)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7), sharey=True)
    draw_scatter(axes[0], summary, box)
    draw_grid(axes[1], m, m.smu_A, r"s/$\mu$ fit: $\alpha_0=$ typical dN/dS only")
    c = draw_grid(axes[2], m, m.smu_B, r"s/$\mu$ fit: + typical pair as anchor")
    for ax, letter in zip(axes, "abc"):
        _panel_label(ax, letter)

    cbar = fig.colorbar(c, ax=axes.tolist(), fraction=0.030, pad=0.015)
    cbar.set_label(r"$\log_{10}(s/\mu)$ (two-class fit)")
    fig.suptitle("Cross-species typical-pair dN/dS vs dS, and its effect on the inferred s/$\\mu$",
                 fontsize=12, y=1.02)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}.pdf / .png")


if __name__ == "__main__":
    main()
