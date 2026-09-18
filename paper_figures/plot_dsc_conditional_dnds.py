"""dSc-conditional dN/dS theory: fit ``<dSc|dS>`` and re-derive the Fig-2B curve.

Produces two figures in ``config.fig_path``:

1. ``dsc_vs_dS_conditional_fit.pdf`` -- dSc vs full-genome dS with the binned
   ``<dSc|dS>`` and the non-parametric smoothing-spline fit of ``G(dS)``.

2. ``dNdS_dsc_conditional.pdf`` -- a modified version of ``dNdS.pdf`` (the Fig 2
   full/recombined/clonal three-panel), where the full-genome panel's theory
   curve is REPLACED by the new spline-``G(dS)`` conditional-expectation curve
   (old Hill/f_r curve kept dashed for comparison).

Reusable fit/theory functions live in ``dsc_conditional_theory.py`` (staging for
a later merge into ``dnds_dynamics.figures``).
"""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from dnds_dynamics import config
from dnds_dynamics.figures.dynamics import (computed_poisson_thinning,
                                            load_dNdS_data)
from dnds_dynamics.figures.theory import (logistic_accumulation,
                                          compute_dNdS_rec_model)
import dsc_conditional_theory as dct

np.random.seed(0)
FIG_DIR = config.fig_path
FIG_DIR.mkdir(parents=True, exist_ok=True)

# clonal-divergence quantity used for <dSc|dS>
DSC_COL = "dSc_region"      # d_S^{(c)} as in eq:...-average ("clonal_contrib" = exact)

# <dSc|dS> bin estimator: "mean" (theory-correct conditional expectation; default)
# or "median" (robustness check, DSC_ESTIMATOR=median). Median runs write
# *_median figures and overlay the old f_r(dS_c) Hill curve for comparison.
ESTIMATOR = os.environ.get("DSC_ESTIMATOR", "mean")
assert ESTIMATOR in ("mean", "median")
SUFFIX = "" if ESTIMATOR == "mean" else f"_{ESTIMATOR}"
# opt-in old-f_r(dS_c) Hill overlay for comparison (DSC_COMPARE_OLD=1); off by default
SHOW_OLD_HILL = os.environ.get("DSC_COMPARE_OLD", "0") == "1"
EST_TAG = "" if ESTIMATOR == "mean" else f" ({ESTIMATOR})"   # annotate only non-default
BIN_TAG = "" if ESTIMATOR == "mean" else f"{ESTIMATOR} "

# --------------------------------------------------------------------------- #
# Data + fit
# --------------------------------------------------------------------------- #
complete_df = load_dNdS_data()
pairs = dct.load_pair_table(complete_df)
rec = dct.estimate_recombined_ratio()
ratio = rec["ratio"]                         # dNbar/dSbar measured from typical pairs

# non-parametric spline fit of <dSc|dS> (assumption-free; the chosen form)
fit_sp = dct.fit_spline_accumulation(pairs["dS"].values, pairs[DSC_COL].values,
                                     smooth=0.6, estimator=ESTIMATOR)
print(f"estimator = {ESTIMATOR}")
print(f"recombined ratio  dNbar/dSbar = {ratio:.4f}  "
      f"(dSbar={rec['dSbar']:.3e}, dNbar={rec['dNbar']:.3e}, n_species={rec['n_species']})")
print(f"<dSc|dS> spline fit on {DSC_COL}: "
      f"support={tuple(f'{s:.1e}' for s in fit_sp['support'])}  R2={fit_sp['r2']:.4f}  "
      f"(diagonal G=dS forced below dS={fit_sp['diag_below']:.2e})")

# conditional-expectation dN/dS theory curve from the spline G(dS)
dS_grid = np.logspace(-6, np.log10(2e-2), 400)
new_curve_sp = dct.conditional_dNdS(dS_grid, fit_sp, ratio)

# old f_r(dS_c) Hill recombination-theory curve (for the median-vs-old comparison)
if SHOW_OLD_HILL:
    dsc_old = np.logspace(-6, -3, 200)
    fr_old = logistic_accumulation(dsc_old, 10 ** (-3.8), k=10)
    dS_old, dNdS_old = compute_dNdS_rec_model(dsc_old, fr_old, 3e-2, 1.0, 1e-1)
    # for figure 1: also show the mean-based binned <dSc|dS> as reference
    mean_x, mean_y, _ = dct.bin_conditional(pairs["dS"].values, pairs[DSC_COL].values,
                                            estimator="mean")

# Poisson-thinned genome-wide dN/dS scatter (shared by both figures)
dS1, dS2 = computed_poisson_thinning(complete_df["core_diff_4D"], complete_df["core_len_4D"])
dS1 = np.asarray(dS1, float); dS2 = np.asarray(dS2, float)
dN = np.asarray(complete_df["core_diff_1D"] / complete_df["core_len_1D"].astype(float), float)
zero = (dS2 == 0)


# --------------------------------------------------------------------------- #
# Figure 1: dSc vs dS with the spline fit
# --------------------------------------------------------------------------- #
mpl.rcParams["font.size"] = 8
fig, axL = plt.subplots(1, 1, figsize=(3.7, 3.1))

dS_all = pairs["dS"].values
dSc_all = pairs[DSC_COL].values
m = np.isfinite(dS_all) & np.isfinite(dSc_all) & (dS_all > 0) & (dSc_all > 0)
axL.scatter(dS_all[m], dSc_all[m], s=2, alpha=0.08, color="tab:grey", rasterized=True)
diag = np.array([1e-6, 3e-2])
axL.plot(diag, diag, ls=":", lw=0.8, color="k", label=r"$dS_c = dS$ (clonal limit)")
if SHOW_OLD_HILL:
    axL.plot(mean_x, mean_y, "o", ms=3, mfc="none", mec="0.6", mew=0.8,
             label=r"binned mean (ref)")
axL.plot(fit_sp["x"], fit_sp["y"], "o", ms=4, color="tab:red",
         label=rf"binned {BIN_TAG}$\langle dS_c|dS\rangle$")
sp_lo, sp_hi = fit_sp["support"]
in_sup = (dS_grid >= sp_lo) & (dS_grid <= sp_hi)
axL.plot(dS_grid[in_sup], fit_sp["func"](dS_grid[in_sup]), color="tab:blue", lw=1.8,
         label=rf"spline fit ($R^2$={fit_sp['r2']:.2f})")
axL.set_xscale("log"); axL.set_yscale("log")
axL.set_xlim([2e-6, 2e-2]); axL.set_ylim([3e-6, 3e-3])
axL.set_xlabel(r"$dS$ (Full genome)")
axL.set_ylabel(r"$dS_c$ (Clonal region)")
axL.legend(fontsize=6.5, frameon=False, loc="lower right")
axL.set_title(r"Clonal divergence accumulation $\langle dS_c\,|\,dS\rangle$",
              fontsize=8)

fig.savefig(FIG_DIR / f"dsc_vs_dS_conditional_fit{SUFFIX}.pdf", bbox_inches="tight", dpi=300)
print("wrote", FIG_DIR / f"dsc_vs_dS_conditional_fit{SUFFIX}.pdf")


# --------------------------------------------------------------------------- #
# Figure 2: modified dNdS.pdf (3-panel), new curve on the full-genome panel
# --------------------------------------------------------------------------- #
mpl.rcParams["font.size"] = 8
fig2, axes = plt.subplots(1, 3, figsize=(7.5, 1.7), dpi=300)
plt.subplots_adjust(wspace=0.15)

# panel 0: full (core) genome
axes[0].scatter(dS1[~zero], dN[~zero] / dS2[~zero], s=1, alpha=0.2,
                color="tab:grey", rasterized=True)
if SHOW_OLD_HILL:
    axes[0].plot(dS_old, dNdS_old, linestyle="--", color="tab:orange", linewidth=0.8,
                 label=r"old $f_r(dS_c)$ theory")
axes[0].plot(dS_grid, new_curve_sp, linestyle="-", color="k", linewidth=0.5,
             label=f"recombination theory ({ESTIMATOR})" if SHOW_OLD_HILL else "recombination theory")

# panel 1: recombined regions
naive_recomb_dS = np.asarray(complete_df["recomb_diff_4D"] / complete_df["recomb_len_4D"].astype(float), float)
recomb_dN = np.asarray(complete_df["recomb_diff_1D"] / complete_df["recomb_len_1D"].astype(float), float)
zr = naive_recomb_dS == 0
axes[1].scatter(dS1[~zr], recomb_dN[~zr] / naive_recomb_dS[~zr], s=1, alpha=0.3,
                color="#FF968D", rasterized=True)

# panel 2: clonal regions
naive_clonal_dS = np.asarray(complete_df["clonal_diff_4D"] / complete_df["clonal_len_4D"].astype(float), float)
clonal_dN = np.asarray(complete_df["clonal_diff_1D"] / complete_df["clonal_len_1D"].astype(float), float)
zc = naive_clonal_dS == 0
axes[2].scatter(dS1[~zc], clonal_dN[~zc] / naive_clonal_dS[~zc], s=1, alpha=0.3,
                color="#AECDE1", rasterized=True)

for ax in axes:
    ax.set_ylim([1e-2, 1e1]); ax.set_xlim([2e-6, 2e-2])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("$dS$ (Full genome)")
    ax.axhline(1, linewidth=0.5, linestyle="--", color="grey")
axes[0].set_ylabel("$dN/dS$")
axes[0].set_title("Full (core) genome")
axes[1].set_title("Recombined regions")
axes[2].set_title("Clonal regions")
axes[1].set_yticklabels([]); axes[2].set_yticklabels([])
axes[0].legend(fontsize=6 if SHOW_OLD_HILL else 7, frameon=False)

fig2.savefig(FIG_DIR / f"dNdS_dsc_conditional{SUFFIX}.pdf", bbox_inches="tight")
print("wrote", FIG_DIR / f"dNdS_dsc_conditional{SUFFIX}.pdf")
