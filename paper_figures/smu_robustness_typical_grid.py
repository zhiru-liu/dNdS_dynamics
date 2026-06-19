"""Reviewer robustness check: how does the fitted clonal s/mu depend on the
assumed *typical-pair* dN/dS and dS?

Context
-------
The main-text two-class purifying-selection fit (Fig. 3,
``clonal_dNdS_purifying_fit.pdf``; produced by
``paper_figures/clonal_dNdS_dynamics.py``) fits

    dN/dS(dS) = alpha0 + (1 - alpha0) * (1 - e^{-s/mu * dS / 2}) / (s/mu * dS / 2)
              = (1 - fd)  + fd        * F(s/mu * dS / 2)            with fd = 1 - alpha0

to the *binned clonal* points by weighted log-least-squares over s/mu, with the
neutral fraction held FIXED at alpha0 = (long-term dN/dS of unrelated/"typical"
pairs).  In the published procedure the typical pairs therefore enter the s/mu
fit ONLY through alpha0 = typical dN/dS -- the typical *dS* (the green crosses'
x-position in the figure) is a plotted reference and never touches the fit.

A reviewer asks how robust the inferred s/mu is to the typical-pair dN/dS values.
This script takes the clonal points exactly as in the paper, then scans a grid of
artificial (typical dN/dS, typical dS) values, setting fd = 1 - typical_dNdS in
each cell, and reports the refit s/mu.  Two panels:

  Panel A ("alpha0 only", faithful to the paper):
      fit s/mu to the clonal bins alone.  s/mu depends only on typical dN/dS
      (the typical-dS axis is flat by construction) -- this *is* the robustness
      statement.

  Panel B ("anchor"):
      additionally include the artificial typical pair as one extra anchor data
      point at (dS = typical_dS, R = typical_dNdS) in the same WLS objective,
      weighted like a single representative clonal bin (median bin weight).  Now
      a small typical_dS forces the curve to decay earlier -> larger s/mu, so the
      typical-dS axis becomes informative.

Both panels are colored by log10(s/mu).  The real operating point (measured
typical dN/dS & dS, pooled across species) is overlaid as a marker.

Output: figures/smu_robustness_typical_grid.{pdf,png}
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import default_rng
from scipy.optimize import minimize_scalar

from dnds_dynamics.figures import dynamics as dynamics_utils
from dnds_dynamics.figures.theory import dNdS_purify_curve
from dnds_dynamics import config

# --------------------------------------------------------------------------
# Settings (mirror paper_figures/clonal_dNdS_dynamics.py)
# --------------------------------------------------------------------------
P_THIN    = 0.5
N_BINS    = 15
MIN_PAIRS = 10
SBMU_BOUNDS = (1e-3, 1e7)
PHI_N, PHI_S = 1.0, 1.0
EPS = 1e-12
SEED = 123  # cell-9 rng seed in the paper script

# grid of artificial typical-pair values
N_GRID = 60
TYP_DNDS_GRID = np.linspace(0.02, 0.30, N_GRID)        # equilibrium dN/dS  -> alpha0
TYP_DS_GRID   = np.geomspace(1e-3, 1e-1, N_GRID)       # typical (recombined) dS


# --------------------------------------------------------------------------
# Helpers (copied verbatim in spirit from the paper script)
# --------------------------------------------------------------------------
def safe_div(num, den):
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    out = np.full_like(den, np.nan, dtype=float)
    np.divide(num, den, out=out, where=(den > 0))
    return out


def safe_div_series(num, den):
    res = safe_div(np.asarray(num), np.asarray(den))
    idx = num.index if hasattr(num, "index") else range(len(res))
    return pd.Series(res, index=pd.Index(idx))


def thin_counts(df, p, rng):
    out = df.copy()
    out["kS_A"] = rng.binomial(out["kS"].astype(int), p)
    out["kS_B"] = out["kS"] - out["kS_A"]
    out["LS_A"] = p * out["LS"]
    out["LS_B"] = (1 - p) * out["LS"]
    out["dS_A"] = out["kS_A"] / np.clip(out["LS_A"], EPS, None)
    out["dS_B"] = out["kS_B"] / np.clip(out["LS_B"], EPS, None)
    return out


def make_bins(dS_A, n_bins):
    dS_pos = dS_A[dS_A > 0]
    lo, hi = dS_pos.min(), dS_pos.max()
    return np.geomspace(lo, hi, n_bins + 1)


def aggregate_bins(df_thin, bin_edges):
    df = df_thin[(df_thin["dS_A"] > 0)].copy()
    df["bin"] = pd.cut(df["dS_A"], bin_edges, labels=False, include_lowest=True)
    gb = df.groupby("bin", dropna=True)
    agg = gb.agg(
        KN=("kN", "sum"), KS_A=("kS_A", "sum"), KS_B=("kS_B", "sum"),
        LN=("LN", "sum"), LS_A=("LS_A", "sum"), LS_B=("LS_B", "sum"),
        n_pairs=("kN", "size"),
    ).reset_index()
    agg["dS_x"] = safe_div_series(agg["KS_A"], agg["LS_A"])
    numer = safe_div_series(agg["KN"], agg["LN"])
    denom = safe_div_series(agg["KS_B"], agg["LS_B"])
    agg["R_hat"] = numer / denom
    x = agg["dS_x"].to_numpy(dtype=float)
    y = agg["R_hat"].to_numpy(dtype=float)
    m = (x > 0) & (y > 0) & np.isfinite(y)
    return agg.loc[m].reset_index(drop=True)


def _select(agg):
    """Filter rows usable for the WLS fit; return x, y, weights."""
    m = (
        (agg["n_pairs"] >= MIN_PAIRS) &
        (agg["KN"] > 0) & (agg["KS_B"] > 0) &
        (agg["dS_x"] > 0) & (agg["R_hat"] > 0) &
        np.isfinite(agg["R_hat"].to_numpy(float))
    )
    x = agg.loc[m, "dS_x"].to_numpy(float)
    y = agg.loc[m, "R_hat"].to_numpy(float)
    KN = agg.loc[m, "KN"].to_numpy(float)
    KS = agg.loc[m, "KS_B"].to_numpy(float)
    w = 1.0 / (PHI_N / np.maximum(KN, 1.0) + PHI_S / np.maximum(KS, 1.0))
    return x, y, w


def fit_sbymu(x, y, w, fd):
    """Weighted log-LS fit of s/mu given fixed deleterious fraction fd."""
    if len(x) < 3:
        return np.nan

    def objective(sbymu):
        pred = dNdS_purify_curve(x, fd, sbymu)
        resid = np.log(y) - np.log(pred)
        return np.average(resid ** 2, weights=w)

    res = minimize_scalar(objective, bounds=SBMU_BOUNDS, method="bounded")
    return res.x


# --------------------------------------------------------------------------
# Build the (fixed) clonal bins, exactly as the paper does
# --------------------------------------------------------------------------
full_dnds_df = dynamics_utils.load_dNdS_data()
counts_df = full_dnds_df[["clonal_diff_1D", "clonal_len_1D",
                          "clonal_diff_4D", "clonal_len_4D"]].copy()
counts_df.columns = ["kN", "LN", "kS", "LS"]

rng = default_rng(SEED)
thin0 = thin_counts(counts_df, P_THIN, rng)
bin_edges = make_bins(thin0["dS_A"], N_BINS)
agg0 = aggregate_bins(thin0, bin_edges).astype(float)

x_cl, y_cl, w_cl = _select(agg0)
w_anchor = np.median(w_cl)  # weight of a single representative clonal bin
print(f"Clonal bins used in fit: {len(x_cl)}")
print(f"Clonal dS range: {x_cl.min():.2e} - {x_cl.max():.2e}")
print(f"Median clonal-bin weight (= anchor weight): {w_anchor:.1f}")

# Sanity: reproduce the published point estimate (fd = 0.9 -> alpha0 = 0.1)
sb_pub = fit_sbymu(x_cl, y_cl, w_cl, fd=0.9)
print(f"Reproduced published point estimate (fd=0.9): s/mu = {sb_pub:.3g}")

# --------------------------------------------------------------------------
# Measured "real" operating point: pooled typical-pair dN/dS and dS
# --------------------------------------------------------------------------
typ = dynamics_utils.load_typical_pair_dNdS_data()
KN_t = typ["core_diff_1D"].sum(); LN_t = typ["core_len_1D"].sum()
KS_t = typ["core_diff_4D"].sum(); LS_t = typ["core_len_4D"].sum()
real_typ_dnds = (KN_t / LN_t) / (KS_t / LS_t)
real_typ_ds = KS_t / LS_t
print(f"Measured typical-pair dN/dS = {real_typ_dnds:.3f}, dS = {real_typ_ds:.3e}")

# Per-species spread of measured typical-pair (dS, dN/dS) -> box overlay
sp_summary = pd.read_csv(
    REPO_ROOT / "outputs" / "typical_dnds_across_species" / "species_summary.csv")
ds_box_lo, ds_box_hi = sp_summary["dS"].min(), sp_summary["dS"].max()
dnds_box_lo, dnds_box_hi = sp_summary["dNdS"].min(), sp_summary["dNdS"].max()
print(f"Across-species dS range:   {ds_box_lo:.3e} - {ds_box_hi:.3e}")
print(f"Across-species dN/dS range: {dnds_box_lo:.3f} - {dnds_box_hi:.3f} "
      f"(n={len(sp_summary)})")

# --------------------------------------------------------------------------
# Scan the grid
# --------------------------------------------------------------------------
smu_A = np.full((N_GRID, N_GRID), np.nan)  # alpha0-only (faithful)
smu_B = np.full((N_GRID, N_GRID), np.nan)  # + typical anchor point

for i, typ_dnds in enumerate(TYP_DNDS_GRID):
    fd = 1.0 - typ_dnds
    # Panel A: clonal bins only (independent of typ_ds)
    sb_A = fit_sbymu(x_cl, y_cl, w_cl, fd=fd)
    for j, typ_ds in enumerate(TYP_DS_GRID):
        smu_A[i, j] = sb_A
        # Panel B: append the artificial typical anchor point
        x_b = np.append(x_cl, typ_ds)
        y_b = np.append(y_cl, typ_dnds)
        w_b = np.append(w_cl, w_anchor)
        smu_B[i, j] = fit_sbymu(x_b, y_b, w_b, fd=fd)

# --------------------------------------------------------------------------
# Plot
# --------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
X, Y = np.meshgrid(TYP_DS_GRID, TYP_DNDS_GRID)  # x=typ_dS, y=typ_dNdS

# color scaled to the actual data range (variation is small -- that is the point)
vmin = np.nanmin(np.log10([smu_A, smu_B]))
vmax = np.nanmax(np.log10([smu_A, smu_B]))
clevels = np.round(np.arange(np.floor(vmin * 20) / 20,
                             vmax + 0.05, 0.05), 2)  # every ~12% in s/mu

from matplotlib.patches import Rectangle

for ax, Z, title in [
    (axes[0], smu_A, r"$\alpha_0=$ typical dN/dS only"),
    (axes[1], smu_B, r"+ typical pair as fit anchor"),
]:
    c = ax.pcolormesh(X, Y, np.log10(Z), shading="auto", cmap="viridis",
                      vmin=vmin, vmax=vmax)
    cs = ax.contour(X, Y, np.log10(Z), levels=clevels,
                    colors="white", linewidths=0.8)
    ax.clabel(cs, fmt=lambda v: f"{10**v/1e3:.1f}k", fontsize=6)
    # across-species range of measured typical-pair (dS, dN/dS)
    ax.add_patch(Rectangle(
        (ds_box_lo, dnds_box_lo), ds_box_hi - ds_box_lo, dnds_box_hi - dnds_box_lo,
        fill=False, edgecolor="red", linewidth=1.4, linestyle="--",
        zorder=4, label="across-species range"))
    ax.scatter([real_typ_ds], [real_typ_dnds], marker="*", s=180,
               color="red", edgecolors="white", linewidths=0.8, zorder=5,
               label="Fig 3 fit")
    ax.set_xscale("log")
    ax.set_xlabel(r"typical-pair $d_S$")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper right", fontsize=7, frameon=True)

axes[0].set_ylabel(r"typical-pair $d_N/d_S$  ($\alpha_0$)")
cbar = fig.colorbar(c, ax=axes, fraction=0.046, pad=0.02)
cbar.set_label(r"$\log_{10}(s/\mu)$ (two-class fit)")

out = config.fig_path / "smu_robustness_typical_grid"
fig.savefig(f"{out}.pdf", dpi=600, bbox_inches="tight")
fig.savefig(f"{out}.png", dpi=150, bbox_inches="tight")
print(f"Wrote {out}.pdf / .png")

# --------------------------------------------------------------------------
# Console summary of the robustness range
# --------------------------------------------------------------------------
print("\n=== s/mu over the grid ===")
print(f"Panel A (alpha0 only): {np.nanmin(smu_A):.2e} - {np.nanmax(smu_A):.2e}")
print(f"Panel B (+ anchor):    {np.nanmin(smu_B):.2e} - {np.nanmax(smu_B):.2e}")
# slice at the measured typical dN/dS
i_real = int(np.argmin(np.abs(TYP_DNDS_GRID - real_typ_dnds)))
print(f"\nAt typical dN/dS ~ {TYP_DNDS_GRID[i_real]:.3f} (closest grid row):")
print(f"  Panel A s/mu = {smu_A[i_real, 0]:.3g} (flat in typ_dS)")
print(f"  Panel B s/mu across typ_dS: {np.nanmin(smu_B[i_real]):.3g} - {np.nanmax(smu_B[i_real]):.3g}")
