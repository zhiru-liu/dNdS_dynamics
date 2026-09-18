"""SI figure: model-independent bounds on the cumulative DFE (main.tex eq:dfe-bound
and its large-dS analogue).

For an arbitrary DFE rho(s), the clonal purifying-accumulation identity

    dN/dS(dS) = int_0^inf ds rho(s) * (1 - e^{-s dS/2mu}) / (s dS / 2mu)

implies, for any threshold s* and any dS (main.tex, "General constraints on the
DFE from the asymptotic behavior of dN/dS"):

    UPPER (eq:dfe-bound):
        int_{s*}^inf rho ds  <=  (1 - dNdS) / (1 - F(s*,dS))
    LOWER (large-dS analogue):
        int_{s*}^inf rho ds  >=  1 - dNdS / F(s*,dS)

    with   F(s*,dS) = (1 - e^{-s* dS/2mu}) / (s* dS / 2mu)   and  sigma := s*/mu.

Because each inequality holds for *every* dS, the tightest bounds come from
scanning dS and taking, at each sigma,

    upper envelope  =  min_dS  (1 - dNdS(dS)) / (1 - F)
    lower envelope  =  max_dS  ( 1 - dNdS(dS) / F )

The true cumulative DFE int_{s*}^inf rho ds must lie between the two envelopes;
we shade that allowed region.

**dNdS(dS) is the empirical clonal dN/dS "mean curve"** -- the binned
bootstrap-median clonal dN/dS of ``clonal_dNdS_purifying_fit.pdf``
(``clonal_dNdS_dynamics.py``), extended on its right-hand side by the unrelated-
pair (typical) species means, exactly the two data series drawn in that figure.
This is NOT an analytical theory curve: the bounds are meant to be
DFE-model-independent, so we plug in the measured clonal dN/dS values directly.
The binned values are regularized with a weighted monotone (isotonic-decreasing)
fit -- the purifying curve is monotone in dS by construction, and this removes
sampling noise in the sparse low-dS bins that would otherwise make the two
envelopes cross. The raw binned means are overlaid (left panel) so the mean curve
remains visible.

dS is scanned over the observed pairwise-divergence support (closest close pairs
~1e-5 out to the recombined equilibrium ~3e-2). The small-dS end sets the
strong-mutation (large s*) upper bound -- exactly the manuscript's argument that
if many mutations were purged almost immediately, dN/dS would already have
dropped below one at these short times.

Output (config.fig_path): ``dfe_cumulative_bounds.pdf``.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy.random import default_rng

from dnds_dynamics import config
from dnds_dynamics.figures import dynamics as dynamics_utils

FIG_DIR = config.fig_path
FIG_DIR.mkdir(parents=True, exist_ok=True)

# thinning / binning / bootstrap settings -- identical to clonal_dNdS_dynamics.py
# so the reproduced mean curve matches clonal_dNdS_purifying_fit.pdf.
P_THIN = 0.5
N_BINS = 15
B = 400
EPS = 1e-12

# --------------------------------------------------------------------------- #
# Verbatim binning helpers from clonal_dNdS_dynamics.py
# --------------------------------------------------------------------------- #
def safe_div(num, den):
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    out = np.full_like(den, np.nan, dtype=float)
    np.divide(num, den, out=out, where=(den > 0))
    return out


def safe_div_series(num, den):
    res = safe_div(np.asarray(num), np.asarray(den))
    return pd.Series(res, index=pd.Index(num.index if hasattr(num, "index") else range(len(res))))


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
    return np.geomspace(dS_pos.min(), dS_pos.max(), n_bins + 1)


def aggregate_bins(df_thin, bin_edges):
    df = df_thin[(df_thin["dS_A"] > 0)].copy()
    df["bin"] = pd.cut(df["dS_A"], bin_edges, labels=False, include_lowest=True)
    gb = df.groupby("bin", dropna=True)
    agg = gb.agg(KN=("kN", "sum"), KS_A=("kS_A", "sum"), KS_B=("kS_B", "sum"),
                 LN=("LN", "sum"), LS_A=("LS_A", "sum"), LS_B=("LS_B", "sum"),
                 n_pairs=("kN", "size")).reset_index()
    agg["dS_x"] = safe_div_series(agg["KS_A"], agg["LS_A"])
    agg["R_hat"] = safe_div_series(agg["KN"], agg["LN"]) / safe_div_series(agg["KS_B"], agg["LS_B"])
    x = agg["dS_x"].to_numpy(float); y = agg["R_hat"].to_numpy(float)
    return agg.loc[(x > 0) & (y > 0) & np.isfinite(y)].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 1. Reproduce the clonal dN/dS "mean curve" (blue Mean+-95%CI line) + the
#    right-hand-side unrelated-pair species means (green X), as in
#    clonal_dNdS_purifying_fit.pdf.  Same rng order -> same curve.
# --------------------------------------------------------------------------- #
full_dnds_df = dynamics_utils.load_dNdS_data()
counts_df = full_dnds_df[["clonal_diff_1D", "clonal_len_1D",
                          "clonal_diff_4D", "clonal_len_4D"]].copy()
counts_df.columns = ["kN", "LN", "kS", "LS"]

rng = default_rng(42)
thin0 = thin_counts(counts_df, P_THIN, rng)
bin_edges = make_bins(thin0["dS_A"], N_BINS)
agg0 = aggregate_bins(thin0, bin_edges).astype(float)

logR_mat = np.full((B, N_BINS), np.nan)
for b in range(B):
    idx = rng.integers(0, len(counts_df), len(counts_df))
    boot = counts_df.iloc[idx].reset_index(drop=True)
    agg_b = aggregate_bins(thin_counts(boot, P_THIN, rng), bin_edges).astype(float)
    if not agg_b.empty:
        logR_mat[b, agg_b["bin"].astype(int).values] = np.log(agg_b["R_hat"].values)

bins_present = agg0["bin"].astype(int).values
clonal_dS = agg0["dS_x"].to_numpy(float)
clonal_r = np.exp(np.nanpercentile(logR_mat, 50.0, axis=0))[bins_present]   # MEAN CURVE
clonal_r_lo = np.exp(np.nanpercentile(logR_mat, 2.5, axis=0))[bins_present]
clonal_r_hi = np.exp(np.nanpercentile(logR_mat, 97.5, axis=0))[bins_present]
clonal_w = 1.0 / (1.0 / agg0["KN"].to_numpy(float) + 1.0 / agg0["KS_B"].to_numpy(float))

# right-hand side: unrelated-pair (typical) dN/dS, one point per species
typ = dynamics_utils.load_typical_pair_dNdS_data().reset_index(drop=False)
typ = typ.rename(columns={"core_diff_1D": "kN", "core_len_1D": "LN",
                          "core_diff_4D": "kS", "core_len_4D": "LS"})
recs = []
for sp, d in typ.groupby("species_name", sort=False):
    th = thin_counts(d[["kN", "LN", "kS", "LS"]].copy(), P_THIN, rng)
    agg_sp = aggregate_bins(th, make_bins(th["dS_A"], 1))
    if not agg_sp.empty:
        row = agg_sp.iloc[0]
        recs.append((float(row["dS_x"]), float(row["R_hat"]),
                     1.0 / (1.0 / float(row["KN"]) + 1.0 / float(row["KS_B"]))))
typ_dS, typ_r, typ_w = (np.array(t) for t in zip(*sorted(recs)))

# --------------------------------------------------------------------------- #
# 2. Build r(dS): weighted monotone (isotonic-decreasing) fit of the mean-curve
#    values, over clonal bins + typical species means.
# --------------------------------------------------------------------------- #
def isotonic_decreasing(y, w):
    """Weighted pool-adjacent-violators, non-increasing in input order."""
    val, wt, cnt = [], [], []
    for yi, wi in zip(np.asarray(y, float), np.asarray(w, float)):
        val.append(yi); wt.append(wi); cnt.append(1)
        while len(val) > 1 and val[-2] < val[-1] - 1e-15:
            v2, w2, c2 = val.pop(), wt.pop(), cnt.pop()
            v1, w1, c1 = val.pop(), wt.pop(), cnt.pop()
            val.append((v1 * w1 + v2 * w2) / (w1 + w2)); wt.append(w1 + w2); cnt.append(c1 + c2)
    out = []
    for v, c in zip(val, cnt):
        out += [v] * c
    return np.array(out)


all_dS = np.concatenate([clonal_dS, typ_dS])
all_r = np.concatenate([clonal_r, typ_r])
all_w = np.concatenate([clonal_w, typ_w])
o = np.argsort(all_dS)
all_dS, all_r, all_w = all_dS[o], all_r[o], all_w[o]
r_iso = np.exp(isotonic_decreasing(np.log(all_r), all_w))   # monotone mean curve


def dNdS_of_dS(dS):
    """Empirical clonal dN/dS at divergence dS (log-interp of the monotone mean
    curve; flat outside the observed support)."""
    dS = np.asarray(dS, float)
    return np.interp(np.log10(dS), np.log10(all_dS), r_iso, left=r_iso[0], right=r_iso[-1])


# --------------------------------------------------------------------------- #
# 3. Bound math + envelope over the observed dS support
# --------------------------------------------------------------------------- #
def F_factor(sigma, dS):
    """F(s*,dS) = (1-e^{-u})/u, u = sigma*dS/2, sigma = s*/mu. Stable at u->0."""
    u = np.asarray(sigma, float) * np.asarray(dS, float) / 2.0
    us = np.where(u == 0, 1.0, u)
    return np.where(u < 1e-8, 1.0 - u / 2.0 + u ** 2 / 6.0,
                    (1.0 - np.exp(-np.clip(u, None, 700.0))) / us)


DS_SCAN_MIN = float(all_dS.min())    # closest close pairs (~5e-6)
DS_SCAN_MAX = float(all_dS.max())    # unrelated-pair equilibrium (~6e-2)
# no data between the last clonal bin and the first unrelated pair -> skip that
# gap entirely (do NOT bound where dN/dS is unmeasured / only interpolated).
GAP_LO = float(clonal_dS.max())      # last clonal mean (~5e-4)
GAP_HI = float(typ_dS.min())         # first unrelated pair (~7e-3)
dS_scan = np.concatenate([
    np.logspace(np.log10(DS_SCAN_MIN), np.log10(GAP_LO), 500),   # clonal range
    np.logspace(np.log10(GAP_HI), np.log10(DS_SCAN_MAX), 400),   # unrelated range
])
r_scan = dNdS_of_dS(dS_scan)
purify = r_scan < 1.0 - 1e-9         # only sub-1 bins carry purifying information

sigma_grid = np.logspace(1, 6, 400)
upper = np.empty_like(sigma_grid)
lower = np.empty_like(sigma_grid)
for i, sig in enumerate(sigma_grid):
    F = F_factor(sig, dS_scan)
    u_vals = np.where(purify, (1.0 - r_scan) / (1.0 - F), np.inf)
    upper[i] = min(np.min(u_vals), 1.0)
    lower[i] = max(np.max(1.0 - r_scan / F), 0.0)

print(f"dS scan support: [{DS_SCAN_MIN:.2e}, {GAP_LO:.2e}] U "
      f"[{GAP_HI:.2e}, {DS_SCAN_MAX:.2e}]  (gap {GAP_LO:.2e}-{GAP_HI:.2e} dropped)")
for s in (1e2, 1e3, 1e4, 1e5):
    i = int(np.argmin(np.abs(sigma_grid - s)))
    print(f"  s*/mu={s:.0e}:  tail in [{lower[i]:.3f}, {upper[i]:.3f}]")

# --------------------------------------------------------------------------- #
# 4. Figure
# --------------------------------------------------------------------------- #
mpl.rcParams["font.size"] = 8
UP_C, LO_C, BAND_C = "#0072B2", "#D55E00", "#999999"
fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.4, 3.2))
plt.subplots_adjust(wspace=0.33)

# --- Panel A: the clonal dN/dS mean curve feeding the bounds -----------------
axA.axvspan(1e-3, 1, color=BAND_C, alpha=0.10, lw=0, zorder=0)
axA.errorbar(clonal_dS, clonal_r,
             yerr=[np.clip(clonal_r - clonal_r_lo, 0, None),
                   np.clip(clonal_r_hi - clonal_r, 0, None)],
             fmt="o", ms=4, lw=0, elinewidth=0.9, capsize=2, color="#0072B2",
             ecolor="#0072B2", zorder=4, label="clonal mean curve (binned)")
axA.plot(typ_dS, typ_r, "X", ms=5, color="#009E73", mec="white", mew=0.4,
         zorder=4, label="unrelated pairs (species)")
# monotone curve drawn only where there is data (gap left blank)
seg_lo = np.logspace(np.log10(DS_SCAN_MIN), np.log10(GAP_LO), 250)
seg_hi = np.logspace(np.log10(GAP_HI), np.log10(DS_SCAN_MAX), 250)
axA.plot(seg_lo, dNdS_of_dS(seg_lo), "-", color="k", lw=1.4, zorder=5,
         label=r"$dN/dS(dS)$ used (monotone)")
axA.plot(seg_hi, dNdS_of_dS(seg_hi), "-", color="k", lw=1.4, zorder=5)
axA.axhline(1, color="grey", ls="--", lw=0.6, zorder=1)
axA.set_xscale("log"); axA.set_yscale("log")
axA.set_xlim([2e-6, 1e-1]); axA.set_ylim([6e-2, 2e0])
axA.set_xlabel(r"$dS$ (Clonal region)"); axA.set_ylabel(r"$dN/dS$ (Clonal region)")
axA.set_title("Input: clonal $dN/dS$ mean curve", fontsize=8.5)
axA.legend(fontsize=6, frameon=False, loc="lower left")

# --- Panel B: bounds on the cumulative DFE ----------------------------------
axB.fill_between(sigma_grid, lower, upper, color=BAND_C, alpha=0.22, lw=0,
                 zorder=1, label="allowed region")
axB.plot(sigma_grid, upper, "-", color=UP_C, lw=1.8, zorder=4,
         label="upper (Eq. 19)")
axB.plot(sigma_grid, lower, "-", color=LO_C, lw=1.8, zorder=4,
         label="lower (Eq. 20)")
axB.set_xscale("log")
axB.set_xlim([1e1, 1e6]); axB.set_ylim([0, 1.02])
axB.set_xlabel(r"$s^*/\mu$")
axB.set_ylabel(r"$\int_{s^*}^{\infty}\rho(s)\,ds$   (fraction with $s/\mu>s^*/\mu$)")
axB.set_title("Bounds on the cumulative DFE", fontsize=8.5)
axB.legend(fontsize=7.5, frameon=False, loc="lower left")

fig.savefig(FIG_DIR / "dfe_cumulative_bounds.pdf", bbox_inches="tight", dpi=300)
print("wrote", FIG_DIR / "dfe_cumulative_bounds.pdf")
