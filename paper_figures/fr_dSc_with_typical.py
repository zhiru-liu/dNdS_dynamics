"""Recombination-dynamics fit + f_r-colored dN/dS equivalence (revision figure).

Produces ONE composite figure (recombination_dynamics_fit_with_fr.pdf):

  Top row  -- published `recombination_dynamics_fit.pdf` panels: f_r vs dS_c
              accumulation (log-log and log-linear), grey close-pair scatter with
              a single "Expected trend" curve. We use the x-residual Hill fit (the
              data-driven fit closest to the published dS*~1.6e-4, k=10).
  Bottom row -- the same pairs colored by f_r: (left) f_r vs dS_c, (right) genome
              dN/dS vs dS, sharing ONE f_r colorbar. A pair's colour is identical
              across the row, so the rare low-f_r pairs are visibly the low-dS /
              dN/dS~1 left end of the decay -- which is why f_r(dS_c) must pass
              through them rather than threading the overwhelming bulk.

See notes/fr_dSc_typical_pairs_dnds_decay.md.
"""
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
from scipy.optimize import minimize

from dnds_dynamics import config
from dnds_dynamics.figures import dynamics as dynamics_utils, theory as theory_utils

FIG_DIR = config.fig_path
FIG_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Hill accumulation + x-residual fit
# --------------------------------------------------------------------------- #
def hill_accumulation(dsc, ds_mid, k):
    """f_r = dS_c^k / (dS_c^k + dS_mid^k)."""
    dsc = np.asarray(dsc)
    return dsc ** k / (dsc ** k + ds_mid ** k)


def inverse_hill_accumulation(fr, ds_mid, k):
    fr = np.asarray(fr)
    return ds_mid * (fr / (1 - fr)) ** (1 / k)


def fit_hill_x_delta(dsc, fr, init_ds_mid=1e-4, init_k=2.0,
                     bounds_log_ds_mid=(-8, -1), bounds_log_k=(-3, 3), eps=1e-5):
    """Fit Hill params by minimizing horizontal residuals in log10(dS_c)."""
    dsc = np.asarray(dsc, float); fr = np.asarray(fr, float)
    m = np.isfinite(dsc) & np.isfinite(fr) & (dsc > 0) & (fr > eps) & (fr < 1 - eps)
    dsc = dsc[m]; fr = fr[m]
    log_dsc_obs = np.log10(dsc)

    def objective(p):
        ds_mid = 10 ** p[0]; k = 10 ** p[1]
        resid = log_dsc_obs - np.log10(inverse_hill_accumulation(fr, ds_mid, k))
        return np.mean(resid ** 2)

    res = minimize(objective, x0=[np.log10(init_ds_mid), np.log10(init_k)],
                   bounds=[bounds_log_ds_mid, bounds_log_k], method="L-BFGS-B")
    return {"ds_mid": 10 ** res.x[0], "k": 10 ** res.x[1], "n_used": len(dsc)}


def y_binned_summary_logx(x, y, bins=20, min_count=50, y_bin_scale="log",
                          interval=80, y_floor=1e-5):
    """Bin by y; within each y-bin summarize x (geo-mean + percentile spread)."""
    x = np.asarray(x); y = np.asarray(y)
    qlo = (100 - interval) / 2; qhi = 100 - qlo
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0)
    if y_bin_scale == "log":
        mask &= y > 0
    x = x[mask]; y = y[mask]
    if y_bin_scale == "linear":
        edges = np.linspace(y.min(), y.max(), bins + 1)
    else:
        edges = np.logspace(np.log10(max(y.min(), y_floor)), np.log10(y.max()), bins + 1)
    bin_idx = np.digitize(y, edges) - 1
    x_geo = np.full(bins, np.nan); x_lo = np.full(bins, np.nan); x_hi = np.full(bins, np.nan)
    y_mean = np.full(bins, np.nan)
    for i in range(bins):
        in_bin = bin_idx == i
        if np.sum(in_bin) >= min_count:
            xb = x[in_bin]; yb = y[in_bin]
            x_geo[i] = 10 ** np.mean(np.log10(xb))
            x_lo[i], x_hi[i] = np.percentile(xb, [qlo, qhi])
            y_mean[i] = np.mean(yb)
    keep = np.isfinite(x_geo)
    return {"x_geo_mean": x_geo[keep], "x_qlo": x_lo[keep], "x_qhi": x_hi[keep], "y_mean": y_mean[keep]}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
full_dnds_df = dynamics_utils.load_dNdS_data()
transfer_df = dynamics_utils.load_detected_transfers()
transfer_df = transfer_df.rename(columns={"Species name": "species_name",
                                          "Sample 1": "sample 1",
                                          "Sample 2": "sample 2"})
transfer_df.set_index(["species_name", "sample 1", "sample 2"], inplace=True)
means = (transfer_df.groupby(level=["species_name", "sample 1", "sample 2"])
         [["Clonal divergence", "Clonal fraction"]].mean())
result = full_dnds_df.join(means, how="right", rsuffix="_mean")
emp_dsc = result["Clonal divergence"]
emp_fr = 1 - result["Clonal fraction"]

# typical (fully recombined) pairs -> theta for the mixture model
typical_df = dynamics_utils.load_typical_pair_dNdS_data().copy()
typical_df["core_div_4D"] = typical_df["core_diff_4D"] / typical_df["core_len_4D"].astype(float)
typical_species = typical_df.groupby(level="species_name").agg(
    core_div_4D=("core_div_4D", "median"),
    core_diff_4D=("core_diff_4D", "median"),
    core_len_4D=("core_len_4D", "median"),
    core_diff_1D=("core_diff_1D", "median"),
    core_len_1D=("core_len_1D", "median"),
)
typical_dsc = typical_species["core_div_4D"].values
theta_typical = float(np.median(typical_dsc))

# single final fit: x-residual (closest to published dS*~1.6e-4, k=10)
fit = fit_hill_x_delta(emp_dsc, emp_fr, init_ds_mid=1e-4, init_k=2)
print(f"x-fit (single final): dS*={fit['ds_mid']:.2e}, k={fit['k']:.3g}")

# genome-wide dN/dS, per pair, with f_r attached (f_r=0 where no transfer detected)
cf = means["Clonal fraction"].reindex(full_dnds_df.index)
fr_all = np.asarray((1 - cf).fillna(0.0).values, dtype=float)
dS1, dS2 = dynamics_utils.computed_poisson_thinning(full_dnds_df["core_diff_4D"], full_dnds_df["core_len_4D"])
dS1 = np.asarray(dS1, float); dS2 = np.asarray(dS2, float)
dN_all = np.asarray((full_dnds_df["core_diff_1D"] / full_dnds_df["core_len_1D"].astype(float)).values, float)
ok = np.isfinite(dS1) & (dS2 > 0) & np.isfinite(dN_all) & np.isfinite(fr_all)
gx = dS1[ok]; gy = dN_all[ok] / dS2[ok]; gc = fr_all[ok]
gorder = np.argsort(gc)  # high f_r on top

# --------------------------------------------------------------------------- #
# Composite figure (2x2, equal panel boxes; colorbar in its own thin column)
# --------------------------------------------------------------------------- #
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm, Normalize
from matplotlib.lines import Line2D
import seaborn as sns  # registers the "rocket"/"flare"/"mako"... colormaps

mpl.rcParams["font.size"] = 7
# Red-toned "flare": low f_r = light orange (visible, not white like Reds),
# high f_r = dark red. Matches the red trend without washing out the low-f_r tail.
CMAP = "flare"
LOG_FR_FLOOR = 1e-3            # f_r floor for the log-color variant (f_r=0 -> floor)
dsc_curve = np.logspace(-6, -1, 300)
fr_curve = hill_accumulation(dsc_curve, fit["ds_mid"], fit["k"])

emp_dsc_arr = np.asarray(emp_dsc, float)
emp_fr_arr = np.asarray(emp_fr, float)
summary = y_binned_summary_logx(emp_dsc_arr, emp_fr_arr, bins=20, min_count=50,
                                y_bin_scale="log", interval=80)
mA = np.isfinite(emp_dsc_arr) & np.isfinite(emp_fr_arr) & (emp_dsc_arr > 0)
xA, frA = emp_dsc_arr[mA], emp_fr_arr[mA]
oA = np.argsort(frA)

dsc_theory = np.logspace(-7, -1, 300)
fr_th = hill_accumulation(dsc_theory, fit["ds_mid"], fit["k"])
dS_th, dNdS_th = theory_utils.compute_dNdS_rec_model(dsc_theory, fr_th, theta_typical, 1.0, 0.1)

XLIM_DSC = [0.5e-6, 2e-3]      # shared by the f_r-vs-dS_c column
YLIM_FR_LOG = [0.5e-4, 1.3]


def make_figure(log_color, out_name):
    if log_color:
        norm = LogNorm(vmin=LOG_FR_FLOOR, vmax=1.0)
        cA = np.clip(frA, LOG_FR_FLOOR, None)
        cB = np.clip(gc, LOG_FR_FLOOR, None)
    else:
        norm = Normalize(vmin=0.0, vmax=0.5)
        cA, cB = frA, gc

    fig = plt.figure(figsize=(5.6, 5.0))
    gs = GridSpec(2, 3, width_ratios=[1, 1, 0.05], wspace=0.5, hspace=0.45, figure=fig)
    ax00 = fig.add_subplot(gs[0, 0]); ax01 = fig.add_subplot(gs[0, 1])
    ax10 = fig.add_subplot(gs[1, 0]); ax11 = fig.add_subplot(gs[1, 1])
    cax = fig.add_subplot(gs[1, 2])

    # Top-left: log-log accumulation, grey + binned mean/spread + expected trend
    ax00.scatter(emp_dsc_arr, emp_fr_arr, s=2, alpha=0.10, color="tab:grey", rasterized=True)
    ax00.fill_betweenx(summary["y_mean"], summary["x_qlo"], summary["x_qhi"],
                       color="tab:red", alpha=0.2, lw=0)
    ax00.plot(summary["x_geo_mean"], summary["y_mean"], color="tab:red", lw=1.0)
    ax00.plot(dsc_curve, fr_curve, color="k", lw=1.0)
    ax00.set_xscale("log"); ax00.set_yscale("log")
    ax00.set_xlim(XLIM_DSC); ax00.set_ylim(YLIM_FR_LOG)
    ax00.set_xlabel(r"$dS_c$ (Clonal region)"); ax00.set_ylabel(r"$f_r$ (Recombined fraction)")

    # Top-right: log-linear accumulation, grey + expected trend (+ legend)
    ax01.scatter(emp_dsc_arr, emp_fr_arr, s=2, alpha=0.10, color="tab:grey", rasterized=True)
    ax01.plot(dsc_curve, fr_curve, color="k", lw=1.0)
    ax01.set_xscale("log"); ax01.set_xlim(XLIM_DSC); ax01.set_ylim([0, 1.0])
    ax01.set_xlabel(r"$dS_c$ (Clonal region)"); ax01.set_ylabel(r"$f_r$ (Recombined fraction)")
    legend_handles = [
        Line2D([0], [0], color="tab:red", lw=1.0, label="Binned mean\n(80% spread)"),
        Line2D([0], [0], color="k", lw=1.0, label="Expected trend"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="tab:grey",
               markeredgecolor="none", markersize=4, alpha=0.6, label="Observed (per pair)"),
    ]
    ax01.legend(handles=legend_handles, fontsize=5.5, frameon=False, loc="upper left")

    # Bottom-left: f_r vs dS_c colored by f_r + expected trend
    ax10.scatter(xA[oA], frA[oA], c=cA[oA], cmap=CMAP, norm=norm, s=4, alpha=0.65, rasterized=True)
    ax10.plot(dsc_curve, fr_curve, color="k", lw=1.0)
    ax10.set_xscale("log"); ax10.set_yscale("log")
    ax10.set_xlim(XLIM_DSC); ax10.set_ylim(YLIM_FR_LOG)
    ax10.set_xlabel(r"$dS_c$ (Clonal region)"); ax10.set_ylabel(r"$f_r$ (Recombined fraction)")

    # Bottom-right: genome dN/dS vs dS colored by f_r + mixture (no legend)
    scB = ax11.scatter(gx[gorder], gy[gorder], c=cB[gorder], cmap=CMAP, norm=norm,
                       s=4, alpha=0.6, rasterized=True)
    ax11.plot(dS_th, dNdS_th, color="k", lw=1.0)
    ax11.axhline(1, lw=0.5, ls="--", color="grey")
    ax11.set_xscale("log"); ax11.set_yscale("log")
    ax11.set_xlim([2e-6, 2e-2]); ax11.set_ylim([1e-2, 1e1])   # match Fig 2 panel
    ax11.set_xlabel(r"$dS$ (Full genome)"); ax11.set_ylabel(r"$dN/dS$", labelpad=-1)

    cb = fig.colorbar(scB, cax=cax)
    cb.set_label(r"$f_r$ (recombined fraction)", fontsize=7)
    cb.ax.tick_params(labelsize=6)

    fig.savefig(FIG_DIR / out_name, bbox_inches="tight", dpi=600)
    print("wrote", FIG_DIR / out_name)


make_figure(log_color=False, out_name="recombination_dynamics_fit_with_fr.pdf")
make_figure(log_color=True, out_name="recombination_dynamics_fit_with_fr_logcolor.pdf")
