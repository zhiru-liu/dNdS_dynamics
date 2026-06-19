"""Refit the clonal-dN/dS purifying model with dense-window (missed-recombination)
SNVs masked, and remake the two publication figures.

Uses the same fit as the main clonal-dN/dS figure (Poisson-thin 4D into
x/denominator; 15 geomspace bins; ratio-of-totals R_hat per bin; WLS fit of s/mu
with fd=0.9 fixed; reference curves s/mu=1e5,1e3; unrelated-pair species means;
detection shading dS>1e-3).

Variants compared:
  - no-Ap, unmasked      (reproduces published s/mu ~ 1.1e4)
  - with-Ap, unmasked    (how much Ap alone distorts the estimate)
  - with-Ap, masked      (requested figure)
  - no-Ap,  masked

Inputs: outputs/masked_clonal_fit/per_pair_masked_clonal_counts.csv
Figures:
  figures/clonal_dNdS_purifying_fit_withAp_masked.{png,pdf}
  figures/clonal_dNdS_species_grid_masked.{png,pdf}
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.random import default_rng
from scipy.optimize import minimize_scalar

REPO_ROOT = Path(__file__).resolve().parents[1]
MPL_CACHE_DIR = REPO_ROOT / ".cache" / "matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

# Reuse the per-species binning/bootstrap so the grid matches exactly.
sys.path.insert(0, str(REPO_ROOT))
from dnds_dynamics.figures import per_species_dnds as psd  # noqa: E402

DN = REPO_ROOT / "data"
TYPICAL_DIR = DN / "gut_microbiome_typical_pair_dNdS"
PAIR_TABLE = REPO_ROOT / "outputs" / "masked_clonal_fit" / "per_pair_masked_clonal_counts.csv"
FIG_FIT = REPO_ROOT / "figures" / "clonal_dNdS_purifying_fit_withAp_masked"
FIG_GRID = REPO_ROOT / "figures" / "clonal_dNdS_species_grid_masked"
AP = "Alistipes_putredinis_61533"

FD_FIXED = 0.9
P_THIN = 0.5
N_BINS = 15
MIN_PAIRS = 10
EPS = 1e-12
DETECTION = 1e-3

# The 16 species shown in the published grid (clonal_dNdS_species_grid_all.pdf),
# in panel order (descending close+clonal pair count). Aligns the grid to the
# original; the aggregate purifying fit still uses the full cohort.
GRID_SPECIES = [
    "Alistipes_putredinis_61533", "Bacteroides_caccae_53434",
    "Bacteroides_uniformis_57318", "Alistipes_onderdonkii_55464",
    "Bacteroides_stercoris_56735", "Bacteroides_massiliensis_44749",
    "Bacteroides_ovatus_58035", "Barnesiella_intestinihominis_62208",
    "Bacteroides_vulgatus_57955", "Parabacteroides_merdae_56972",
    "Parabacteroides_distasonis_56985", "Dialister_invisus_61905",
    "Akkermansia_muciniphila_55290", "Bacteroides_cellulosilyticus_58046",
    "Alistipes_shahii_62199", "Bacteroides_eggerthii_54457",
]


def dNdS_purify_curve(dS, fd, sbymu):
    return (1 - fd) + fd * (1 - np.exp(-sbymu * dS / 2)) / (sbymu * dS / 2)


def thin_counts(kS, LS, kN, LN, p, rng):
    kS = kS.astype(int)
    kS_A = rng.binomial(np.clip(kS, 0, None), p)
    kS_B = kS - kS_A
    LS_A = p * LS; LS_B = (1 - p) * LS
    dS_A = kS_A / np.clip(LS_A, EPS, None)
    return pd.DataFrame({"kS_A": kS_A, "kS_B": kS_B, "LS_A": LS_A, "LS_B": LS_B,
                         "dS_A": dS_A, "kN": kN, "LN": LN})


def make_bins(dS_A, n_bins):
    d = dS_A[dS_A > 0]
    return np.geomspace(d.min(), d.max(), n_bins + 1)


def aggregate_bins(th, edges):
    df = th[th["dS_A"] > 0].copy()
    df["bin"] = pd.cut(df["dS_A"], edges, labels=False, include_lowest=True)
    gb = df.groupby("bin", dropna=True)
    agg = gb.agg(KN=("kN", "sum"), LN=("LN", "sum"), KS_A=("kS_A", "sum"),
                 KS_B=("kS_B", "sum"), LS_A=("LS_A", "sum"), LS_B=("LS_B", "sum"),
                 n_pairs=("kN", "size")).reset_index()
    agg["dS_x"] = agg["KS_A"] / agg["LS_A"].replace(0, np.nan)
    agg["R_hat"] = (agg["KN"] / agg["LN"].replace(0, np.nan)) / (agg["KS_B"] / agg["LS_B"].replace(0, np.nan))
    return agg


def fit_sbymu(agg, fd=FD_FIXED, bounds=(1e-3, 1e7)):
    m = ((agg["n_pairs"] >= MIN_PAIRS) & (agg["KN"] > 0) & (agg["KS_B"] > 0) &
         (agg["dS_x"] > 0) & (agg["R_hat"] > 0) & np.isfinite(agg["R_hat"]))
    if m.sum() < 3:
        return np.nan
    x = agg.loc[m, "dS_x"].to_numpy(float); y = agg.loc[m, "R_hat"].to_numpy(float)
    KN = agg.loc[m, "KN"].to_numpy(float); KS = agg.loc[m, "KS_B"].to_numpy(float)
    w = 1.0 / (1.0 / np.maximum(KN, 1.0) + 1.0 / np.maximum(KS, 1.0))

    def obj(sb):
        pred = dNdS_purify_curve(x, fd, sb)
        return np.average((np.log(y) - np.log(pred)) ** 2, weights=w)
    return minimize_scalar(obj, bounds=bounds, method="bounded").x


def prep_counts(df, masked: bool):
    kS = df["clonal_diff_4D"].to_numpy(float).copy()
    kN = df["clonal_diff_1D"].to_numpy(float).copy()
    if masked:
        kS = np.clip(kS - df["rm4"].to_numpy(float), 0, None)
        kN = np.clip(kN - df["rm1"].to_numpy(float), 0, None)
    LS = df["clonal_len_4D"].to_numpy(float)
    LN = df["clonal_len_1D"].to_numpy(float)
    return kS, LS, kN, LN


def fit_variant(df, masked, seed=123):
    rng = default_rng(seed)
    kS, LS, kN, LN = prep_counts(df, masked)
    th = thin_counts(kS, LS, kN, LN, P_THIN, rng)
    edges = make_bins(th["dS_A"].to_numpy(), N_BINS)
    agg = aggregate_bins(th, edges)
    sb = fit_sbymu(agg)
    # bootstrap CI
    sbs = []
    for _ in range(200):
        idx = rng.integers(0, len(df), len(df))
        d = df.iloc[idx]
        kS2, LS2, kN2, LN2 = prep_counts(d, masked)
        th2 = thin_counts(kS2, LS2, kN2, LN2, P_THIN, rng)
        a2 = aggregate_bins(th2, edges)
        s2 = fit_sbymu(a2)
        if np.isfinite(s2):
            sbs.append(s2)
    sbs = np.array(sbs)
    ci = (np.percentile(sbs, 2.5), np.percentile(sbs, 97.5)) if len(sbs) else (np.nan, np.nan)
    return sb, ci, th, agg, edges


def species_unrelated_means():
    """Per-species unrelated-pair dN/dS vs dS (ratio-of-totals over typical pairs)."""
    out = []
    for f in sorted(TYPICAL_DIR.glob("*.csv")):
        t = pd.read_csv(f)
        if not {"core_diff_4D", "core_len_4D", "core_diff_1D", "core_len_1D"}.issubset(t.columns):
            continue
        dS = t["core_diff_4D"].sum() / max(t["core_len_4D"].sum(), 1)
        dN = t["core_diff_1D"].sum() / max(t["core_len_1D"].sum(), 1)
        if dS > 0:
            out.append((f.stem, dS, dN / dS))
    return pd.DataFrame(out, columns=["species", "dS", "dNdS"])


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair-table", type=Path, default=PAIR_TABLE,
                    help="per-pair counts; mask cols rm1/rm4 or moved_1d/moved_4d")
    ap.add_argument("--suffix", default="", help="appended to output figure names")
    args = ap.parse_args()
    global FIG_FIT, FIG_GRID
    if args.suffix:
        FIG_FIT = FIG_FIT.with_name(FIG_FIT.name + args.suffix)
        FIG_GRID = FIG_GRID.with_name(FIG_GRID.name + args.suffix)

    df_all = pd.read_csv(args.pair_table)
    if "rm1" not in df_all.columns and "moved_1d" in df_all.columns:
        df_all = df_all.rename(columns={"moved_1d": "rm1", "moved_4d": "rm4"})
    df_noap = df_all[df_all["species"] != AP]
    print(f"table={args.pair_table.name}  pairs: all={len(df_all)}, "
          f"no-Ap={len(df_noap)}, Ap={len(df_all)-len(df_noap)}")

    print("\n=== fitted s/mu (fd=0.9 fixed) ===")
    table = []
    for label, d, mk in [("no-Ap, unmasked", df_noap, False),
                         ("with-Ap, unmasked", df_all, False),
                         ("no-Ap, masked", df_noap, True),
                         ("with-Ap, masked", df_all, True)]:
        sb, ci, *_ = fit_variant(d, mk)
        table.append((label, sb, ci))
        print(f"  {label:20s}: s/mu = {sb:.3g}   95% CI [{ci[0]:.3g}, {ci[1]:.3g}]")

    # ---- Figure 1: purifying fit, with-Ap masked ----
    sb_m, ci_m, th_m, agg_m, edges = fit_variant(df_all, masked=True)
    # also fit with-Ap unmasked for overlay reference
    sb_u, *_ = fit_variant(df_all, masked=False)
    _plot_fit(df_all, th_m, agg_m, sb_m, ci_m, sb_u)

    # ---- Figure 2: species grid (masked) ----
    _plot_grid(df_all, sb_m)
    print(f"\nwrote {FIG_FIT.with_suffix('.png')}")
    print(f"wrote {FIG_GRID.with_suffix('.png')}")


def _per_pair_xy(df, masked, seed=7):
    rng = default_rng(seed)
    kS, LS, kN, LN = prep_counts(df, masked)
    th = thin_counts(kS, LS, kN, LN, P_THIN, rng)
    dS_A = th["dS_A"].to_numpy()
    dS_B = th["kS_B"].to_numpy() / np.clip(th["LS_B"].to_numpy(), EPS, None)
    dN = th["kN"].to_numpy() / np.clip(th["LN"].to_numpy(), EPS, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        R = dN / dS_B
    return dS_A, R


def _plot_fit(df, th, agg, sb, ci, sb_unmasked):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dS_A, R = _per_pair_xy(df, masked=True)
    ok = (dS_A > 0) & np.isfinite(R) & (R > 0)
    grid = np.geomspace(1e-6, 1e-1, 400)
    unrel = species_unrelated_means()

    fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.axvspan(DETECTION, 1e-1, color="0.92", zorder=0)
    ax.scatter(dS_A[ok], R[ok], s=5, alpha=0.12, color="#AECDE1", linewidths=0, rasterized=True, zorder=1)

    m = ((agg["n_pairs"] >= MIN_PAIRS) & (agg["dS_x"] > 0) & (agg["R_hat"] > 0) & np.isfinite(agg["R_hat"]))
    ax.plot(agg.loc[m, "dS_x"], agg.loc[m, "R_hat"], "-o", color="#0072B2", ms=4, lw=1.4,
            label="Mean (ratio-of-totals) per bin", zorder=4)

    ax.plot(grid, dNdS_purify_curve(grid, FD_FIXED, sb), lw=2, color="tab:orange",
            label=fr"Purifying fit, masked ($s/\mu={sb:.2g}$)", zorder=6)
    ax.plot(grid, dNdS_purify_curve(grid, FD_FIXED, sb_unmasked), lw=1.6, ls="-.", color="#d62728",
            label=fr"Fit, unmasked w/ Ap ($s/\mu={sb_unmasked:.2g}$)", zorder=5)
    ax.plot(grid, dNdS_purify_curve(grid, FD_FIXED, 1e5), ls="--", lw=1.2, color="#E69F00",
            label=r"Ref: $s/\mu=10^5$", zorder=3)
    ax.plot(grid, dNdS_purify_curve(grid, FD_FIXED, 1e3), ls=":", lw=1.4, color="#E69F00",
            label=r"Ref: $s/\mu=10^3$", zorder=3)
    if np.isfinite(ci[0]):
        ax.fill_between(grid, dNdS_purify_curve(grid, FD_FIXED, ci[0]),
                        dNdS_purify_curve(grid, FD_FIXED, ci[1]), color="tab:orange", alpha=0.15, zorder=2)
    ax.scatter(unrel["dS"], unrel["dNdS"], marker="x", s=28, color="#009E73",
               label="Unrelated pairs (species mean)", zorder=5)

    ax.axhline(1, color="0.5", lw=0.8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1e-6, 1e-1); ax.set_ylim(8e-2, 1.5e1)
    ax.set_xlabel("$dS$ (clonal region)"); ax.set_ylabel("$dN/dS$ (clonal region)")
    ax.set_title("Clonal dN/dS purifying fit — with A. putredinis, dense windows masked")
    ax.legend(fontsize=7, loc="lower left", framealpha=0.9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_FIT.with_suffix(f".{ext}"), bbox_inches="tight")
    plt.close(fig)


CLONAL_COLOR = "#AECDE1"   # light blue scatter (matches original figure_utils.clonal_color)
TREND_UNMASK = "#9e9e9e"   # grey
TREND_MASK = "#0072B2"     # deep blue


def _species_thinned(sub, seed=42):  # match original ThinSettings.rng_seed=42
    """Shared Poisson thinning on UNMASKED 4D: same x & denominator for both lines,
    so the masked-vs-unmasked difference is purely the nonsynonymous numerator."""
    rng = default_rng(seed)
    kS = sub["clonal_diff_4D"].to_numpy(float)
    LS = sub["clonal_len_4D"].to_numpy(float)
    kS_A = rng.binomial(np.clip(kS, 0, None).astype(int), P_THIN)
    kS_B = kS - kS_A
    LS_A = P_THIN * LS; LS_B = (1 - P_THIN) * LS
    dS_A = kS_A / np.clip(LS_A, EPS, None)
    LN = sub["clonal_len_1D"].to_numpy(float)
    kN_u = sub["clonal_diff_1D"].to_numpy(float)
    kN_m = np.clip(kN_u - sub["rm1"].to_numpy(float), 0, None)
    return dict(dS_A=dS_A, kS_A=kS_A, LS_A=LS_A, kS_B=kS_B, LS_B=LS_B, LN=LN, kN_u=kN_u, kN_m=kN_m)


def _auto_bin_edges(dS_A, target=50, min_bins=4, max_bins=20, min_pairs_per_bin=20):
    """Per-species adaptive geometric bin edges, matching
    per_species_dnds._auto_bin_count / ps_make_bins_auto:
      n_bins = clip(n_pairs // target, min_bins, max_bins), but 0 if
      n_pairs < min_pairs_per_bin*min_bins; edges span the species' dS_A range.
    """
    d = dS_A[(dS_A > 0) & np.isfinite(dS_A)]
    n = d.size
    if n <= 0:
        return None
    nb = max(min_bins, min(max_bins, n // max(1, target)))
    nb = max(min_bins, nb) if n >= min_pairs_per_bin * min_bins else 0
    if nb <= 0:
        return None
    lo, hi = float(d.min()), float(d.max())
    if not (np.isfinite(lo) and np.isfinite(hi)) or lo <= 0 or hi <= lo:
        return None
    return np.geomspace(lo, hi, nb + 1)


def _binned_trend(t, kN, edges, min_pairs=20, min_syn=1, nboot=300, seed=0):
    """Ratio-of-totals dN/dS per dS bin with bootstrap 95% CI. Returns x, y, lo, hi.

    Drops bins with < ``min_pairs`` pairs (matches the original
    per_species_dnds.min_pairs_per_bin=20) or too few pooled synonymous SNVs in
    the denominator, which otherwise produce noisy low-dS outlier points.
    """
    rng = default_rng(seed)
    dS_A = t["dS_A"]; idx = np.digitize(dS_A, edges) - 1
    xs, ys, los, his = [], [], [], []
    for b in range(len(edges) - 1):
        m = (idx == b) & (dS_A > 0)
        ii = np.flatnonzero(m)
        if ii.size < min_pairs or t["kS_B"][ii].sum() < min_syn:
            continue
        denom = t["kS_B"][ii].sum() / max(t["LS_B"][ii].sum(), EPS)
        numer = kN[ii].sum() / max(t["LN"][ii].sum(), EPS)
        if denom <= 0 or numer <= 0:
            continue
        x = t["kS_A"][ii].sum() / max(t["LS_A"][ii].sum(), EPS)
        rr = []
        for _ in range(nboot):
            s = ii[rng.integers(0, ii.size, ii.size)]
            d = t["kS_B"][s].sum() / max(t["LS_B"][s].sum(), EPS)
            n = kN[s].sum() / max(t["LN"][s].sum(), EPS)
            if d > 0 and n > 0:
                rr.append(n / d)
        if not rr:
            continue
        xs.append(x); ys.append(numer / denom)
        los.append(np.percentile(rr, 2.5)); his.append(np.percentile(rr, 97.5))
    return map(np.asarray, (xs, ys, los, his))


def _plot_grid(df, sb):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    # palette matching the main figure (figure_utils COLORS)
    C_CLOUD = "#AECDE1"; C_TREND = "#0072B2"; C_NOMASK = "#9e9e9e"
    C_FIT = "tab:orange"; C_SPECIES = "#009E73"; C_NEUTRAL = "#6E6E6E"; C_SHADE = "#AAAAAA"
    XLIM = (1e-6, 1e-1); YLIM = (0.03, 20.0)
    SHADE_MIN, SHADE_MAX = 1e-3, 1.0

    present = set(df["species"].unique())
    species = [s for s in GRID_SPECIES if s in present]
    missing = [s for s in GRID_SPECIES if s not in present]
    if missing:
        print(f"  grid: {len(missing)} reference species missing from data: {missing}")
    grid = np.geomspace(max(XLIM[0], 1e-12), XLIM[1], 400)
    yfit = dNdS_purify_curve(grid, FD_FIXED, sb)
    unrel = species_unrelated_means().set_index("species")

    # Per-species binning settings (match the main figure).
    settings = psd.ThinSettings(auto_bins=True, target_pairs_per_bin=50, min_pairs_per_bin=20,
                                min_bins=4, min_bins_retained=3, min_pairs_per_species=10)

    def _trend(agg0, boots):
        """(x, med, lo, hi) from an (agg0, boots) pair, or empty."""
        if agg0 is None or len(agg0) == 0:
            return (np.array([]),) * 4
        m = agg0.merge(boots, on="bin", how="inner") if (boots is not None and len(boots)) \
            else agg0.assign(med=agg0["R_hat"], lo=agg0["R_hat"], hi=agg0["R_hat"])
        if "med" not in m:
            m["med"], m["lo"], m["hi"] = m["R_hat"], m["R_hat"], m["R_hat"]
        return (m["dS_x"].to_numpy(float), m["med"].to_numpy(float),
                m["lo"].to_numpy(float), m["hi"].to_numpy(float))

    ncols = 4
    nrows = int(np.ceil(len(species) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, max(3.0, 2.0 + 2.0 * nrows)),
                             dpi=300, sharex=True, sharey=True)
    axes = np.atleast_1d(axes).reshape(nrows, ncols)
    legend_done = False

    for k, sp in enumerate(species):
        ax = axes[k // ncols, k % ncols]
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        sub = df[df["species"] == sp].reset_index(drop=True)
        # one synonymous thinning (unmasked) -> shared bins/denominator/x-axis.
        tb_u = psd.prepare_thin_base_for_classes(sub, settings, classes=("all",))["all"]
        # masking removes ONLY the nonsynonymous (1D) missed-tract SNVs; the
        # synonymous binning/denominator (and thus the bins) stay fixed.
        tb_m = tb_u.copy()
        tb_m["kN"] = np.clip(sub["clonal_diff_1D"].to_numpy(float) - sub["rm1"].to_numpy(float), 0, None)
        agg_u, boots_u = psd.ps_bootstrap_binned(tb_u, "kN", "LN", settings)
        agg_m, boots_m = psd.ps_bootstrap_binned(tb_m, "kN", "LN", settings)
        # unmasked scatter cloud (light blue), exactly the reference cloud
        with np.errstate(divide="ignore", invalid="ignore"):
            R_pair = (tb_u["kN"].to_numpy() / np.clip(tb_u["LN"].to_numpy(), EPS, None)) / \
                     (tb_u["kS_B"].to_numpy() / np.clip(tb_u["LS_B"].to_numpy(), EPS, None))
        x_sc = tb_u["dS_A"].to_numpy()
        ok = (x_sc > 0) & np.isfinite(R_pair) & (R_pair > 0)
        ax.scatter(x_sc[ok], R_pair[ok], s=6, alpha=0.20, color=C_CLOUD,
                   linewidths=0, rasterized=True, zorder=1)
        xu, yu, lou, hiu = _trend(agg_u, boots_u)
        xm, ym, lom, him = _trend(agg_m, boots_m)
        if len(xu):
            ax.errorbar(xu, yu, yerr=[np.clip(yu - lou, 0, None), np.clip(hiu - yu, 0, None)],
                        fmt="-o", lw=1.5, ms=4.5, mew=0.0, color=C_NOMASK, ecolor=C_NOMASK,
                        elinewidth=1.0, capsize=2.5, capthick=1.0, zorder=3,
                        label="No mask (median ± 95% CI)")
        if len(xm):
            ax.errorbar(xm, ym, yerr=[np.clip(ym - lom, 0, None), np.clip(him - ym, 0, None)],
                        fmt="-o", lw=1.5, ms=4.5, mew=0.0, color=C_TREND, ecolor=C_TREND,
                        elinewidth=1.0, capsize=2.5, capthick=1.0, zorder=4,
                        label="Masked (median ± 95% CI)")
        ax.plot(grid, yfit, lw=2.0, color=C_FIT, zorder=6,
                label=fr"Purifying fit (agg) $s/\mu={sb:.2g}$")
        if sp in unrel.index:
            ax.scatter([unrel.loc[sp, "dS"]], [unrel.loc[sp, "dNdS"]], marker="X", s=50,
                       linewidths=0.8, edgecolors="white", color=C_SPECIES, zorder=7,
                       label="Unrelated pairs (species mean)")
        ax.axvspan(SHADE_MIN, SHADE_MAX, color=C_SHADE, alpha=0.12, zorder=0)
        ax.axhline(1.0, color=C_NEUTRAL, linestyle="-", lw=1.0, zorder=2)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(XLIM); ax.set_ylim(YLIM)
        ax.set_title(f"{sp} (n={len(sub)})", fontsize=10)
        if not legend_done:
            sh = Patch(facecolor=C_SHADE, alpha=0.12, label="Approx. detection limit")
            h, lab = ax.get_legend_handles_labels()
            h.append(sh); lab.append("Approx. detection limit")
            ax.legend(h, lab, frameon=True, loc="lower left", ncol=1, handlelength=2.2, fontsize=7)
            legend_done = True

    for j in range(len(species), nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")
    fig.text(0.5, 0.02, r"$dS$ (Clonal region)", ha="center")
    fig.text(0.01, 0.5, r"$dN/dS$ (Clonal region)", va="center", rotation="vertical")
    fig.tight_layout(rect=(0.03, 0.03, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(FIG_GRID.with_suffix(f".{ext}"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
