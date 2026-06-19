"""Reviewer R1.2 -- is the spread in clonal dN/dS around the purifying-fit curve
larger than a NAIVE Poisson null with realistic 1D/4D site densities and clonal
genome lengths?  Pooled, and within species.

The figure `clonal_dNdS_purifying_fit.pdf` shows a large light-blue cloud of
per-pair dN/dS straddling the orange curve over ~2 decades.  Most of that cloud
sits at low dS where each pair carries only a handful of synonymous diffs, so the
per-pair ratio dN/dS is a ratio of small integer counts and is intrinsically
noisy.  This script quantifies how much of the spread is finite-site sampling
noise (predicted by the model) vs genuine pair-to-pair / within-species
heterogeneity (excess over the model).

NULL MODEL (naive purifying, shared-thinning conditional): the single global
curve
    dN/dS = f(dS) = (1-fd) + fd * (1 - exp(-(s/mu) dS/2)) / ((s/mu) dS/2)
with fd=0.9 and the fitted s/mu holds for EVERY pair, with NO biological
pair-to-pair variation.

To absorb the real data-processing into the theory, the synonymous channel is
NOT re-simulated: in each replicate we run the figure's actual processing on the
REAL 4D counts -- Poisson-thin into A/B halves (p=0.5), x = dS_A = kS_A/LS_A,
denom = kS_B/LS_B -- and reuse that exact thinning for both clouds.  The ONLY
simulated quantity is the nonsynonymous numerator, drawn from the curve at the
pair's own synonymous clock dS_i = kS_i/LS_i:
    kN_null ~ Poisson( f(dS_i) * dS_i * LN_i )
    R_obs  = (kN_obs  / LN) / denom     R_null = (kN_null / LN) / denom
So the x-axis, the denominator, and the thinning noise are IDENTICAL between data
and theory; the observed-vs-theory spread difference is purely the real
nonsynonymous numerator deviating from Poisson-around-the-curve.

We compare, in the figure's fixed dS_A bins, the conditional spread of
log10(dN/dS) for the real cloud vs this theory -- pooled and per species.

Outputs (figures/):
  purify_spread_null_overlay.pdf     scatter + observed vs null spread envelopes
  purify_spread_per_bin.pdf          SD(log10 dN/dS): observed vs null + inflation
  purify_spread_within_species.pdf   per-species variance-inflation factor
and a printed summary table.
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
sys.path.insert(0, str(REPO_ROOT))
MPL_CACHE = REPO_ROOT / ".cache" / "matplotlib"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

from dnds_dynamics.figures import dynamics as dynamics_utils  # noqa: E402
from dnds_dynamics.figures.theory import dNdS_purify_curve     # noqa: E402

FIGDIR = REPO_ROOT / "figures"
FD = 0.9
P_THIN = 0.5
N_BINS = 15
MIN_PAIRS = 10          # min pairs/bin to fit (matches figure)
# Min UNIQUE pairs/bin to draw a spread band. The cloud is pooled over many
# thinning replicates, so len(o) is replicate-inflated (~unique*N_SIM); the
# threshold must be on unique pairs or near-empty end bins masquerade as bands.
# 95% percentile envelopes are tail-sensitive -> need a few dozen real pairs;
# SD-based (within-species) tolerates fewer since SD uses all points.
MIN_UNIQUE_OVERLAY = 40
MIN_UNIQUE_SPECIES = 20
EPS = 1e-12
DETECTION = 1e-3
N_SIM = 200             # null replicates (per-pair resimulation)
N_OBS_THIN = 50         # thinning realizations to average the OBSERVED spread
AP = "Alistipes_putredinis_61533"


# --------------------------------------------------------------------------- #
#  Data + fit (reproduce the published 5.4e3 fit on the same counts)
# --------------------------------------------------------------------------- #
def load_counts():
    df = dynamics_utils.load_dNdS_data()
    sp = df.index.get_level_values("species_name").to_numpy()
    c = pd.DataFrame({
        "species": sp,
        "kN": df["clonal_diff_1D"].to_numpy(float),
        "LN": df["clonal_len_1D"].to_numpy(float),
        "kS": df["clonal_diff_4D"].to_numpy(float),
        "LS": df["clonal_len_4D"].to_numpy(float),
    })
    return c[(c.LN > 0) & (c.LS > 0)].reset_index(drop=True)


def thin(kS, LS, p, rng):
    kS = np.clip(kS, 0, None).astype(int)
    kS_A = rng.binomial(kS, p)
    kS_B = kS - kS_A
    return kS_A, kS_B, p * LS, (1 - p) * LS


def fit_sbymu(c, rng):
    kS_A, kS_B, LS_A, LS_B = thin(c.kS.to_numpy(), c.LS.to_numpy(), P_THIN, rng)
    dS_A = kS_A / np.clip(LS_A, EPS, None)
    pos = dS_A > 0
    edges = np.geomspace(dS_A[pos].min(), dS_A[pos].max(), N_BINS + 1)
    b = np.clip(np.digitize(dS_A, edges) - 1, 0, N_BINS - 1)
    t = pd.DataFrame(dict(b=b, kN=c.kN, LN=c.LN, kS_A=kS_A, kS_B=kS_B,
                          LS_A=LS_A, LS_B=LS_B, dS_A=dS_A))
    t = t[t.dS_A > 0]
    g = t.groupby("b").agg(KN=("kN", "sum"), LN=("LN", "sum"), KSA=("kS_A", "sum"),
                           KSB=("kS_B", "sum"), LSA=("LS_A", "sum"),
                           LSB=("LS_B", "sum"), n=("kN", "size")).reset_index()
    g["dS_x"] = g.KSA / g.LSA
    g["R"] = (g.KN / g.LN) / (g.KSB / g.LSB)
    m = (g.n >= MIN_PAIRS) & (g.KN > 0) & (g.KSB > 0) & (g.dS_x > 0) & (g.R > 0) & np.isfinite(g.R)
    x = g.dS_x[m].to_numpy(); y = g.R[m].to_numpy()
    w = 1.0 / (1.0 / np.maximum(g.KN[m], 1) + 1.0 / np.maximum(g.KSB[m], 1))

    def obj(sb):
        return np.average((np.log(y) - np.log(dNdS_purify_curve(x, FD, sb))) ** 2, weights=w)
    return float(minimize_scalar(obj, bounds=(1e-3, 1e7), method="bounded").x), edges


# --------------------------------------------------------------------------- #
#  Per-pair scatter (x = dS_A, R = dN/dS) for one thinning realization
# --------------------------------------------------------------------------- #
def pair_xy(kN, LN, kS, LS, rng):
    kS_A, kS_B, LS_A, LS_B = thin(kS, LS, P_THIN, rng)
    dS_A = kS_A / np.clip(LS_A, EPS, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        R = (kN / LN) / (kS_B / np.clip(LS_B, EPS, None))
    ok = (dS_A > 0) & (kS_B > 0) & np.isfinite(R) & (R > 0)
    return dS_A, R, ok


def binned_logR_spread(dS_A, R, ok, edges):
    """Per-bin distribution of log10 R: returns dict bin->array of log10 R."""
    b = np.digitize(dS_A, edges) - 1
    out = {}
    for k in range(len(edges) - 1):
        sel = ok & (b == k)
        if sel.sum():
            out[k] = np.log10(R[sel])
    return out


def accumulate(per_bin_lists, spread_dict):
    for k, v in spread_dict.items():
        per_bin_lists.setdefault(k, []).append(v)


# --------------------------------------------------------------------------- #
#  Paired observed/null spread with SHARED thinning ("theory absorbs the
#  synonymous data processing"): in every replicate we thin the REAL 4D counts
#  once and reuse that exact thinning for both clouds, so the x-axis (dS_A), the
#  denominator (kS_B/LS_B) and the thinning noise are IDENTICAL between data and
#  theory.  The ONLY simulated quantity is the nonsynonymous numerator:
#      kN_null ~ Poisson( f(dS_i) * dS_i * LN_i ),   dS_i = kS_i / LS_i
#  Hence the observed-vs-null spread difference is purely the real kN deviating
#  from Poisson-around-the-curve (biology / undetected recombination), with the
#  thinning + denominator sampling fully shared into the theory prediction.
# --------------------------------------------------------------------------- #
def model_muN(c, sbymu):
    LN, LS, kS = c.LN.to_numpy(), c.LS.to_numpy(), c.kS.to_numpy()
    dS_true = kS / np.clip(LS, EPS, None)        # observed synonymous clock
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = dNdS_purify_curve(dS_true, FD, sbymu)
    ratio = np.where(dS_true > 0, ratio, 1.0)    # curve -> 1 as dS -> 0
    return np.clip(ratio * dS_true * LN, 0, None)


def paired_spread(c, edges, sbymu, n_rep, seed=0):
    """Returns (obs_bins, null_bins): dict bin -> array of log10(dN/dS), where
    each replicate shares one real thinning of the observed 4D counts between the
    two clouds and only the null's nonsynonymous numerator is re-simulated."""
    rng = default_rng(seed)
    kN, LN, kS, LS = (c.kN.to_numpy(), c.LN.to_numpy(),
                      c.kS.to_numpy(), c.LS.to_numpy())
    muN = model_muN(c, sbymu)
    obs_bins, null_bins = {}, {}
    for _ in range(n_rep):
        # ONE shared thinning of the REAL synonymous counts (data processing)
        kS_A, kS_B, LS_A, LS_B = thin(kS, LS, P_THIN, rng)
        dS_A = kS_A / np.clip(LS_A, EPS, None)
        denom = kS_B / np.clip(LS_B, EPS, None)
        base_ok = (dS_A > 0) & (kS_B > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            R_obs = (kN / LN) / denom                 # real numerator
            kN_null = rng.poisson(muN)
            R_null = (kN_null / LN) / denom           # theory numerator
        accumulate(obs_bins, binned_logR_spread(
            dS_A, R_obs, base_ok & np.isfinite(R_obs) & (R_obs > 0), edges))
        accumulate(null_bins, binned_logR_spread(
            dS_A, R_null, base_ok & np.isfinite(R_null) & (R_null > 0), edges))
    return ({k: np.concatenate(v) for k, v in obs_bins.items()},
            {k: np.concatenate(v) for k, v in null_bins.items()})


def spread_table(obs, null, edges, n_rep, min_unique):
    """Per-bin spread stats, keeping only bins with >= ``min_unique`` UNIQUE
    pairs.  ``obs``/``null`` are pooled over ``n_rep`` thinning replicates, so the
    mean unique-pair occupancy of a bin is len(o)/n_rep (each pair lands in one
    bin per replicate); we threshold on that, not the replicate-inflated len(o)."""
    rows = []
    for k in range(len(edges) - 1):
        if k not in obs or k not in null:
            continue
        o, n = obs[k], null[k]
        n_unique = len(o) / max(n_rep, 1)
        if n_unique < min_unique:
            continue
        xc = np.sqrt(edges[k] * edges[k + 1])
        rows.append(dict(
            bin=k, dS_center=xc, n_obs=len(o), n_unique=n_unique,
            sd_obs=o.std(), sd_null=n.std(),
            iqr_obs=np.subtract(*np.percentile(o, [75, 25])),
            iqr_null=np.subtract(*np.percentile(n, [75, 25])),
            p16_obs=np.percentile(o, 16), p84_obs=np.percentile(o, 84),
            p16_null=np.percentile(n, 16), p84_null=np.percentile(n, 84),
            p2_obs=np.percentile(o, 2.5), p97_obs=np.percentile(o, 97.5),
            p2_null=np.percentile(n, 2.5), p97_null=np.percentile(n, 97.5),
        ))
    t = pd.DataFrame(rows)
    t["inflation"] = t.sd_obs / t.sd_null               # SD ratio
    t["excess_var"] = t.sd_obs**2 - t.sd_null**2         # extra log10-variance
    t["excess_sd"] = np.sqrt(np.clip(t.excess_var, 0, None))
    return t


# --------------------------------------------------------------------------- #
#  Figures
# --------------------------------------------------------------------------- #
def fig_overlay(c, edges, sbymu, tab):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = default_rng(7)
    dS_A, R, ok = pair_xy(c.kN.to_numpy(), c.LN.to_numpy(), c.kS.to_numpy(), c.LS.to_numpy(), rng)
    grid = np.geomspace(1e-6, 1e-1, 400)

    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=200)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.axvspan(DETECTION, 1e-1, color="0.92", zorder=0)
    ax.scatter(dS_A[ok], R[ok], s=5, alpha=0.10, color="#AECDE1", linewidths=0,
               rasterized=True, zorder=1, label="Per-pair dN/dS (data)")
    ax.plot(grid, dNdS_purify_curve(grid, FD, sbymu), lw=2, color="tab:orange",
            zorder=6, label=fr"Purifying fit ($s/\mu={sbymu:.2g}$)")

    xc = tab.dS_center.to_numpy()
    # null 2.5-97.5 band (what the naive model predicts)
    ax.fill_between(xc, 10**tab.p2_null, 10**tab.p97_null, color="tab:orange",
                    alpha=0.18, zorder=2,
                    label="Theory 95% spread (shared thinning + Poisson numerator)")
    # observed 2.5-97.5 envelope
    ax.plot(xc, 10**tab.p2_obs, color="#0072B2", lw=1.3, ls="--", zorder=5)
    ax.plot(xc, 10**tab.p97_obs, color="#0072B2", lw=1.3, ls="--", zorder=5,
            label="Observed 95% spread")

    ax.axhline(1, color="0.5", lw=0.8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1e-6, 1e-1); ax.set_ylim(8e-2, 1.5e1)
    ax.set_xlabel("$dS$ (clonal region)"); ax.set_ylabel("$dN/dS$ (clonal region)")
    ax.set_title("Observed spread vs theory (shared thinning) around the purifying fit")
    ax.legend(fontsize=7, loc="lower left", framealpha=0.9)
    fig.tight_layout()
    out = FIGDIR / "purify_spread_null_overlay.pdf"
    fig.savefig(out, bbox_inches="tight"); fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    return out


def fig_per_bin(tab):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=200)
    x = tab.dS_center
    a1.plot(x, tab.sd_obs, "-o", color="#0072B2", label="Observed")
    a1.plot(x, tab.sd_null, "-o", color="tab:orange",
            label="Theory (shared thinning + Poisson numerator)")
    a1.plot(x, tab.excess_sd, "-^", color="#666", ms=4, label="Excess (quadrature)")
    a1.set_xscale("log"); a1.set_xlabel("$dS$ (clonal region)")
    a1.set_ylabel(r"SD of $\log_{10}(dN/dS)$ within bin")
    a1.axvspan(DETECTION, 1e-1, color="0.92", zorder=0)
    a1.set_title("Conditional spread per dS bin"); a1.legend(fontsize=8)
    for s in ("top", "right"):
        a1.spines[s].set_visible(False)

    a2.axhline(1, color="0.5", lw=0.8)
    a2.plot(x, tab.inflation, "-o", color="#9C27B0")
    a2.set_xscale("log"); a2.set_xlabel("$dS$ (clonal region)")
    a2.set_ylabel(r"SD inflation  $\mathrm{SD}_\mathrm{obs}/\mathrm{SD}_\mathrm{null}$")
    a2.axvspan(DETECTION, 1e-1, color="0.92", zorder=0)
    a2.set_title("Overdispersion factor (1 = shared thinning + Poisson numerator)")
    for s in ("top", "right"):
        a2.spines[s].set_visible(False)
    fig.tight_layout()
    out = FIGDIR / "purify_spread_per_bin.pdf"
    fig.savefig(out, bbox_inches="tight"); fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    return out


def within_species(c, edges, sbymu, min_pairs=120):
    """Per-species pooled overdispersion: weighted-mean SD inflation across that
    species' dS bins (weighted by observed pair-count per bin)."""
    rows = []
    counts = c.species.value_counts()
    for sp in counts[counts >= min_pairs].index:
        cs = c[c.species == sp]
        n_rep = max(N_SIM, 200)
        obs, nul = paired_spread(cs, edges, sbymu, n_rep=n_rep, seed=11)
        tab = spread_table(obs, nul, edges, n_rep=n_rep, min_unique=MIN_UNIQUE_SPECIES)
        if tab.empty:
            continue
        w = tab.n_obs.to_numpy(float)
        infl = np.sqrt(np.average(tab.sd_obs**2, weights=w) /
                       np.average(tab.sd_null**2, weights=w))
        rows.append(dict(species=sp, n_pairs=int(len(cs)), n_bins=len(tab),
                         inflation=infl,
                         sd_obs=np.sqrt(np.average(tab.sd_obs**2, weights=w)),
                         sd_null=np.sqrt(np.average(tab.sd_null**2, weights=w))))
    return pd.DataFrame(rows).sort_values("inflation", ascending=False).reset_index(drop=True)


def fig_within_species(ws):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, max(3, 0.32 * len(ws))), dpi=200)
    y = np.arange(len(ws))[::-1]
    ax.barh(y, ws.inflation, color="#0072B2", alpha=0.85)
    ax.axvline(1, color="0.4", lw=1, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([s.replace("_", " ") for s in ws.species], fontsize=7)
    ax.set_xlabel(r"Within-species SD inflation  $\mathrm{SD}_\mathrm{obs}/\mathrm{SD}_\mathrm{null}$")
    ax.set_title("Overdispersion of clonal dN/dS beyond theory (shared thinning), per species")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    out = FIGDIR / "purify_spread_within_species.pdf"
    fig.savefig(out, bbox_inches="tight"); fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    c = load_counts()
    rng = default_rng(42)
    sbymu, edges = fit_sbymu(c, rng)
    print(f"loaded {len(c)} pairs, {c.species.nunique()} species; fitted s/mu = {sbymu:.3g}")

    # median per-pair sampling intuition
    med_kS = np.median(c.kS); med_kN = np.median(c.kN)
    print(f"median kS={med_kS:.0f} (=> kS_B~{med_kS/2:.0f}), median kN={med_kN:.0f}")
    cv = np.sqrt(1/max(med_kN, 1) + 1/max(med_kS/2, 1))
    print(f"=> Poisson CV of dN/dS at the median pair ~ {cv:.2f}  "
          f"(~{cv/np.log(10):.2f} dex)")

    print("\nComputing observed & null spreads (pooled, shared thinning) ...")
    obs, nul = paired_spread(c, edges, sbymu, n_rep=N_SIM, seed=0)
    tab = spread_table(obs, nul, edges, n_rep=N_SIM, min_unique=MIN_UNIQUE_OVERLAY)
    cols = ["dS_center", "n_unique", "sd_obs", "sd_null", "inflation", "excess_sd"]
    print(tab[cols].to_string(index=False,
          formatters={"dS_center": "{:.1e}".format, "n_unique": "{:.0f}".format,
                      "sd_obs": "{:.3f}".format, "sd_null": "{:.3f}".format,
                      "inflation": "{:.2f}".format, "excess_sd": "{:.3f}".format}))

    # pooled headline: variance-weighted by observed pairs/bin
    w = tab.n_obs.to_numpy(float)
    pooled_obs = np.sqrt(np.average(tab.sd_obs**2, weights=w))
    pooled_null = np.sqrt(np.average(tab.sd_null**2, weights=w))
    print(f"\nPOOLED (pair-weighted across bins): SD_obs={pooled_obs:.3f} dex, "
          f"SD_theory={pooled_null:.3f} dex, inflation={pooled_obs/pooled_null:.2f}, "
          f"log-var from shared processing + Poisson numerator = "
          f"{(pooled_null**2/pooled_obs**2)*100:.0f}%, "
          f"excess (numerator overdispersion) = {(1-pooled_null**2/pooled_obs**2)*100:.0f}%")

    o1 = fig_overlay(c, edges, sbymu, tab)
    o2 = fig_per_bin(tab)
    print(f"wrote {o1}\nwrote {o2}")

    print("\nWithin-species overdispersion ...")
    ws = within_species(c, edges, sbymu)
    print(ws.to_string(index=False,
          formatters={"inflation": "{:.2f}".format, "sd_obs": "{:.3f}".format,
                      "sd_null": "{:.3f}".format}))
    o3 = fig_within_species(ws)
    print(f"wrote {o3}")

    tab.to_csv(REPO_ROOT / "outputs" / "purify_spread_pooled.csv", index=False)
    ws.to_csv(REPO_ROOT / "outputs" / "purify_spread_within_species.csv", index=False)


if __name__ == "__main__":
    main()
