"""Cross-species typical-pair dN/dS vs dS (reviewer trend: diverse species -> lower dN/dS).

A reviewer noted "a downward trend in mean dN/dS across species, with more diverse
species having lower mean dN/dS." This script builds the figure to examine that
trend properly: one averaged (all-1D) dN/dS-vs-dS point per species from TYPICAL
(fully recombined / unrelated) QP pairs, with the two correlation-artifact controls
that make such a plot honest:

  1. POISSON THINNING of the synonymous counts. dN/dS has dS in its denominator and
     dS is the x-axis, so estimating both from the same synonymous sites manufactures
     a spurious negative correlation from sampling noise alone. We split the
     synonymous SNVs into an x-axis half (A) and an independent denominator half (B),
     so the plotted correlation is real, not shared-denominator noise.

  2. The P. VULGATUS within- vs cross-clade control. dN/dS declines with divergence
     WITHIN every species (time-dependence of purifying selection; the premise of the
     Fig 3/4 decay curves). "Typical" pairs sit at different divergences across
     species, so a cross-species mean-dN/dS-vs-dS trend partly just re-traces that
     within-species decay. The QP "B. vulgatus" catalog co-bins P. dorei, giving a
     single "species" sampled across a wide dS range (within-clade ~0.6% -> cross-clade
     vulgatus x dorei ~6%). If that single-species trajectory has the same downward
     slope as the cross-species cloud, the cross-species trend is largely the
     time-dependence effect rather than 22 independent species-level differences.

Stage 1 (sampling, cached) writes per-species sampled-pair tables to
  outputs/typical_dnds_across_species/pairs/<species>.csv
Stage 2 (aggregation + figure) writes
  outputs/typical_dnds_across_species/species_summary.csv
  figures/typical_dnds_vs_ds_across_species.{png,pdf}

Run with the dNdS_310 env. Needs /Volumes/Botein mounted (QP catalogs +
pairwise identical-block-fraction matrices). Re-run is cheap (per-species cache);
pass --resample to force re-sampling.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/Users/Device6/Documents/Research/bgoodlab/dNdS/dNdS_dynamics_revision")
sys.path.insert(0, str(REPO))
os.environ.setdefault("MPLCONFIGDIR", str(REPO / ".cache" / "matplotlib"))

# QP SNVHelper + fully-recombined-pair machinery, now from the package.
from dnds_dynamics import config as dyn_config  # noqa: E402
from dnds_dynamics.snv_helpers.qp import load_qp_snv_helper  # noqa: E402

BLACKLIST = {"Lachnospiraceae_bacterium_51870"}
VULGATUS = "Bacteroides_vulgatus_57955"
CLADE_CSV = REPO / "outputs" / "pvulgatus_vs_qp_dnds" / "qp_sample_div_to_ref.csv"

OUT_DIR = REPO / "outputs" / "typical_dnds_across_species"
PAIRS_DIR = OUT_DIR / "pairs"
SUMMARY_CSV = OUT_DIR / "species_summary.csv"
FIG = REPO / "figures" / "typical_dnds_vs_ds_across_species"

N_PAIRS = 100                  # target fully-recombined pairs per species/category
RECOMB_THRESHOLD = dyn_config.fully_recombined_threshold  # identical-block frac <= this
RNG_SEED = 0


# ---------------------------------------------------------------------------
# Stage 1: sample fully-recombined pairs and tabulate per-pair 4D / 1D counts
# ---------------------------------------------------------------------------
def _pair_counts(h, s1, s2) -> dict:
    """Per-pair 4D (synonymous) and 1D (all nonsynonymous) diff/length counts."""
    diff = h.compute_pairwise_snvs(s1, s2)
    cov = h.compute_pairwise_coverage(s1, s2)
    return {
        "sample 1": s1, "sample 2": s2,
        "core_len_4D": int((cov & h.core_4D).sum()),
        "core_diff_4D": int((diff & h.snv_4D).sum()),
        "core_len_1D": int((cov & h.core_1D).sum()),
        "core_diff_1D": int((diff & h.snv_1D).sum()),
    }


def _fully_recombined_pairs(h) -> list[tuple[str, str]]:
    """All sample pairs with identical-block fraction <= RECOMB_THRESHOLD."""
    if not hasattr(h, "identical_block_frac"):
        h.load_identical_block()
    M = h.identical_block_frac
    vals = M.to_numpy(float)
    samples = list(M.index)
    iu, ju = np.triu_indices(len(samples), k=1)
    ok = np.isfinite(vals[iu, ju]) & (vals[iu, ju] <= RECOMB_THRESHOLD)
    return [(str(samples[i]), str(samples[j])) for i, j in zip(iu[ok], ju[ok])]


def _clade_map() -> dict[str, str]:
    cm = pd.read_csv(CLADE_CSV, dtype={"sample": str})
    return dict(zip(cm["sample"].astype(str), cm["clade"].astype(str)))


def _sample_pairs(pairs: list[tuple[str, str]], n: int, rng) -> list[tuple[str, str]]:
    if len(pairs) <= n:
        return pairs
    idx = rng.choice(len(pairs), size=n, replace=False)
    return [pairs[i] for i in idx]


def sample_species(species: str, rng) -> pd.DataFrame | None:
    """Sample up to N_PAIRS fully-recombined pairs; tabulate counts.

    For P. vulgatus, classify each pair by clade and tag category
    'within_clade' (generic point) vs 'cross_clade' (vulgatus x dorei), sampling
    up to N_PAIRS of each. Other species get category 'typical'.
    """
    h = load_qp_snv_helper(species, compute_bi_snvs=False, annotate=True)
    try:
        pairs = _fully_recombined_pairs(h)
    except Exception as e:  # noqa: BLE001
        print(f"  [skip] {species}: no identical-block matrix ({e})")
        return None
    if not pairs:
        print(f"  [skip] {species}: no fully-recombined pairs")
        return None

    rows: list[dict] = []
    if species == VULGATUS and CLADE_CSV.exists():
        cm = _clade_map()
        within, cross = [], []
        for a, b in pairs:
            ca, cb = cm.get(a), cm.get(b)
            if ca is None or cb is None:
                continue
            (within if ca == cb else cross).append((a, b))
        for cat, plist in (("within_clade", within), ("cross_clade", cross)):
            for a, b in _sample_pairs(plist, N_PAIRS, rng):
                rows.append({"category": cat, **_pair_counts(h, a, b)})
        print(f"  {species}: within={len(within)} cross={len(cross)} "
              f"(sampled <= {N_PAIRS} each)")
    else:
        for a, b in _sample_pairs(pairs, N_PAIRS, rng):
            rows.append({"category": "typical", **_pair_counts(h, a, b)})
        print(f"  {species}: {len(pairs)} fully-recombined pairs (sampled {len(rows)})")

    df = pd.DataFrame(rows)
    df.insert(0, "species_name", species)
    return df


def stage_sample(resample: bool) -> None:
    PAIRS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)
    species_list = sorted(p.name for p in dyn_config.snv_data_path.glob("*")
                          if not p.name.startswith(".") and p.name not in BLACKLIST)
    print(f"{len(species_list)} species in {dyn_config.snv_data_path}")
    for sp in species_list:
        out = PAIRS_DIR / f"{sp}.csv"
        if out.exists() and not resample:
            print(f"  [cached] {sp}")
            continue
        df = sample_species(sp, rng)
        if df is not None and len(df):
            df.to_csv(out, index=False)


# ---------------------------------------------------------------------------
# Stage 2: thinned pooled aggregation + bootstrap CIs
# ---------------------------------------------------------------------------
def thinned_point(df: pd.DataFrame, p_thin: float = 0.5, n_thin: int = 40,
                  n_boot: int = 600, seed: int = 42) -> dict | None:
    """Pooled all-1D dN/dS vs dS with Poisson-thinned synonymous split + CIs.

    x-axis dS estimated from synonymous half A; dN/dS denominator from the
    independent half B. Central estimate averages totals over ``n_thin``
    thinnings; CIs from ``n_boot`` replicates that resample pairs AND re-thin.
    """
    kS = df["core_diff_4D"].to_numpy(float)
    LS = df["core_len_4D"].to_numpy(float)
    kN = df["core_diff_1D"].to_numpy(float)
    LN = df["core_len_1D"].to_numpy(float)
    if len(df) < 2 or LS.sum() <= 0 or LN.sum() <= 0:
        return None
    rng = np.random.default_rng(seed)

    def _one(kSi, LSi, kNi, LNi):
        kSA = rng.binomial(kSi.astype(np.int64), p_thin).astype(float)
        kSB = kSi - kSA
        LSA, LSB = p_thin * LSi, (1 - p_thin) * LSi
        dS_x = kSA.sum() / max(LSA.sum(), 1e-12)
        denom = (kSB.sum() / max(LSB.sum(), 1e-12))
        if denom <= 0:
            return np.nan, np.nan
        R = (kNi.sum() / max(LNi.sum(), 1e-12)) / denom
        return dS_x, R

    ds_acc, r_acc, m = 0.0, 0.0, 0
    for _ in range(n_thin):
        d, r = _one(kS, LS, kN, LN)
        if np.isfinite(d) and np.isfinite(r):
            ds_acc += d; r_acc += r; m += 1
    if m == 0:
        return None
    dS_hat, R_hat = ds_acc / m, r_acc / m

    n = len(df)
    bd, br = np.full(n_boot, np.nan), np.full(n_boot, np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        bd[b], br[b] = _one(kS[idx], LS[idx], kN[idx], LN[idx])
    return {
        "n_pairs": int(n),
        "dS": float(dS_hat),
        "dS_lo": float(np.nanpercentile(bd, 2.5)),
        "dS_hi": float(np.nanpercentile(bd, 97.5)),
        "dNdS": float(R_hat),
        "dNdS_lo": float(np.nanpercentile(br, 2.5)),
        "dNdS_hi": float(np.nanpercentile(br, 97.5)),
    }


def _interpair_spread(sub: pd.DataFrame) -> dict:
    """16-84 percentile spread of per-pair dS and dN/dS (descriptive variation)."""
    dS = sub["core_diff_4D"].to_numpy(float) / sub["core_len_4D"].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        dNdS = (sub["core_diff_1D"].to_numpy(float) / sub["core_len_1D"].to_numpy(float)) / dS
    m = np.isfinite(dS) & np.isfinite(dNdS) & (dS > 0) & (dNdS > 0)
    dS, dNdS = dS[m], dNdS[m]
    return {
        "dS_p16": float(np.percentile(dS, 16)), "dS_p84": float(np.percentile(dS, 84)),
        "dNdS_p16": float(np.percentile(dNdS, 16)), "dNdS_p84": float(np.percentile(dNdS, 84)),
    }


def stage_aggregate() -> pd.DataFrame:
    rows = []
    for f in sorted(PAIRS_DIR.glob("*.csv")):
        df = pd.read_csv(f)
        sp = df["species_name"].iloc[0]
        for cat, sub in df.groupby("category"):
            pt = thinned_point(sub)
            if pt is None:
                continue
            label = sp if cat == "typical" else f"{sp} [{cat.replace('_clade','')}]"
            rows.append({"species_name": sp, "category": cat, "label": label,
                         **pt, **_interpair_spread(sub)})
    summary = pd.DataFrame(rows).sort_values("dS").reset_index(drop=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_CSV, index=False)
    return summary


# ---------------------------------------------------------------------------
# Stage 3: figure + correlation
# ---------------------------------------------------------------------------
def _draw_spread(ax, g: pd.DataFrame, ecolor: str, zorder: int) -> None:
    """16-84 inter-pair spread as explicit span lines (hlines/vlines).

    Spans are drawn between the true percentiles, so the marker (placed at the
    pooled aggregate value) may sit anywhere on the span -- including its edge for
    right-skewed species -- without the one-sided clipping that an asymmetric
    errorbar would produce when the aggregate falls outside [p16, p84].
    """
    ax.hlines(g["dNdS"], g["dS_p16"], g["dS_p84"], color=ecolor, lw=0.8, alpha=0.9, zorder=zorder)
    ax.vlines(g["dS"], g["dNdS_p16"], g["dNdS_p84"], color=ecolor, lw=0.8, alpha=0.9, zorder=zorder)


def make_figure(summary: pd.DataFrame, logscale: bool, out: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr, pearsonr

    out.parent.mkdir(parents=True, exist_ok=True)

    # Generic cross-species cloud: 'typical' species + vulgatus within-clade point.
    generic = summary[(summary["category"] == "typical")
                      | (summary["category"] == "within_clade")].copy()
    vul_within = summary[(summary["species_name"] == VULGATUS)
                         & (summary["category"] == "within_clade")]
    vul_cross = summary[(summary["species_name"] == VULGATUS)
                        & (summary["category"] == "cross_clade")]

    x = generic["dS"].to_numpy(float)
    y = generic["dNdS"].to_numpy(float)
    ok = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    g = generic[ok].copy()
    x, y = x[ok], y[ok]
    rho, p_rho = spearmanr(x, y)
    r_log, p_log = pearsonr(np.log10(x), np.log10(y))
    r_lin, p_lin = pearsonr(x, y)

    fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=200)
    # blue cloud = all generic species EXCEPT the vulgatus within-clade point (which
    # is highlighted separately in orange below, though still counted in the stats).
    cloud = g[g["species_name"] != VULGATUS]
    _draw_spread(ax, cloud, "#9ecae1", zorder=2)
    ax.scatter(cloud["dS"], cloud["dNdS"], s=26, color="#0072B2", linewidths=0,
               alpha=0.9, zorder=3, label="species (typical pairs)")

    # vulgatus within-clade (control low-dS) and cross-clade (vulgatus x dorei) points,
    # not connected; both with their own inter-pair spread spans.
    if len(vul_cross):
        _draw_spread(ax, vul_cross, "#fdae6b", zorder=4)
        ax.scatter(vul_cross["dS"], vul_cross["dNdS"], s=70, marker="D",
                   color="#D55E00", linewidths=0, zorder=5,
                   label="P. vulgatus x dorei (cross-clade)")
    if len(vul_within):
        _draw_spread(ax, vul_within, "#fdae6b", zorder=4)
        ax.scatter(vul_within["dS"], vul_within["dNdS"], s=70, marker="D",
                   facecolor="none", edgecolor="#D55E00", linewidths=1.6, zorder=6,
                   label="P. vulgatus within-clade")

    # OLS trend line over the generic cloud (in the plotted space)
    if logscale:
        coef = np.polyfit(np.log10(x), np.log10(y), 1)
        xs = np.linspace(np.log10(x).min(), np.log10(x).max(), 100)
        ax.plot(10 ** xs, 10 ** np.polyval(coef, xs), "k--", lw=1.0, alpha=0.7,
                label=f"log-log fit (slope {coef[0]:.2f})")
    else:
        coef = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, np.polyval(coef, xs), "k--", lw=1.0, alpha=0.7,
                label=f"linear fit (slope {coef[0]:.2f})")

    if logscale:
        ax.axhline(1, ls=":", lw=0.6, color="grey")
        ax.set_xscale("log"); ax.set_yscale("log")
    else:
        # data-driven y range so the trend is visible (the y=1 neutral line would
        # otherwise compress everything into the bottom fifth of the panel).
        ymax = float(np.nanmax(np.r_[y, generic["dNdS_p84"].to_numpy(float),
                                     vul_cross["dNdS_p84"].to_numpy(float)
                                     if len(vul_cross) else [0.0]]))
        ax.set_ylim(0, 1.12 * ymax)
    ax.set_xlabel(r"typical-pair synonymous divergence $dS$")
    ax.set_ylabel(r"typical-pair $dN/dS$")
    ax.set_title(
        f"Cross-species typical dN/dS vs dS  ({'log-log' if logscale else 'linear'})\n"
        f"Spearman $\\rho$={rho:.2f} (p={p_rho:.1g});  "
        + (f"log-log Pearson r={r_log:.2f}" if logscale else f"linear Pearson r={r_lin:.2f}")
        + "    (bars: 16-84% inter-pair spread)", fontsize=9.5)
    ax.legend(loc="best", fontsize=7, framealpha=0.9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.with_suffix('.png')}")
    return {"rho": rho, "p_rho": p_rho, "r_log": r_log, "r_lin": r_lin, "slope": coef[0]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["sample", "aggregate", "all"], default="all")
    ap.add_argument("--resample", action="store_true",
                    help="force re-sampling even if per-species cache exists")
    args = ap.parse_args()

    if args.stage in ("sample", "all"):
        stage_sample(args.resample)
    if args.stage in ("aggregate", "all"):
        summary = stage_aggregate()
        print("\n=== species summary (sorted by dS) ===")
        print(summary[["label", "n_pairs", "dS", "dNdS"]].round(5).to_string(index=False))
        s_log = make_figure(summary, logscale=True, out=FIG)
        s_lin = make_figure(summary, logscale=False, out=FIG.with_name(FIG.name + "_linear"))
        print(f"\nSpearman rho={s_log['rho']:.3f} (p={s_log['p_rho']:.2g}); "
              f"log-log Pearson r={s_log['r_log']:.3f}, slope={s_log['slope']:.3f}; "
              f"linear Pearson r={s_lin['r_lin']:.3f}")
        print(f"wrote {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
