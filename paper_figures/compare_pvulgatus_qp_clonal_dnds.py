"""Compare isolate P. vulgatus vs QP B. vulgatus dN/dS dynamics.

Same species, same MIDAS reference (Bacteroides_vulgatus_57955), two datasets:
  - QP: LiuGood2024 metagenome quasi-phased SNVs (published close/clonal CSVs).
  - ISO: 369 NCBI isolates, new CP-HMM recombination cache (P. dorei dropped).

Two comparisons, overlaid QP vs ISO:
  1. Stratification: per-pair dN/dS vs dS for full-core / recombined / clonal.
  2. Clonal dN/dS vs clonal dS decay, split by missense and nonsense, using the
     published per_species_dnds Poisson-thinning + adaptive binning + bootstrap.

Outputs:
  figures/pvulgatus_vs_qp_dnds_comparison.{png,pdf}
  outputs/pvulgatus_vs_qp_dnds/{stratification_summary.csv, clonal_decay_binned.csv}
"""
from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

REPO = Path("/Users/Device6/Documents/Research/bgoodlab/dNdS/dNdS_dynamics_revision")
sys.path.insert(0, str(REPO))
os.environ.setdefault("MPLCONFIGDIR", str(REPO / ".cache" / "matplotlib"))

from dnds_dynamics.dnds import stratified as sd  # noqa: E402
from dnds_dynamics.figures import per_species_dnds as ps  # noqa: E402
from dnds_dynamics.figures.theory import dNdS_purify_curve  # noqa: E402
# Adaptive per-species binning used by the published clonal-dN/dS species grid
# (clonal_dNdS_species_grid_all.pdf): geometric log-spaced bins, ~target pairs
# per bin, with bootstrap ribbons. Pools the discrete low-dS bands (few-4D-SNV
# pairs, where dS_A is quantized) into continuous pooled-ratio bins.
from dnds_dynamics.figures.per_species_dnds import (  # noqa: E402
    ThinSettings as GridThinSettings,
    ps_bootstrap_binned,
    ps_make_bins_auto,
    ps_aggregate_bins,
)

SP = "Bacteroides_vulgatus_57955"
QP_BASE = REPO / "data"
# Isolate dN/dS dir. Defaults to the new single-clade iterative qpcore CP-HMM run
# (P. dorei dropped); override with ISO_DNDS_DIR (e.g. the old
# pvulgatus_isolate_cphmm run, or the qpcore_extended tract-extension run).
ISO_DNDS = Path(os.environ.get(
    "ISO_DNDS_DIR",
    REPO / "outputs" / "pvulgatus_isolate_cphmm_qpcore" / "snv_table" / "dnds",
))
ISO_IDENT_FRAC = Path("/Volumes/Botein/ncbi_isolates/Phocaeicola_vulgatus/snv_table/identical_fraction.parquet")
# Isolate dN/dS under the 1D-EXTENDED recombination mask (same pairs, more of the
# genome reclassified clonal->recombined). Overlaid on the clonal-decay row.
ISO_DNDS_EXT = Path(os.environ.get(
    "ISO_DNDS_EXT_DIR",
    REPO / "outputs" / "pvulgatus_isolate_cphmm_qpcore_extended" / "snv_table" / "dnds",
))
# Harmonize the isolate close-pair selection to QP's published cutoff.
# NB: microbiome_evolution's "clonal_fraction_cutoff" is computed by
# close_pair_utils.compute_clonal_fraction = fraction of genome BLOCKS with zero
# SNPs, i.e. the *identical-block fraction* (genome blocks) -- the SAME quantity
# as ISO_IDENT_FRAC here. (It is NOT the CP-HMM inferred clonal_fraction =
# 1 - transfer_len/genome_len, which the cache also stores; the two correlate at
# r~0.98 but differ.) The isolate CP-HMM ran on identical_fraction > 0.5; this
# 0.75 is an additional comparison-time filter to match the QP cutoff.
# Override with ISO_IF_CUTOFF=none (all analyzed pairs, i.e. IF>0.5) for an
# all-pairs version; CMP_SUFFIX appends to the figure/output names.
_IFC_ENV = os.environ.get("ISO_IF_CUTOFF", "0.75").strip().lower()
ISO_IF_CUTOFF = None if _IFC_ENV in ("none", "all", "") else float(_IFC_ENV)
# Optional INFERRED clonal-fraction filter for the isolates (cache field
# clonal_fraction = 1 - recombined_len/genome_len, the stage2-3 quantity;
# distinct from the block identical_fraction). Set ISO_CF_CUTOFF to drop the more
# recombined / diverged isolate pairs by inferred clonal fraction instead of (or
# in addition to) the identical-fraction filter.
_CFC_ENV = os.environ.get("ISO_CF_CUTOFF", "none").strip().lower()
ISO_CF_CUTOFF = None if _CFC_ENV in ("none", "off", "") else float(_CFC_ENV)
_CMP_SUFFIX = os.environ.get("CMP_SUFFIX", "")
OUT_DIR = REPO / "outputs" / f"pvulgatus_vs_qp_dnds{_CMP_SUFFIX}"
FIG = REPO / "figures" / f"pvulgatus_vs_qp_dnds_comparison{_CMP_SUFFIX}"
# Clade-membership cache always lives in the canonical (un-suffixed) dir.
_CLADE_DIR = REPO / "outputs" / "pvulgatus_vs_qp_dnds"
# Typical / unrelated-pair tables (fully recombined pairs) for the species-mean
# anchor point shown in the published clonal-dN/dS grid panels.
QP_TYPICAL = QP_BASE / "gut_microbiome_typical_pair_dNdS" / f"{SP}.csv"
ISO_TYPICAL = ISO_DNDS / "typical_pairs_fully_recombined.csv"

# High-contrast, colorblind-safe (Wong): blue vs vermillion, plus distinct markers.
QP_COLOR = "#0072B2"
ISO_COLOR = "#D55E00"
ISO_EXT_COLOR = "#009E73"   # bluish green = ISO under the 1D-extended mask
QP_MARKER = "o"
ISO_MARKER = "^"
ISO_EXT_MARKER = "s"
STRATA = ["full_core", "recombined", "clonal"]
STRATUM_TITLE = {"full_core": "Full (core) genome", "recombined": "Recombined regions",
                 "clonal": "Clonal regions"}
CLASSES = ["all", "missense", "nonsense"]
CLASS_TITLE = {"all": "Clonal dN/dS (all 1D)", "missense": "Clonal missense dN/dS",
               "nonsense": "Clonal nonsense dN/dS"}

# Main-text fit parameters (from paper_figures/clonal_dNdS_dynamics.py; the aggregate
# fit shown in Fig 3 and the per-class fits in Fig 4). FIXED constants -- nothing is
# refit on the QP/ISO data here.
#   all      = Fig 3 aggregate fit (neutral + one deleterious class)
#   missense = Fig 4 missense fit  (neutral + one deleterious class)
#   nonsense = Fig 4 nonsense fit  (neutral + weak + strong class)
MAINTEXT_THEORY = {
    "all":      ("one_class", dict(fd=0.90, sbymu=5.4e3)),
    "missense": ("one_class", dict(fd=0.8870, sbymu=4.79e3)),
    "nonsense": ("two_class", dict(fd=0.9916, alpha2=0.328, s1mu=2.33e4, s2mu=1e7)),
}
THEORY_LABEL = {
    "all": "Fig 3 2-class fit",
    "missense": "Fig 4 2-class fit",
    "nonsense": "Fig 4 3-class fit",
}


def _add_clonal_from_core(df: pd.DataFrame) -> pd.DataFrame:
    for s in ["diff_4D", "len_4D", "diff_1D", "len_1D"]:
        col = f"clonal_{s}"
        if col not in df.columns or df[col].isna().all():
            df[col] = df[f"core_{s}"]
    return df


def _add_ids(df: pd.DataFrame) -> pd.DataFrame:
    for c in sd.ID_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    return df


QP_CLADE_CSV = _CLADE_DIR / "qp_sample_div_to_ref.csv"


def qp_clade_map() -> "dict[str, str] | None":
    """Sample -> clade ('vulgatus'/'dorei') map for the QP B. vulgatus catalog.

    Read from the cached ``QP_CLADE_CSV`` (each QP sample's 4D divergence to the
    reference, thresholded into clades). The QP catalog co-bins P. dorei: samples
    split cleanly into a vulgatus mode (~0.6% 4D divergence from the reference) and
    a dorei mode (~6%). We use this to
    drop *cross-clade* (vulgatus-dorei) pairs, which are spuriously divergent;
    within-clade pairs of EITHER clade are kept (preserving data — a fairer
    like-for-like comparison without discarding the dorei-dorei pairs). Returns
    None if the membership file is absent (no filtering, with a warning)."""
    if not QP_CLADE_CSV.exists():
        print(f"[warn] {QP_CLADE_CSV} not found; QP NOT clade-filtered. "
              f"Regenerate by computing each QP sample's 4D divergence to the "
              f"reference and splitting at the vulgatus/dorei modes.")
        return None
    cm = pd.read_csv(QP_CLADE_CSV, dtype={"sample": str})
    return dict(zip(cm["sample"].astype(str), cm["clade"].astype(str)))


def _filter_pairs_within_clade(df: pd.DataFrame, clade_map: "dict[str, str] | None",
                               label: str) -> pd.DataFrame:
    """Keep only WITHIN-clade pairs (both samples same clade); drop cross-clade.

    Within-clade pairs of either vulgatus or dorei are retained; only spurious
    vulgatus-dorei pairs are removed. For the close/clonal tables this is a no-op
    (those pairs are already within-clade by construction), so all data is kept.
    """
    if clade_map is None:
        return df
    c1 = df["sample 1"].astype(str).map(clade_map)
    c2 = df["sample 2"].astype(str).map(clade_map)
    same = c1.notna() & c2.notna() & (c1 == c2)
    n_cross = int((c1.notna() & c2.notna() & (c1 != c2)).sum())
    print(f"QP {label}: within-clade filter {len(df)} -> {int(same.sum())} pairs "
          f"(dropped {n_cross} cross-clade)")
    return df[same].reset_index(drop=True)


def load_qp() -> pd.DataFrame:
    clade_map = qp_clade_map()
    close = pd.read_csv(QP_BASE / "gut_microbiome_close_pair_dNdS" / f"{SP}.csv")
    close["event_count"] = (close.get("recomb_len_4D", 0).fillna(0) > 0).astype(int)
    close = _filter_pairs_within_clade(close, clade_map, "close")
    clonal = _add_clonal_from_core(pd.read_csv(QP_BASE / "gut_microbiome_clonal_pair_dNdS" / f"{SP}.csv"))
    clonal["event_count"] = 0
    clonal = _filter_pairs_within_clade(clonal, clade_map, "clonal")
    return _add_ids(pd.concat([close, clonal], ignore_index=True))


def _attach_identical_fraction(df: pd.DataFrame) -> pd.DataFrame:
    ifr = pd.read_parquet(ISO_IDENT_FRAC)
    key = lambda a, b: tuple(sorted((str(a), str(b))))
    ifmap = {key(a, b): f for a, b, f in
             zip(ifr["sample_1"], ifr["sample_2"], ifr["identical_fraction"])}
    df = df.copy()
    df["identical_fraction"] = [ifmap.get(key(a, b), np.nan)
                                for a, b in zip(df["sample 1"], df["sample 2"])]
    return df


def load_iso_from(dnds_dir: Path, if_cutoff: float = ISO_IF_CUTOFF) -> pd.DataFrame:
    close = _attach_identical_fraction(pd.read_csv(dnds_dir / "close_pairs_with_recombination.csv"))
    clonal = _attach_identical_fraction(
        _add_clonal_from_core(pd.read_csv(dnds_dir / "clonal_pairs_no_recombination.csv")))
    df = _add_ids(pd.concat([close, clonal], ignore_index=True))
    if if_cutoff is not None:
        n0 = len(df)
        df = df[df["identical_fraction"] > float(if_cutoff)].reset_index(drop=True)
        print(f"ISO close-pair filter identical_fraction>{if_cutoff}: {n0} -> {len(df)} pairs")
    if ISO_CF_CUTOFF is not None:
        n0 = len(df)
        df = df[df["clonal_fraction"] > float(ISO_CF_CUTOFF)].reset_index(drop=True)
        print(f"ISO close-pair filter clonal_fraction(inferred)>{ISO_CF_CUTOFF}: {n0} -> {len(df)} pairs")
    return df


def load_iso(if_cutoff: float = ISO_IF_CUTOFF) -> pd.DataFrame:
    return load_iso_from(ISO_DNDS, if_cutoff)


# Settings mirror the published species-grid call (cell 35 of clonal_dNdS_dynamics):
# geometric auto bins, ~50 pairs/bin target, drop bins <20 pairs, 400 bootstraps.
GRID_BIN_SETTINGS = GridThinSettings(
    p_thin=0.5,
    rng_seed=42,
    auto_bins=True,
    target_pairs_per_bin=50,
    min_pairs_per_bin=20,
    min_bins=4,
    max_bins=20,
    min_bins_retained=3,
    binning_strategy="geom",
    B=400,
    min_pairs_per_species=10,
)

# Nonsense mutations are rare, so synonymous-balanced geom bins still hold very few
# nonsense *events* per bin (~20-70), making the bin-to-bin median jagged. Use
# coarser bins (more pairs/bin, higher retention floor) for nonsense so each bin
# carries more nonsense events and the trend is less Poisson-noisy. The CIs below
# already widen for sparse bins; this trades a little dS-resolution for stability.
BIN_SETTINGS_BY_CLASS = {
    "all": GRID_BIN_SETTINGS,
    "missense": GRID_BIN_SETTINGS,
    "nonsense": dataclasses.replace(
        GRID_BIN_SETTINGS, target_pairs_per_bin=100, min_pairs_per_bin=35,
        min_bins=3, max_bins=6, min_bins_retained=2,
    ),
}

# Per-class plotting limits, matching the published species-grid panels
# (clonal_dNdS_species_grid_{all,missense}.pdf use ylim (0.03, 20); the nonsense
# grid uses (1e-3, 20)). xlim matches the grids' (1e-6, 1e-1).
CLASS_XLIM = (1e-6, 1e-1)
CLASS_YLIM = {"all": (3e-2, 2e1), "missense": (3e-2, 2e1), "nonsense": (1e-3, 2e1)}


def fit_purify_sbymu(agg: pd.DataFrame, fd: float,
                     bounds: "tuple[float, float]" = (1e-3, 1e7),
                     min_pairs: int = 20) -> float:
    """WLS fit of the single-class purifying s/mu on a binned aggregate.

    Same WLS objective as the main clonal-dN/dS fit: minimize the
    inverse-variance-weighted mean squared log-residual between the binned
    ratio-of-totals ``R_hat`` and ``dNdS_purify_curve(dS_x, fd, s/mu)``. ``fd`` is
    fixed per class (from the high-dS / typical-pair equilibrium). Returns NaN if
    fewer than 3 informative bins.
    """
    if agg is None or len(agg) == 0:
        return float("nan")
    m = ((agg["n_pairs"] >= min_pairs) & (agg["KN"] > 0) & (agg["KS_B"] > 0)
         & (agg["dS_x"] > 0) & (agg["R_hat"] > 0) & np.isfinite(agg["R_hat"].to_numpy(float)))
    if int(m.sum()) < 3:
        return float("nan")
    x = agg.loc[m, "dS_x"].to_numpy(float)
    y = agg.loc[m, "R_hat"].to_numpy(float)
    KN = agg.loc[m, "KN"].to_numpy(float)
    KS = agg.loc[m, "KS_B"].to_numpy(float)
    w = 1.0 / (1.0 / np.maximum(KN, 1.0) + 1.0 / np.maximum(KS, 1.0))

    def objective(sbymu: float) -> float:
        pred = dNdS_purify_curve(x, fd, sbymu)
        return float(np.average((np.log(y) - np.log(pred)) ** 2, weights=w))

    return float(minimize_scalar(objective, bounds=bounds, method="bounded").x)


# --- 2-class ("three-class": neutral + weak + strong) purifying model, as in the
#     nonsense grid (clonal_dNdS_dynamics.py). ---
TWO_CLASS_S2_BYMU = 1e7      # fixed "very strong" class
TWO_CLASS_DS_THRESH = 3e-5   # early-regime cutoff used to estimate alpha2


def _F_small(x):
    x = np.asarray(x, dtype=float)
    return (1.0 - np.exp(-x)) / np.clip(x, 1e-30, None)


def dnds_two_class(dS, fd, alpha2, s1mu, s2mu=TWO_CLASS_S2_BYMU):
    """Neutral fraction (1-fd) + weak class (alpha1=fd-alpha2, s1mu) + strong (alpha2, s2mu)."""
    alpha2 = float(np.clip(alpha2, 0.0, fd))
    alpha1 = fd - alpha2
    return (1.0 - fd) + alpha1 * _F_small(0.5 * s1mu * np.asarray(dS, float)) \
        + alpha2 * _F_small(0.5 * s2mu * np.asarray(dS, float))


def estimate_alpha2_early(tb: pd.DataFrame, ds_thresh: float = TWO_CLASS_DS_THRESH) -> float:
    """alpha2 = 1 - (low-dS ratio-of-totals), over pairs with dS_A < ds_thresh."""
    m = (tb["dS_A"] > 0) & (tb["dS_A"] < ds_thresh) & (tb["kS_B"] > 0) & (tb["LN"] > 0)
    if not np.any(m):
        return float("nan")
    KN, LN = tb.loc[m, "kN"].sum(), tb.loc[m, "LN"].sum()
    KS, LS = tb.loc[m, "kS_B"].sum(), tb.loc[m, "LS_B"].sum()
    if LN <= 0 or LS <= 0 or KS <= 0:
        return float("nan")
    r_low = (KN / LN) / (KS / LS)   # LN already carries the 3-opp normalization
    return float(np.clip(1.0 - r_low, 0.0, 1.0))


def fit_two_class_s1(agg: pd.DataFrame, fd: float, alpha2: float,
                     bounds=(1e-3, 1e7), min_pairs: int = 20) -> float:
    """WLS fit of s1/mu in the 2-class model, given fixed fd and alpha2."""
    if agg is None or len(agg) == 0 or not (np.isfinite(fd) and np.isfinite(alpha2)):
        return float("nan")
    m = ((agg["n_pairs"] >= min_pairs) & (agg["KN"] > 0) & (agg["KS_B"] > 0)
         & (agg["dS_x"] > 0) & (agg["R_hat"] > 0) & np.isfinite(agg["R_hat"].to_numpy(float)))
    if int(m.sum()) < 3:
        return float("nan")
    x = agg.loc[m, "dS_x"].to_numpy(float)
    y = agg.loc[m, "R_hat"].to_numpy(float)
    KN = agg.loc[m, "KN"].to_numpy(float)
    KS = agg.loc[m, "KS_B"].to_numpy(float)
    w = 1.0 / (1.0 / np.maximum(KN, 1.0) + 1.0 / np.maximum(KS, 1.0))

    def objective(s1mu):
        pred = dnds_two_class(x, fd, alpha2, s1mu)
        return float(np.average((np.log(y) - np.log(pred)) ** 2, weights=w))

    return float(minimize_scalar(objective, bounds=bounds, method="bounded").x)


def within_clade_typical(typ_df: "pd.DataFrame | None", name: str = "",
                         min_gap_ratio: float = 2.5) -> "pd.DataFrame | None":
    """Drop cross-clade pairs from a typical-pair table via the bimodal dS gap.

    The QP "B. vulgatus" sample co-bins a second clade (P. dorei) onto the same
    MIDAS reference, so its typical (unrelated) pairs are bimodal in 4D dS: a
    within-clade mode (~0.01) and a much higher cross-clade vulgatus-dorei mode
    (~0.065). Sort the per-pair dS, find the largest consecutive ratio gap, and if
    it exceeds ``min_gap_ratio`` keep only the low (within-clade) mode. Unimodal
    tables (e.g. the ISO isolates, with dorei already dropped) pass through
    unchanged.
    """
    if typ_df is None or len(typ_df) < 4:
        return typ_df
    dS = (typ_df["core_diff_4D"] / typ_df["core_len_4D"]).to_numpy(float)
    order = np.argsort(dS)
    s = dS[order]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = s[1:] / np.clip(s[:-1], 1e-12, None)
    i = int(np.argmax(ratios))
    if ratios[i] >= min_gap_ratio:
        cutoff = s[i]
        keep = dS <= cutoff
        print(f"[{name} typical] bimodal 4D-dS gap {s[i]:.4f}->{s[i + 1]:.4f} "
              f"(x{ratios[i]:.1f}): dropping {int((~keep).sum())}/{len(dS)} cross-clade "
              f"pairs (dS>{cutoff:.4f}); keeping {int(keep.sum())} within-clade.")
        return typ_df.loc[keep].reset_index(drop=True)
    return typ_df


def typical_pair_point(typ_df: pd.DataFrame, cls: str,
                       p_thin: float = 0.5, seed: int = 42) -> "tuple[float, float] | None":
    """Species-mean dN/dS point from typical (fully recombined / unrelated) pairs.

    Mirrors the published grid's "Unrelated pairs (species mean)" cross: pool the
    typical pairs into one bin, Poisson-thin synonymous into an x-split (A) and an
    independent denominator (B), and return ``(dS_x, dN/dS)``. Class maps to the
    same opportunity convention as ``prepare_thin_base_for_classes`` (3 opps/site
    for missense/nonsense). Returns ``None`` if the table is empty/degenerate.
    """
    if typ_df is None or typ_df.empty:
        return None
    OPPS_PER_SITE = 3.0
    kS = typ_df["core_diff_4D"].to_numpy(float)
    LS = typ_df["core_len_4D"].to_numpy(float)
    if cls == "all":
        kN = typ_df["core_diff_1D"].to_numpy(float)
        LN = typ_df["core_len_1D"].to_numpy(float)
    elif cls == "missense":
        kN = typ_df["core_mut_m"].to_numpy(float)
        LN = typ_df["core_m"].to_numpy(float) / OPPS_PER_SITE
    elif cls == "nonsense":
        kN = typ_df["core_mut_n"].to_numpy(float)
        LN = typ_df["core_n"].to_numpy(float) / OPPS_PER_SITE
    else:
        raise ValueError(f"Unknown class {cls}")
    rng = np.random.default_rng(seed)
    kS_A = rng.binomial(np.clip(kS, 0, None).astype(int), p_thin)
    kS_B = kS - kS_A
    LS_A, LS_B = p_thin * LS, (1 - p_thin) * LS
    dS_x = kS_A.sum() / max(LS_A.sum(), 1e-12)
    denom = (kS_B.sum() / max(LS_B.sum(), 1e-12))
    if denom <= 0 or LN.sum() <= 0:
        return None
    R = (kN.sum() / LN.sum()) / denom
    if not (np.isfinite(dS_x) and dS_x > 0 and np.isfinite(R) and R > 0):
        return None
    return float(dS_x), float(R)


def adaptive_binned(tb: pd.DataFrame, settings: GridThinSettings = GRID_BIN_SETTINGS) -> pd.DataFrame:
    """Adaptive geometric binning + bootstrap ribbons (published-grid strategy).

    ``tb`` is a thin-base frame (already Poisson-thinned) carrying ``dS_A``,
    ``kS_A``/``kS_B``/``LS_A``/``LS_B`` and the class ``kN``/``LN`` columns. Bins
    are geometric in ``dS_A`` with ~``target_pairs_per_bin`` pairs each (so the
    quantized few-SNV bands below ~1e-4 pool together), and each bin's dN/dS is a
    pooled ratio-of-totals with a 95% bootstrap interval over resampled pairs.
    Returns columns ``dS_x``/``R_hat``/``n_pairs``/``lo``/``med``/``hi``.
    """
    agg0, boots = ps_bootstrap_binned(tb, kN_col="kN", LN_col="LN", settings=settings)
    if len(agg0) == 0:
        return pd.DataFrame()
    out = agg0[["bin", "dS_x", "R_hat", "KN", "KS_B", "n_pairs"]].copy()
    bmap = boots.set_index("bin")
    for col in ("lo", "med", "hi"):
        out[col] = out["bin"].map(bmap[col])
    return out.reset_index(drop=True)


def binned_in_edges(tb: pd.DataFrame, edges: "np.ndarray | None",
                    settings: GridThinSettings, n_boot: int = 400, seed: int = 42) -> pd.DataFrame:
    """Published binning (single thinning, pairs bootstrap) but in FIXED edges.

    Pools ``tb`` (already thinned) into the given ``edges`` via the published
    ``ps_aggregate_bins`` ratio-of-totals, then bootstraps over pairs (fixed edges)
    for the bin median/CI -- so any dataset can be shown in the QP bin grid with
    the same estimator as ``clonal_dNdS_species_grid``. Returns ``ps_aggregate_bins``
    columns plus ``med``/``lo``/``hi``."""
    if edges is None:
        return pd.DataFrame()
    nb = len(edges) - 1
    agg0 = ps_aggregate_bins(tb, edges, kN_col="kN", LN_col="LN",
                             min_pairs_per_bin=settings.min_pairs_per_bin)
    if len(agg0) == 0:
        return agg0
    rng = np.random.default_rng(seed)
    tbr = tb.reset_index(drop=True)
    n = len(tbr)
    M = np.full((n_boot, nb), np.nan)
    for b in range(n_boot):
        a = ps_aggregate_bins(tbr.iloc[rng.integers(0, n, n)], edges, kN_col="kN", LN_col="LN",
                              min_pairs_per_bin=settings.min_pairs_per_bin)
        if len(a):
            M[b, a["bin"].to_numpy(int)] = a["R_hat"].to_numpy(float)
    bins = agg0["bin"].to_numpy(int)
    agg0 = agg0.copy()
    agg0["lo"] = np.nanpercentile(M, 2.5, axis=0)[bins]
    agg0["med"] = np.nanpercentile(M, 50.0, axis=0)[bins]
    agg0["hi"] = np.nanpercentile(M, 97.5, axis=0)[bins]
    return agg0


def _auto_geom_edges(dS_full: np.ndarray, settings: GridThinSettings) -> "np.ndarray | None":
    """Geometric bin edges from the (un-thinned) dS distribution.

    Edges are placed in dS space (E[dS_A]=dS), so they are stable across thinning
    realizations -- unlike edges derived from a single random A/B split. Bin count
    follows the same auto rule as the published grid (~target pairs per bin)."""
    d = dS_full[(dS_full > 0) & np.isfinite(dS_full)]
    n = int(d.size)
    if n < settings.min_pairs_per_bin * 2:
        return None
    nb = max(settings.min_bins, min(settings.max_bins, n // max(1, settings.target_pairs_per_bin)))
    nb = max(2, int(nb))
    return np.geomspace(float(d.min()), float(d.max()), nb + 1)


def shared_geom_edges(tbs: "list[pd.DataFrame]", settings: GridThinSettings) -> "np.ndarray | None":
    """Common geometric bin edges across several datasets (for like-for-like bins).

    Edges span the pooled dS range of all ``tbs``; the bin count uses the auto rule
    on the SMALLER dataset's pair count, so neither dataset is over-binned (the
    sparser one keeps ~target pairs per bin and the denser one simply has more)."""
    dS_list = [np.divide(t["kS"].to_numpy(float), t["LS"].to_numpy(float),
                         out=np.zeros(len(t)), where=t["LS"].to_numpy(float) > 0)
               for t in tbs]
    pos = [d[(d > 0) & np.isfinite(d)] for d in dS_list]
    pos = [d for d in pos if d.size]
    if not pos:
        return None
    n_min = min(d.size for d in pos)
    if n_min < settings.min_pairs_per_bin * 2:
        return None
    nb = max(settings.min_bins, min(settings.max_bins, n_min // max(1, settings.target_pairs_per_bin)))
    nb = max(2, int(nb))
    all_pos = np.concatenate(pos)
    return np.geomspace(float(all_pos.min()), float(all_pos.max()), nb + 1)


def robust_binned(tb: pd.DataFrame, settings: GridThinSettings = GRID_BIN_SETTINGS,
                  n_thin: int = 40, n_boot: int = 600, p: float = 0.5,
                  seed: int = 42, edges: "np.ndarray | None" = None) -> pd.DataFrame:
    """Thinning-averaged binned clonal dN/dS with a thinning-aware bootstrap band.

    The published strategy Poisson-thins the synonymous counts ONCE (into an
    x-axis half A and an independent denominator half B), bins on dS_A, and
    bootstraps only over pairs. Because bin membership depends on that single
    random split, individual bins can dip/spike, and the pairs-only bootstrap --
    blind to the thinning -- reports a falsely tight band there.

    Here instead:
      * bin edges are fixed in dS space (stable across thinnings);
      * the central estimate pools numerator/denominator totals over ``n_thin``
        independent thinnings (soft bin membership -> averages out the per-split
        dips);
      * the 95% band comes from ``n_boot`` replicates that BOTH resample pairs and
        re-thin, so it reflects pair sampling and thinning variance together.

    ``tb`` carries raw per-pair ``kS``/``LS``/``kN``/``LN`` (from
    ``prepare_thin_base_for_classes``). Returns the same columns as
    :func:`adaptive_binned` (``dS_x``/``R_hat``/``med``/``lo``/``hi``/``KN``/
    ``KS_B``/``n_pairs``).
    """
    rng = np.random.default_rng(seed)
    kS = tb["kS"].to_numpy(float); LS = tb["LS"].to_numpy(float)
    kN = tb["kN"].to_numpy(float); LN = tb["LN"].to_numpy(float)
    if edges is None:
        dS_full = np.divide(kS, LS, out=np.zeros_like(kS), where=LS > 0)
        edges = _auto_geom_edges(dS_full, settings)
    if edges is None:
        return pd.DataFrame()
    edges = np.asarray(edges, dtype=float)
    nb = len(edges) - 1

    def _bin_totals(kSi, LSi, kNi, LNi):
        kSA = rng.binomial(kSi.astype(np.int64), p).astype(float)
        kSB = kSi - kSA
        LSA, LSB = p * LSi, (1 - p) * LSi
        dSA = np.divide(kSA, LSA, out=np.zeros_like(kSA), where=LSA > 0)
        bi = np.digitize(dSA, edges) - 1
        ok = (dSA > 0) & (bi >= 0) & (bi < nb)
        b = bi[ok]
        tot = {name: np.bincount(b, weights=col[ok], minlength=nb)
               for name, col in (("kN", kNi), ("LN", LNi), ("kSA", kSA),
                                 ("LSA", LSA), ("kSB", kSB), ("LSB", LSB))}
        cnt = np.bincount(b, minlength=nb).astype(float)
        return tot, cnt

    # central estimate: average totals over n_thin independent thinnings
    acc = {k: np.zeros(nb) for k in ("kN", "LN", "kSA", "LSA", "kSB", "LSB")}
    npair = np.zeros(nb)
    for _ in range(n_thin):
        tot, cnt = _bin_totals(kS, LS, kN, LN)
        for k in acc:
            acc[k] += tot[k]
        npair += cnt
    npair /= n_thin
    dS_x = np.divide(acc["kSA"], acc["LSA"], out=np.full(nb, np.nan), where=acc["LSA"] > 0)
    num = np.divide(acc["kN"], acc["LN"], out=np.full(nb, np.nan), where=acc["LN"] > 0)
    den = np.divide(acc["kSB"], acc["LSB"], out=np.full(nb, np.nan), where=acc["LSB"] > 0)
    R = np.divide(num, den, out=np.full(nb, np.nan), where=den > 0)

    # band: resample pairs AND re-thin each replicate
    n = len(kS)
    M = np.full((n_boot, nb), np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        tot, cnt = _bin_totals(kS[idx], LS[idx], kN[idx], LN[idx])
        nb_num = np.divide(tot["kN"], tot["LN"], out=np.full(nb, np.nan), where=tot["LN"] > 0)
        nb_den = np.divide(tot["kSB"], tot["LSB"], out=np.full(nb, np.nan), where=tot["LSB"] > 0)
        r = np.divide(nb_num, nb_den, out=np.full(nb, np.nan), where=nb_den > 0)
        r[cnt < 1] = np.nan
        M[b] = r
    lo = np.nanpercentile(M, 2.5, axis=0)
    hi = np.nanpercentile(M, 97.5, axis=0)

    out = pd.DataFrame({
        "dS_x": dS_x, "R_hat": R, "med": R, "lo": lo, "hi": hi,
        "KN": acc["kN"] / n_thin, "KS_B": acc["kSB"] / n_thin, "n_pairs": npair,
    })
    keep = ((npair >= settings.min_pairs_per_bin) & (out["dS_x"] > 0)
            & np.isfinite(out["R_hat"]) & (out["R_hat"] > 0))
    return out[keep].reset_index(drop=True)


def count_balanced_binned(tb: pd.DataFrame, *, n_bins: int = 10,
                          min_syn_per_bin: int = 20, floor: float = 1e-5,
                          min_log_width: float = 0.12, n_boot: int = 400,
                          seed: int = 42) -> pd.DataFrame:
    """Pooled clonal dN/dS in bins holding ~equal synonymous-SNV counts.

    The x-axis dS_A = (synonymous SNVs)/L is quantized at low divergence (the
    "1-SNV", "2-SNV" bands). Fixed-width/geometric bins slice finer than that
    step and split a single SNV-count band across bins. Here:

    - all pairs with dS_A < ``floor`` (default 1e-5; the 1-/2-SNV pairs) pool
      into one low-dS bin;
    - above the floor, edges are placed at equal cumulative-synonymous-count
      quantiles (so each bin has ~equal statistical weight), then thinned so no
      bin is narrower than ``min_log_width`` decades (no sub-quantization bins).

    Within a bin the estimate is a pooled ratio-of-totals (sum kN / sum LN over
    the independent denominator sum kS_B / sum LS_B), with bootstrap CIs.
    """
    df = tb[(tb["dS_A"] > 0) & np.isfinite(tb["dS_A"])].reset_index(drop=True)
    if len(df) < max(2 * min_syn_per_bin, 10):
        return pd.DataFrame()

    dS = df["dS_A"].to_numpy(float)
    order = np.argsort(dS)
    dS_s = dS[order]
    cw = np.cumsum(df["kS_A"].to_numpy(float)[order])
    total = cw[-1]
    nb = int(max(2, min(n_bins, total // max(1, min_syn_per_bin))))
    cand = dS_s[np.clip(np.searchsorted(cw, np.linspace(0, total, nb + 1)[1:-1]),
                        0, len(dS_s) - 1)]

    lo, hi = dS_s[0] * (1 - 1e-9), dS_s[-1] * (1 + 1e-9)
    raw = [lo] + ([floor] if lo < floor < hi else []) + list(cand) + [hi]
    raw = np.unique(np.asarray(raw, float))
    # thin edges so each bin spans >= min_log_width decades (keep the top edge)
    kept = [raw[0]]
    for e in raw[1:]:
        if np.log10(e) - np.log10(kept[-1]) >= min_log_width:
            kept.append(e)
    kept[-1] = hi
    edges = np.unique(np.asarray(kept, float))
    if edges.size < 3:
        return pd.DataFrame()

    def _agg(d: pd.DataFrame) -> pd.DataFrame:
        b = pd.cut(d["dS_A"], edges, labels=False, include_lowest=True).rename("bin")
        a = d.groupby(b).agg(KN=("kN", "sum"), LN=("LN", "sum"), KSA=("kS_A", "sum"),
                             LSA=("LS_A", "sum"), KSB=("kS_B", "sum"), LSB=("LS_B", "sum"),
                             n=("dS_A", "size"))
        a["dS_x"] = a["KSA"] / a["LSA"]
        a["R"] = (a["KN"] / a["LN"]) / (a["KSB"] / a["LSB"])
        return a

    base = _agg(df)
    rng = np.random.default_rng(seed)
    M = np.full((n_boot, edges.size - 1), np.nan)
    for b in range(n_boot):
        a = _agg(df.iloc[rng.integers(0, len(df), len(df))])
        M[b, a.index.to_numpy(int)] = a["R"].to_numpy(float)
    out = base.reset_index()
    out["lo"] = np.nanpercentile(M, 2.5, axis=0)[out["bin"].to_numpy(int)]
    out["med"] = np.nanpercentile(M, 50.0, axis=0)[out["bin"].to_numpy(int)]
    out["hi"] = np.nanpercentile(M, 97.5, axis=0)[out["bin"].to_numpy(int)]
    out["n_pairs"] = out["n"]
    out = out[(out["dS_x"] > 0) & np.isfinite(out["R"]) & (out["R"] > 0)].reset_index(drop=True)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG.parent.mkdir(parents=True, exist_ok=True)
    qp = load_qp()
    # ISO uses the 1D-EXTENDED recombination mask throughout (the corrected
    # clonal/recombined partition); the raw detected-mask version is not shown.
    iso_src = ISO_DNDS_EXT if (ISO_DNDS_EXT / "close_pairs_with_recombination.csv").exists() else ISO_DNDS
    iso = load_iso_from(iso_src, ISO_IF_CUTOFF)
    iso_typical_path = iso_src / "typical_pairs_fully_recombined.csv"
    print(f"ISO source (mask): {iso_src}")

    # ---- stratification points + summary ----
    strat = {}
    summ_rows = []
    for name, d in [("QP", qp), ("ISO", iso)]:
        pts = sd.compute_stratified_points(d, seed=0, thinning_p=0.5)
        strat[name] = pts
        s = sd.summarize_stratified_points(pts)
        s.insert(0, "dataset", name)
        summ_rows.append(s)
    strat_summary = pd.concat(summ_rows, ignore_index=True)
    strat_summary.to_csv(OUT_DIR / "stratification_summary.csv", index=False)

    # ---- clonal decay ----
    # Published estimator (single Poisson thinning, pooled ratio-of-totals, pairs
    # bootstrap, ~50 pairs/bin). Bin EDGES are defined once by QP per class and the
    # SAME edges are used for QP and ISO so they are comparable bin-by-bin; curves
    # are drawn at the shared bin midpoints with published-style error bars.
    settings = ps.ThinSettings()
    datasets_df = {"QP": qp, "ISO": iso}
    thin = {n: ps.prepare_thin_base_for_classes(d, settings, classes=tuple(CLASSES))
            for n, d in datasets_df.items()}
    binned = {n: {} for n in datasets_df}
    scatter = {n: {} for n in datasets_df}
    edges_by_cls, mids_by_cls = {}, {}
    binned_rows = []
    for cls in CLASSES:
        edges = ps_make_bins_auto(thin["QP"][cls]["dS_A"], GRID_BIN_SETTINGS)
        edges_by_cls[cls] = edges
        mids_by_cls[cls] = np.sqrt(edges[:-1] * edges[1:]) if edges is not None else None
        for name in datasets_df:
            tb = thin[name][cls]
            b = binned_in_edges(tb, edges, GRID_BIN_SETTINGS)
            if len(b):
                b = b.copy()
                b["x_mid"] = mids_by_cls[cls][b["bin"].to_numpy(int)]
            binned[name][cls] = b
            with np.errstate(divide="ignore", invalid="ignore"):
                R = (tb["kN"].to_numpy(float) / tb["LN"].to_numpy(float)) / \
                    (tb["kS_B"].to_numpy(float) / tb["LS_B"].to_numpy(float))
            scatter[name][cls] = (tb["dS_A"].to_numpy(float), R)
            if len(b):
                bb = b.copy(); bb.insert(0, "dataset", name); bb.insert(1, "class", cls)
                binned_rows.append(bb)
    pd.concat(binned_rows, ignore_index=True).to_csv(OUT_DIR / "clonal_decay_binned.csv", index=False)

    # ---- typical / unrelated-pair species-mean anchor point (per dataset/class) ----
    # QP typical: keep WITHIN-clade pairs (drop only cross-clade vulgatus-dorei);
    # dorei-dorei typical pairs are retained. ISO is already a single clade.
    clade_map = qp_clade_map()
    qp_typ = pd.read_csv(QP_TYPICAL) if QP_TYPICAL.exists() else None
    if qp_typ is not None:
        qp_typ = _filter_pairs_within_clade(qp_typ, clade_map, "typical")
    iso_typ = within_clade_typical(
        pd.read_csv(iso_typical_path) if iso_typical_path.exists() else None, "ISO")
    typ_tables = {"QP": qp_typ, "ISO": iso_typ}
    typ_points = {n: {} for n in ("QP", "ISO")}
    typ_rows = []
    for name in ("QP", "ISO"):
        for cls in CLASSES:
            pt = typical_pair_point(typ_tables[name], cls, p_thin=GRID_BIN_SETTINGS.p_thin,
                                    seed=GRID_BIN_SETTINGS.rng_seed)
            typ_points[name][cls] = pt
            if pt is not None:
                typ_rows.append({"dataset": name, "class": cls, "dS_x": pt[0], "dNdS": pt[1],
                                 "n_pairs": int(len(typ_tables[name]))})
    if typ_rows:
        pd.DataFrame(typ_rows).to_csv(OUT_DIR / "typical_pair_points.csv", index=False)

    # ---- theory curve per class = main-text (Fig 3 / Fig 4) parameters (no refit) ----
    pooled_theory = MAINTEXT_THEORY

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context({"font.size": 8}):
        fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2), dpi=200)

        DATASETS = [("QP", QP_COLOR, QP_MARKER), ("ISO", ISO_COLOR, ISO_MARKER)]

        # Row 1: stratification scatter overlay (distinct color + marker per dataset)
        for ax, stratum in zip(axes[0], STRATA):
            for name, color, marker in DATASETS:
                sub = strat[name]
                sub = sub[(sub["stratum"] == stratum) & sub["plot_valid"]]
                ax.scatter(sub["plot_dS"], sub["dNdS"], s=5, alpha=0.30, color=color,
                           marker=marker, linewidths=0, rasterized=True, label=name)
            if stratum == "full_core":
                d_s, dnds = sd._recombination_theory_curve()
                ax.plot(d_s, dnds, "k-", lw=0.7, label="recomb. theory")
            ax.axhline(1, ls="--", lw=0.5, color="grey")
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlim(2e-6, 5e-2); ax.set_ylim(1e-2, 1e1)
            ax.set_title(STRATUM_TITLE[stratum])
            ax.set_xlabel("$dS$ (full core)")
            leg = ax.legend(loc="upper right", fontsize=6, framealpha=0.9, markerscale=2)
            for lh in leg.legend_handles:
                lh.set_alpha(1)
        axes[0][0].set_ylabel("$dN/dS$")

        # Row 2: clonal decay in the FIXED QP bin grid. Background = faint per-pair
        # scatter (QP + ISO). Binned points = published-style error bars (bin median
        # + 95% bootstrap CI) at shared bin midpoints, for QP and ISO (1D-extended
        # mask). One POOLED (QP+ISO) theory curve per panel: single-class purifying
        # for all/missense, 2-class ("three-class") for nonsense, as in the grids.
        clonal_sets = [("QP", QP_COLOR, QP_MARKER, "QP"),
                       ("ISO", ISO_COLOR, ISO_MARKER, "ISO (1D-extended mask)")]
        fit_rows = []
        for ax, cls in zip(axes[1], CLASSES):
            xlim, ylim = CLASS_XLIM, CLASS_YLIM[cls]
            dS_grid = np.geomspace(xlim[0], xlim[1], 400)
            for name, color, marker, _disp in clonal_sets:
                x_sc, y_sc = scatter[name][cls]
                m = np.isfinite(x_sc) & np.isfinite(y_sc) & (x_sc > 0) & (y_sc > 0)
                ax.scatter(x_sc[m], y_sc[m], s=4, alpha=0.08, color=color, marker=marker,
                           linewidths=0, rasterized=True, zorder=1)
            for name, color, marker, disp in clonal_sets:
                b = binned[name][cls]
                if len(b) == 0:
                    continue
                x = b["x_mid"].to_numpy(float)
                yerr = [np.clip(b["med"] - b["lo"], 0, None).to_numpy(float),
                        np.clip(b["hi"] - b["med"], 0, None).to_numpy(float)]
                ax.errorbar(x, b["med"].to_numpy(float), yerr=yerr, fmt="-" + marker,
                            color=color, ecolor=color, lw=1.4, ms=4, mew=0,
                            elinewidth=1.0, capsize=2.5, capthick=1.0, zorder=4, label=disp)
            # main-text (Fig 3 / Fig 4) theory curve, fixed parameters (not refit)
            kind, par = pooled_theory[cls]
            if kind == "two_class":
                yth = dnds_two_class(dS_grid, par["fd"], par["alpha2"], par["s1mu"], par["s2mu"])
                lbl = (fr"{THEORY_LABEL[cls]} ($\alpha_2$={par['alpha2']:.2f}, "
                       fr"$s_1/\mu$={par['s1mu']:.2g})")
                ax.plot(dS_grid, yth, color="k", lw=1.3, ls="-", alpha=0.9, zorder=6, label=lbl)
                fit_rows.append({"class": cls, "model": "two_class", **par})
            elif kind == "one_class":
                yth = dNdS_purify_curve(dS_grid, par["fd"], par["sbymu"])
                lbl = fr"{THEORY_LABEL[cls]} ($s/\mu$={par['sbymu']:.2g})"
                ax.plot(dS_grid, yth, color="k", lw=1.3, ls="-", alpha=0.9, zorder=6, label=lbl)
                fit_rows.append({"class": cls, "model": "one_class", **par})
            # typical / unrelated-pair anchor (QP + ISO; mask-independent)
            for name, color in (("QP", QP_COLOR), ("ISO", ISO_COLOR)):
                pt = typ_points.get(name, {}).get(cls)
                if pt is None:
                    continue
                ax.scatter([pt[0]], [pt[1]], marker="X", s=55, color=color,
                           edgecolors="white", linewidths=0.8, zorder=7)
            ax.axhline(1, ls="--", lw=0.5, color="grey")
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlim(*xlim); ax.set_ylim(*ylim)
            ax.set_title(CLASS_TITLE[cls])
            ax.set_xlabel("clonal $dS$")
            ax.legend(loc="lower left", fontsize=5.0, framealpha=0.9, ncol=1)
        axes[1][0].set_ylabel("clonal $dN/dS$")
        if fit_rows:
            pd.DataFrame(fit_rows).to_csv(OUT_DIR / "clonal_purifying_fits.csv", index=False)

        fig.suptitle("P. vulgatus isolates vs P. vulgatus QP metagenome — same MIDAS reference",
                     fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        for ext in ("png", "pdf"):
            fig.savefig(FIG.with_suffix(f".{ext}"), bbox_inches="tight")
        plt.close(fig)

    print("=== stratification summary ===")
    print(strat_summary[["dataset", "stratum", "num_valid_points", "aggregate_dNdS"]].round(4).to_string(index=False))
    print(f"\nwrote {FIG.with_suffix('.png')}")
    print(f"wrote {OUT_DIR}/stratification_summary.csv, clonal_decay_binned.csv")


if __name__ == "__main__":
    main()
