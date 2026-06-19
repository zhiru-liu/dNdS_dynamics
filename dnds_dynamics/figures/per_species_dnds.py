
"""
per_species_dnds.py  (adaptive-binning edition)

Adds:
- Adaptive binning per species (auto determine n_bins).
- Option to use log-spaced ("geom") or equal-count ("quantile") bins.
- Stricter per-bin filtering via min_pairs_per_bin.
- Species-level skip if too few bins remain after filtering.

See ThinSettings for new options.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Helpers (names prefixed with ps_ to avoid clobbering your notebook)
# ---------------------------------------------------------------------

def ps_safe_div(num, den):
    """Vectorized division; NaN when den<=0."""
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    out = np.full_like(den, np.nan, dtype=float)
    np.divide(num, den, out=out, where=(den > 0))
    return out

def ps_safe_div_series(num, den):
    res = ps_safe_div(np.asarray(num), np.asarray(den))
    try:
        idx = num.index
    except Exception:
        idx = pd.RangeIndex(len(res))
    return pd.Series(res, index=idx)

@dataclass
class ThinSettings:
    # Poisson thinning
    p_thin: float = 0.5
    rng_seed: int = 42

    # Binning
    n_bins: int = 15                 # baseline if auto_bins=False
    auto_bins: bool = True           # adapt bin count per species
    target_pairs_per_bin: int = 50   # aim for this many pairs per bin
    min_pairs_per_bin: int = 20      # drop bins with fewer than this
    min_bins: int = 4                # minimum bins to try when auto
    max_bins: int = 20               # cap bins when auto
    binning_strategy: str = "geom"   # "geom" (log-spaced) or "quantile"

    # Bootstrap
    B: int = 400

    # Species-level filters
    min_pairs_per_species: int = 10    # skip sparse species
    min_bins_retained: int = 3         # after bin filtering, require at least this many bins


def ps_get_species_col(df: pd.DataFrame,
                       candidates: Tuple[str,...]=("species_name","species","Species","taxon","taxon_name")) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def ps_thin_counts(syn_df: pd.DataFrame, p: float, rng: np.random.Generator, eps: float=1e-12) -> pd.DataFrame:
    """Poisson thinning of synonymous counts into (A,B) with exposures.
    Expects columns: kS, LS
    Returns new cols: kS_A, kS_B, LS_A, LS_B, dS_A, dS_B
    """
    out = syn_df.copy()
    out = out.reset_index(drop=True)

    kS = out['kS'].astype(int).to_numpy()
    out['kS_A'] = rng.binomial(kS, p)
    out['kS_B'] = kS - out['kS_A']

    out['LS_A'] = p * out['LS']
    out['LS_B'] = (1 - p) * out['LS']

    eps = float(eps)
    out['dS_A'] = out['kS_A'] / np.clip(out['LS_A'], eps, None)
    out['dS_B'] = out['kS_B'] / np.clip(out['LS_B'], eps, None)
    return out


def _auto_bin_count(n_pairs: int, settings: ThinSettings) -> int:
    if not settings.auto_bins:
        return settings.n_bins
    if n_pairs <= 0:
        return 0
    n = max(settings.min_bins, min(settings.max_bins, n_pairs // max(1, settings.target_pairs_per_bin)))
    # if extremely sparse, still allow a small number of bins
    return max(settings.min_bins, n) if n_pairs >= settings.min_pairs_per_bin * settings.min_bins else 0


def ps_make_bins_auto(dS_A: pd.Series, settings: ThinSettings) -> Optional[np.ndarray]:
    """Compute bin edges based on settings and dS_A distribution.
    Returns None if insufficient data to bin.
    """
    dS_pos = pd.Series(dS_A).astype(float)
    dS_pos = dS_pos[np.isfinite(dS_pos) & (dS_pos > 0)]
    n_pairs = int(dS_pos.size)
    n_bins = _auto_bin_count(n_pairs, settings)
    if n_bins <= 0:
        return None
    if settings.binning_strategy == "quantile":
        # equal-count bins (unique edges only)
        qs = np.linspace(0, 1, n_bins + 1)
        edges = np.quantile(dS_pos.to_numpy(), qs, method="linear")
        edges = np.unique(edges)
        # need at least 3 unique edges to form >=2 bins; but we will filter bins later anyway
        return edges if edges.size >= 3 else None
    else:
        # geometric (log-spaced) bins
        lo, hi = float(dS_pos.min()), float(dS_pos.max())
        if not (np.isfinite(lo) and np.isfinite(hi)) or lo <= 0 or hi <= lo:
            return None
        return np.geomspace(lo, hi, n_bins + 1)


def ps_aggregate_bins(thin_df: pd.DataFrame, bin_edges: np.ndarray,
                      kN_col: str, LN_col: str,
                      min_pairs_per_bin: int,
                      eps: float=1e-12) -> pd.DataFrame:
    """
    Pooled x and pooled ratio-of-totals with independent denom from split B.

    Requires thin_df to have at least: dS_A, kS_A, kS_B, LS_A, LS_B, plus the
    nonsyn columns kN_col and LN_col provided as args.

    Drops bins with n_pairs < min_pairs_per_bin.
    """
    df = thin_df.copy()
    df = df[(df['dS_A'] > 0) & np.isfinite(df['dS_A'])]

    if bin_edges is None or (np.asarray(bin_edges).size < 3):
        return pd.DataFrame(columns=['bin','KN','KS_A','KS_B','LN','LS_A','LS_B','n_pairs','dS_x','R_hat'])

    df['bin'] = pd.cut(df['dS_A'], bin_edges, labels=False, include_lowest=True)

    gb = df.groupby('bin', dropna=True)
    agg = gb.agg(
        KN   = (kN_col, 'sum'),
        KS_A = ('kS_A', 'sum'),
        KS_B = ('kS_B', 'sum'),
        LN   = (LN_col, 'sum'),
        LS_A = ('LS_A', 'sum'),
        LS_B = ('LS_B', 'sum'),
        n_pairs=('dS_A','size')
    ).reset_index()

    # drop sparse bins
    agg = agg[agg['n_pairs'] >= int(min_pairs_per_bin)].reset_index(drop=True)

    # pooled x and pooled ratio-of-totals
    agg['dS_x']  = ps_safe_div_series(agg['KS_A'], agg['LS_A'])
    numer        = ps_safe_div_series(agg['KN'],   agg['LN'])
    denom        = ps_safe_div_series(agg['KS_B'], agg['LS_B'])
    agg['R_hat'] = numer / denom

    # keep only clean rows
    x = agg['dS_x'].to_numpy(dtype=float)
    y = agg['R_hat'].to_numpy(dtype=float)
    m = (x > 0) & (y > 0) & np.isfinite(y)
    return agg.loc[m].reset_index(drop=True)


def ps_bootstrap_binned(thin_base: pd.DataFrame,
                        kN_col: str, LN_col: str,
                        settings: ThinSettings) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(settings.rng_seed)

    edges = ps_make_bins_auto(thin_base['dS_A'], settings=settings)
    if edges is None or len(edges) < 3:
        return (pd.DataFrame(columns=['bin','dS_x','R_hat','n_pairs']),
                pd.DataFrame(columns=['bin','lo','med','hi']))

    n_total_bins = len(edges) - 1

    agg0 = ps_aggregate_bins(thin_base, edges, kN_col=kN_col, LN_col=LN_col,
                             min_pairs_per_bin=settings.min_pairs_per_bin)
    if len(agg0) < settings.min_bins_retained:
        return (pd.DataFrame(columns=['bin','dS_x','R_hat','n_pairs']),
                pd.DataFrame(columns=['bin','lo','med','hi']))

    m = len(thin_base)
    if m == 0:
        return (pd.DataFrame(columns=['bin','dS_x','R_hat','n_pairs']),
                pd.DataFrame(columns=['bin','lo','med','hi']))

    Rmats = []
    for _ in range(settings.B):
        idx = rng.integers(0, m, size=m)
        th_b = thin_base.iloc[idx].reset_index(drop=True)
        agg_b = ps_aggregate_bins(th_b, edges, kN_col=kN_col, LN_col=LN_col,
                                  min_pairs_per_bin=settings.min_pairs_per_bin)

        R_b = np.full(n_total_bins, np.nan, dtype=float)
        if len(agg_b) > 0:
            bins_b = agg_b['bin'].to_numpy(dtype=int)
            vals_b = agg_b['R_hat'].to_numpy(dtype=float)
            mask = (bins_b >= 0) & (bins_b < n_total_bins) & np.isfinite(vals_b)
            R_b[bins_b[mask]] = vals_b[mask]
        Rmats.append(R_b)

    if len(Rmats) == 0:
        boot_quants = pd.DataFrame({'bin': agg0['bin'], 'lo': np.nan, 'med': np.nan, 'hi': np.nan})
        return agg0, boot_quants

    Rm = np.vstack(Rmats)
    q_lo = np.nanpercentile(Rm, 2.5, axis=0)
    q_md = np.nanpercentile(Rm, 50.0, axis=0)
    q_hi = np.nanpercentile(Rm, 97.5, axis=0)

    sel_bins = agg0['bin'].to_numpy(dtype=int)
    boot_quants = pd.DataFrame({
        'bin': sel_bins,
        'lo' : q_lo[sel_bins],
        'med': q_md[sel_bins],
        'hi' : q_hi[sel_bins],
    })
    return agg0, boot_quants



# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

@dataclass
class SpeciesResults:
    species: str
    # class -> (agg0, boots)
    binned_by_class: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]
    # per-pair scatter after thinning (for each class we store a tuple
    # (dS_A, R_pair) arrays)
    scatter_by_class: Dict[str, Tuple[np.ndarray, np.ndarray]]
    n_pairs: int


def prepare_thin_base_for_classes(full_df: pd.DataFrame,
                                  settings: ThinSettings,
                                  classes: Tuple[str,...]=("all","missense","nonsense")) -> Dict[str, pd.DataFrame]:
    """
    Given full_df for a species (or any subset), construct a thin_base dataframe
    with synonymous thinning + attach the appropriate nonsyn counts for each class.
    Returns mapping class -> thin_base frame with columns:
      kS, LS, kS_A, kS_B, LS_A, LS_B, dS_A, dS_B, and class-specific kN/LN
    """
    rng = np.random.default_rng(settings.rng_seed)

    # synonymous base
    syn_df = pd.DataFrame({
        'kS': full_df['clonal_diff_4D'].to_numpy(),
        'LS': full_df['clonal_len_4D'].to_numpy(),
    })
    thin_syn = ps_thin_counts(syn_df, p=settings.p_thin, rng=rng)

    OPPS_PER_SITE = 3.0  # 3 opps per site for clonal regions

    out = {}
    for cls in classes:
        tb = thin_syn.copy()
        if cls == "all":
            tb['kN'] = full_df['clonal_diff_1D'].to_numpy()
            tb['LN'] = full_df['clonal_len_1D'].to_numpy()
        elif cls == "missense":
            tb['kN'] = full_df['clonal_mut_m'].to_numpy()
            tb['LN'] = full_df['clonal_m'   ].to_numpy() / OPPS_PER_SITE
        elif cls == "nonsense":
            tb['kN'] = full_df['clonal_mut_n'].to_numpy()
            tb['LN'] = full_df['clonal_n'   ].to_numpy() / OPPS_PER_SITE
        else:
            raise ValueError(f"Unknown class {cls}")
        out[cls] = tb
    return out


def compute_species_results(full_df: pd.DataFrame,
                            settings: ThinSettings,
                            classes: Tuple[str,...]=("all","missense","nonsense")) -> Optional[SpeciesResults]:
    """
    Run the pipeline on a dataframe that is already filtered to a single species.
    Returns SpeciesResults containing binned curves + bootstrap ribbons and
    per-pair scatter for convenience.
    May return None if species is too sparse.
    """
    if len(full_df) < settings.min_pairs_per_species:
        return None

    # prepare thin bases for each class
    tbs = prepare_thin_base_for_classes(full_df, settings, classes=classes)

    binned = {}
    scatter = {}
    n_bins_ok = False
    for cls, tb in tbs.items():
        agg0, boots = ps_bootstrap_binned(tb, kN_col="kN", LN_col="LN", settings=settings)
        binned[cls] = (agg0, boots)
        n_bins_ok = n_bins_ok or (len(agg0) >= settings.min_bins_retained)

        # per-pair scatter (x = dS_A; y = ratio using denom from B)
        numer = ps_safe_div(tb['kN'], tb['LN'])
        denom = ps_safe_div(tb['kS_B'], tb['LS_B'])
        R_pair = numer / denom
        scatter[cls] = (tb['dS_A'].to_numpy(dtype=float), R_pair.astype(float))

    if not n_bins_ok:
        return None

    return SpeciesResults(
        species="(unknown in this call)",
        binned_by_class=binned,
        scatter_by_class=scatter,
        n_pairs=len(full_df),
    )


def compute_all_species(full_df: pd.DataFrame,
                        settings: ThinSettings,
                        species_col: Optional[str]=None,
                        classes: Tuple[str,...]=("all","missense","nonsense")) -> Dict[str, SpeciesResults]:
    """
    Group full_df by species, compute per-species results, and return a dict.
    """
    if species_col is None:
        species_col = ps_get_species_col(full_df)
        if species_col is None:
            raise ValueError("Could not find a species column. Pass species_col explicitly.")

    out: Dict[str, SpeciesResults] = {}
    for sp, df_sp in full_df.groupby(species_col, sort=False):
        df_sp = df_sp.reset_index(drop=True).astype(float)
        res = compute_species_results(df_sp, settings=settings, classes=classes)
        if res is None:
            continue
        res.species = str(sp)
        out[str(sp)] = res
    return out


def plot_species_grid(results: Dict[str, SpeciesResults],
                      which_class: str = "all",
                      ncols: int = 4,
                      figsize: Tuple[int,int] = (16, 3),
                      scatter_alpha: float = 0.2,
                      marker_size: float = 6.0,
                      title_prefix: Optional[str] = None,
                      sharex: bool = True,
                      sharey: bool = True) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create a grid of small-multiples for the given class ("all", "missense", "nonsense").
    Each panel shows per-pair scatter + bootstrap ribbon + median curve.
    """
    if not results:
        raise ValueError("Empty results passed to plot_species_grid.")

    species = list(results.keys())
    n = len(species)
    ncols = max(1, int(ncols))
    nrows = (n + ncols - 1) // ncols

    # Choose a taller figure if many rows are needed
    if figsize == (16, 3):
        figsize = (16, max(3, 2 + 2*nrows))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=sharex, sharey=sharey)
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    for idx, sp in enumerate(species):
        r = results[sp]
        ax = axes[idx // ncols, idx % ncols]

        # scatter
        x_sc, y_sc = r.scatter_by_class[which_class]
        ax.scatter(x_sc, y_sc, s=marker_size, alpha=scatter_alpha, linewidths=0, rasterized=True)

        # ribbon + median
        agg0, boots = r.binned_by_class[which_class]
        if len(boots)>0 and len(agg0)>0:
            # align to shown bins
            x = agg0['dS_x'].to_numpy(dtype=float)
            lo = boots['lo'].to_numpy(dtype=float)
            md = boots['med'].to_numpy(dtype=float)
            hi = boots['hi'].to_numpy(dtype=float)
            ax.fill_between(x, lo, hi, alpha=0.2, linewidth=0)
            ax.plot(x, md, lw=1.8)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(f"{sp} (n={r.n_pairs})", fontsize=10)
        ax.grid(True, which="both", ls=':', alpha=0.3)

    # turn off any empty axes
    for j in range(n, nrows*ncols):
        axes[j // ncols, j % ncols].axis('off')

    if title_prefix:
        fig.suptitle(f"{title_prefix} — class: {which_class}", y=0.98, fontsize=12)
    fig.tight_layout()
    return fig, axes


def export_species_binned(results: Dict[str, SpeciesResults],
                          which_class: str = "all") -> pd.DataFrame:
    """
    Collect per-species binned medians into a tidy dataframe for downstream use.
    Returns columns: species, bin, dS_x, R_med, R_lo, R_hi, n_pairs_species
    """
    rows = []
    for sp, res in results.items():
        agg0, boots = res.binned_by_class[which_class]
        if len(agg0)==0:
            continue
        # merge on 'bin' to align
        merged = agg0[['bin','dS_x']].merge(boots[['bin','lo','med','hi']], on='bin', how='left')
        for _, row in merged.iterrows():
            rows.append({
                'species': sp,
                'bin': int(row['bin']),
                'dS_x': float(row['dS_x']),
                'R_med': float(row['med']),
                'R_lo' : float(row['lo']),
                'R_hi' : float(row['hi']),
                'n_pairs_species': int(res.n_pairs),
            })
    return pd.DataFrame.from_records(rows)


# Convenience: single-call end-to-end
def run_per_species(full_dnds_df: pd.DataFrame,
                    classes: Tuple[str,...]=("all","missense","nonsense"),
                    species_col: Optional[str]=None,
                    settings: Optional[ThinSettings]=None,
                    ncols: int = 4) -> Dict[str, SpeciesResults]:
    """
    Compute results for all species and draw a default grid for each class.
    Returns the 'results' dict so you can reuse it.
    """
    if settings is None:
        settings = ThinSettings()
    res = compute_all_species(full_dnds_df, settings=settings, species_col=species_col, classes=classes)

    # one grid per class
    for cls in classes:
        plot_species_grid(res, which_class=cls, ncols=ncols,
                          title_prefix="Per-species dN/dS dynamics")

    return res
