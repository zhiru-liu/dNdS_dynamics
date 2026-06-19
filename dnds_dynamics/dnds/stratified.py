"""Derived stratified dN/dS tables for isolate dN/dS outputs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..snv_helpers.isolate import DEFAULT_SNV_TABLE_ROOT


CLOSE_DNDS_NAME = "close_pairs_with_recombination.csv"
CLONAL_DNDS_NAME = "clonal_pairs_no_recombination.csv"
TYPICAL_DNDS_NAME = "typical_pairs_fully_recombined.csv"

STRATIFIED_POINTS_NAME = "stratified_dnds_points.csv"
STRATIFIED_SUMMARY_NAME = "stratified_dnds_summary.csv"
STRATIFIED_METADATA_NAME = "stratified_dnds_metadata.json"

ID_COLUMNS = [
    "species_name",
    "sample 1",
    "sample 2",
    "pair_class",
    "event_count",
    "dedup_event_count",
    "clonal_divergence",
    "clonal_fraction",
]

STRATA = [
    ("full_core", "core_diff_4D", "core_len_4D", "core_diff_1D", "core_len_1D"),
    ("recombined", "recomb_diff_4D", "recomb_len_4D", "recomb_diff_1D", "recomb_len_1D"),
    ("clonal", "clonal_diff_4D", "clonal_len_4D", "clonal_diff_1D", "clonal_len_1D"),
]


@dataclass(frozen=True)
class StratifiedDndsResult:
    accession: str
    output_dir: Path
    points_path: Path
    summary_path: Path
    metadata_path: Path
    plot_paths: tuple[Path, ...]
    metadata: dict


def poisson_thinning(
    diffs: pd.Series | np.ndarray,
    opportunities: pd.Series | np.ndarray,
    *,
    seed: int = 0,
    p: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split synonymous differences into two independent rate estimates.

    This mirrors ``computed_poisson_thinning`` in the original dNdS repo, with a
    deterministic generator so the derived plotting table can be regenerated.
    """
    if not 0 < p < 1:
        raise ValueError("p must be between 0 and 1")
    k_syn = pd.to_numeric(pd.Series(diffs), errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    l_syn = pd.to_numeric(pd.Series(opportunities), errors="coerce").to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    k_syn_a = rng.binomial(k_syn, p)
    k_syn_b = k_syn - k_syn_a
    l_syn_a = l_syn * p
    l_syn_b = l_syn * (1 - p)
    d_syn_a = _safe_div_array(k_syn_a, l_syn_a)
    d_syn_b = _safe_div_array(k_syn_b, l_syn_b)
    return d_syn_a, d_syn_b, k_syn_a, k_syn_b


def load_close_pair_dnds(dnds_dir: Path | str) -> pd.DataFrame:
    """Load close-recombination and close-no-recombination outputs as one table."""
    dnds_dir = Path(dnds_dir)
    pieces: list[pd.DataFrame] = []

    close_path = dnds_dir / CLOSE_DNDS_NAME
    if close_path.exists():
        close = pd.read_csv(close_path)
        if "pair_class" not in close.columns:
            close["pair_class"] = "close_recombination"
        pieces.append(close)

    clonal_path = dnds_dir / CLONAL_DNDS_NAME
    if clonal_path.exists():
        clonal = pd.read_csv(clonal_path)
        if "pair_class" not in clonal.columns:
            clonal["pair_class"] = "close_no_recombination"
        for suffix in ["len_4D", "len_1D", "diff_4D", "diff_1D"]:
            clonal[f"clonal_{suffix}"] = clonal[f"core_{suffix}"]
        for suffix in ["len_4D", "len_1D", "diff_4D", "diff_1D"]:
            clonal[f"recomb_{suffix}"] = np.nan
        pieces.append(clonal)

    if not pieces:
        raise FileNotFoundError(f"No close/clonal dN/dS files found in {dnds_dir}")

    combined = pd.concat(pieces, ignore_index=True, sort=False)
    for col in ID_COLUMNS:
        if col not in combined.columns:
            combined[col] = np.nan
    return combined


def compute_stratified_points(
    pair_df: pd.DataFrame,
    *,
    seed: int = 0,
    thinning_p: float = 0.5,
) -> pd.DataFrame:
    """Compute point-level data used by the dN/dS separation plot."""
    df = pair_df.copy()
    d_syn_a, d_syn_b, k_syn_a, k_syn_b = poisson_thinning(
        df["core_diff_4D"],
        df["core_len_4D"],
        seed=seed,
        p=thinning_p,
    )
    df["core_dS_thin_x"] = d_syn_a
    df["core_dS_thin_denominator"] = d_syn_b
    df["core_diff_4D_thin_x"] = k_syn_a
    df["core_diff_4D_thin_denominator"] = k_syn_b
    df["core_dS_naive"] = _safe_div_series(df["core_diff_4D"], df["core_len_4D"])
    df["core_dN_naive"] = _safe_div_series(df["core_diff_1D"], df["core_len_1D"])

    rows = []
    for stratum, k_s_col, l_s_col, k_n_col, l_n_col in STRATA:
        if stratum == "recombined":
            stratum_df = df[df["event_count"].fillna(0) > 0].copy()
        else:
            stratum_df = df.copy()
        if stratum_df.empty:
            continue
        rows.append(
            _make_stratum_points(
                stratum_df,
                stratum=stratum,
                k_s_col=k_s_col,
                l_s_col=l_s_col,
                k_n_col=k_n_col,
                l_n_col=l_n_col,
            )
        )

    if not rows:
        return pd.DataFrame()
    points = pd.concat(rows, ignore_index=True, sort=False)
    points["plot_valid"] = (
        np.isfinite(points["plot_dS"])
        & (points["plot_dS"] > 0)
        & np.isfinite(points["dS_for_dNdS"])
        & (points["dS_for_dNdS"] > 0)
        & np.isfinite(points["dN"])
        & (points["dN"] > 0)
        & np.isfinite(points["dNdS"])
        & (points["dNdS"] > 0)
    )
    return points


def summarize_stratified_points(points: pd.DataFrame) -> pd.DataFrame:
    """Aggregate point-level stratified dN/dS values by stratum."""
    rows = []
    for stratum, group in points.groupby("stratum", sort=False):
        valid = group[group["plot_valid"]]
        d_s_agg = _scalar_div(valid["synonymous_diffs"].sum(), valid["synonymous_sites"].sum())
        d_n_agg = _scalar_div(valid["nonsynonymous_diffs"].sum(), valid["nonsynonymous_sites"].sum())
        rows.append(
            {
                "stratum": stratum,
                "num_points": int(group.shape[0]),
                "num_valid_points": int(valid.shape[0]),
                "synonymous_diffs": int(valid["synonymous_diffs"].sum()),
                "synonymous_sites": float(valid["synonymous_sites"].sum()),
                "nonsynonymous_diffs": int(valid["nonsynonymous_diffs"].sum()),
                "nonsynonymous_sites": float(valid["nonsynonymous_sites"].sum()),
                "aggregate_dS": d_s_agg,
                "aggregate_dN": d_n_agg,
                "aggregate_dNdS": _scalar_div(d_n_agg, d_s_agg),
                "median_point_dNdS": float(valid["dNdS"].median()) if len(valid) else np.nan,
                "q25_point_dNdS": float(valid["dNdS"].quantile(0.25)) if len(valid) else np.nan,
                "q75_point_dNdS": float(valid["dNdS"].quantile(0.75)) if len(valid) else np.nan,
                "min_plot_dS": float(valid["plot_dS"].min()) if len(valid) else np.nan,
                "max_plot_dS": float(valid["plot_dS"].max()) if len(valid) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def save_dnds_separation_plot(points: pd.DataFrame, output_path: Path | str) -> Path:
    """Save a three-panel dN/dS separation plot matching the original repo."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    colors = {
        "full_core": "tab:grey",
        "recombined": "#FF968D",
        "clonal": "#AECDE1",
    }
    titles = {
        "full_core": "Full (core) genome",
        "recombined": "Recombined regions",
        "clonal": "Clonal regions",
    }
    alphas = {
        "full_core": 0.25,
        "recombined": 0.35,
        "clonal": 0.35,
    }

    with plt.rc_context({"font.size": 8}):
        fig, axes = plt.subplots(1, 3, figsize=(7.5, 1.7), dpi=300)
        fig.subplots_adjust(wspace=0.15)
        for ax, stratum in zip(axes, ["full_core", "recombined", "clonal"]):
            sub = points[(points["stratum"] == stratum) & points["plot_valid"]]
            ax.scatter(
                sub["plot_dS"],
                sub["dNdS"],
                s=1,
                alpha=alphas[stratum],
                color=colors[stratum],
                rasterized=True,
            )
            if stratum == "full_core":
                d_s_arr, dnds_arr = _recombination_theory_curve()
                ax.plot(d_s_arr, dnds_arr, linestyle="-", color="k", linewidth=0.5)
            ax.set_ylim([1e-2, 1e1])
            ax.set_xlim([2e-6, 2e-2])
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("$dS$ (Full genome)")
            ax.axhline(1, linewidth=0.5, linestyle="--", color="grey")
            ax.set_title(titles[stratum])
        axes[0].set_ylabel("$dN/dS$")
        axes[1].set_yticklabels([])
        axes[2].set_yticklabels([])
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
    return output_path


def compute_stratified_isolate_dnds(
    accession: str,
    *,
    table_root: Path | str = DEFAULT_SNV_TABLE_ROOT,
    dnds_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
    seed: int = 0,
    thinning_p: float = 0.5,
    plot_formats: tuple[str, ...] = ("png", "pdf"),
) -> StratifiedDndsResult:
    """Compute and cache stratified dN/dS separation tables for one accession."""
    table_root = Path(table_root)
    if dnds_dir is None:
        dnds_dir = table_root / accession / "dnds"
    dnds_dir = Path(dnds_dir)
    if output_dir is None:
        output_dir = dnds_dir
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pair_df = load_close_pair_dnds(dnds_dir)
    points = compute_stratified_points(pair_df, seed=seed, thinning_p=thinning_p)
    summary = summarize_stratified_points(points)

    points_path = output_dir / STRATIFIED_POINTS_NAME
    summary_path = output_dir / STRATIFIED_SUMMARY_NAME
    metadata_path = output_dir / STRATIFIED_METADATA_NAME
    points.to_csv(points_path, index=False)
    summary.to_csv(summary_path, index=False)

    plot_paths = []
    for fmt in plot_formats:
        fmt = fmt.lower().lstrip(".")
        if not fmt:
            continue
        plot_paths.append(save_dnds_separation_plot(points, output_dir / f"stratified_dnds_separation.{fmt}"))

    metadata = {
        "accession": accession,
        "seed": seed,
        "thinning_p": thinning_p,
        "input_dir": str(dnds_dir),
        "num_input_pairs": int(pair_df.shape[0]),
        "num_points": int(points.shape[0]),
        "num_valid_points": int(points["plot_valid"].sum()) if "plot_valid" in points else 0,
        "outputs": {
            "points": str(points_path),
            "summary": str(summary_path),
            "plots": [str(path) for path in plot_paths],
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    return StratifiedDndsResult(
        accession=accession,
        output_dir=output_dir,
        points_path=points_path,
        summary_path=summary_path,
        metadata_path=metadata_path,
        plot_paths=tuple(plot_paths),
        metadata=metadata,
    )


def _make_stratum_points(
    df: pd.DataFrame,
    *,
    stratum: str,
    k_s_col: str,
    l_s_col: str,
    k_n_col: str,
    l_n_col: str,
) -> pd.DataFrame:
    out = df[ID_COLUMNS].copy()
    out["stratum"] = stratum
    out["plot_dS"] = df["core_dS_thin_x"].to_numpy(dtype=float)
    out["core_dS_thin_denominator"] = df["core_dS_thin_denominator"].to_numpy(dtype=float)
    out["core_dS_naive"] = df["core_dS_naive"].to_numpy(dtype=float)
    out["core_dN_naive"] = df["core_dN_naive"].to_numpy(dtype=float)
    out["synonymous_diffs"] = pd.to_numeric(df[k_s_col], errors="coerce")
    out["synonymous_sites"] = pd.to_numeric(df[l_s_col], errors="coerce")
    out["nonsynonymous_diffs"] = pd.to_numeric(df[k_n_col], errors="coerce")
    out["nonsynonymous_sites"] = pd.to_numeric(df[l_n_col], errors="coerce")
    out["dS_region"] = _safe_div_series(out["synonymous_diffs"], out["synonymous_sites"])
    out["dN"] = _safe_div_series(out["nonsynonymous_diffs"], out["nonsynonymous_sites"])
    if stratum == "full_core":
        out["dS_for_dNdS"] = out["core_dS_thin_denominator"]
        out["dS_method"] = "poisson_thinned_full_core_4D"
    else:
        out["dS_for_dNdS"] = out["dS_region"]
        out["dS_method"] = "naive_region_4D"
    out["dNdS"] = _safe_div_series(out["dN"], out["dS_for_dNdS"])
    return out


def _safe_div_series(num: pd.Series, denom: pd.Series) -> pd.Series:
    num = pd.to_numeric(num, errors="coerce").astype(float)
    denom = pd.to_numeric(denom, errors="coerce").astype(float)
    out = num.divide(denom)
    return out.where((denom > 0) & np.isfinite(out))


def _safe_div_array(num: np.ndarray, denom: np.ndarray) -> np.ndarray:
    num = np.asarray(num, dtype=float)
    denom = np.asarray(denom, dtype=float)
    out = np.full(num.shape, np.nan, dtype=float)
    np.divide(num, denom, out=out, where=(denom > 0))
    out[~np.isfinite(out)] = np.nan
    return out


def _scalar_div(num: float, denom: float) -> float:
    if not np.isfinite(num) or not np.isfinite(denom) or denom <= 0:
        return np.nan
    return float(num / denom)


def _recombination_theory_curve() -> tuple[np.ndarray, np.ndarray]:
    theta = 3e-2
    dnds_clonal = 1
    dnds_recomb = 1e-1
    ds_mid = 10 ** (-3.8)
    k = 10
    d_s_clonal = np.logspace(-6, -3)
    f_recomb = 1 / (1 + np.exp(-k * (np.log10(d_s_clonal) - np.log10(ds_mid))))
    d_s = (1 - f_recomb) * d_s_clonal + f_recomb * theta
    d_n = (1 - f_recomb) * dnds_clonal * d_s_clonal + f_recomb * dnds_recomb * theta
    return d_s, d_n / d_s
