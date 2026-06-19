"""dN/dS counting logic for UHGG isolate SNV tables."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from ..snv_helpers.isolate import DEFAULT_SNV_TABLE_ROOT, IsolateSNVHelper


DEFAULT_IDENTICAL_FRACTION_ROOT = config.LIUGOOD_CF_ISOLATES

# Mirrors close_pair_hmm/cphmm/config.py: HMM_BLOCK_SIZE * HMM_MIN_SEQ_LEN.
# Contigs with fewer covered 4D sites than this in a given pair are skipped by
# the CP-HMM, so their dN/dS counts cannot be split into recomb/clonal regions
# consistently with the recombination cache. Passing this threshold to the
# count functions reproduces the same per-pair contig filter.
CPHMM_HMM_BLOCK_SIZE = 10
CPHMM_HMM_MIN_SEQ_LEN = 100
DEFAULT_MIN_CONTIG_4D_SITES = CPHMM_HMM_BLOCK_SIZE * CPHMM_HMM_MIN_SEQ_LEN

BASE_COLUMNS = [
    "species_name",
    "sample 1",
    "sample 2",
    "pair_class",
    "core_diff",
    "core_len",
    "core_len_4D",
    "core_len_1D",
    "core_diff_4D",
    "core_diff_1D",
]

CLOSE_COLUMNS = BASE_COLUMNS + [
    "event_count",
    "dedup_event_count",
    "clonal_divergence",
    "clonal_fraction",
    "recomb_len_4D",
    "recomb_len_1D",
    "recomb_diff_4D",
    "recomb_diff_1D",
    "clonal_len_4D",
    "clonal_len_1D",
    "clonal_diff_4D",
    "clonal_diff_1D",
    "clonal_n",
    "clonal_m",
    "clonal_nn",
    "recomb_n",
    "recomb_m",
    "recomb_nn",
    "clonal_mut_n",
    "clonal_mut_m",
    "clonal_mut_nn",
    "recomb_mut_n",
    "recomb_mut_m",
    "recomb_mut_nn",
]

CLONAL_COLUMNS = BASE_COLUMNS + [
    "event_count",
    "dedup_event_count",
    "clonal_divergence",
    "clonal_fraction",
    "clonal_n",
    "clonal_m",
    "clonal_nn",
    "clonal_mut_n",
    "clonal_mut_m",
    "clonal_mut_nn",
]

TYPICAL_COLUMNS = BASE_COLUMNS + [
    "identical_block_fraction",
    "core_n",
    "core_m",
    "core_nn",
    "core_mut_n",
    "core_mut_m",
    "core_mut_nn",
]


@dataclass(frozen=True)
class DndsResult:
    accession: str
    output_dir: Path
    close_path: Path
    clonal_path: Path
    typical_path: Path
    summary_path: Path
    summary: dict


def _sample_pair_id(sample1: str, sample2: str) -> tuple[str, str]:
    return str(sample1), str(sample2)


def short_contig_site_mask(
    helper: IsolateSNVHelper,
    covered: pd.Series,
    min_contig_4D_sites: int,
) -> pd.Series:
    """Return a site-level mask that drops contigs with too few covered 4D sites.

    Reproduces the per-pair contig filter used by the CP-HMM: a contig is kept
    only if it has at least ``min_contig_4D_sites`` covered 4D sites in the
    given pair (``covered`` is the pair's site-coverage Series). Sites on
    excluded contigs are set to False; all other sites stay True.
    """
    if min_contig_4D_sites is None or min_contig_4D_sites <= 0:
        return pd.Series(True, index=helper.index, name="keep_contig")
    contigs = helper.index.get_level_values("Contig")
    covered_4D = covered & helper.core_4D
    per_contig_counts = covered_4D.groupby(contigs).sum()
    keep_contigs = per_contig_counts[per_contig_counts >= int(min_contig_4D_sites)].index
    keep = pd.Series(contigs.isin(keep_contigs), index=helper.index, name="keep_contig")
    return keep


def _count_mut_types(values: pd.Series) -> dict[str, int]:
    return {
        "mut_n": int((values == "n").sum()),
        "mut_m": int((values == "m").sum()),
        "mut_nn": int((values == "nn").sum()),
    }


def _region_opportunities(helper: IsolateSNVHelper, covered: pd.Series, region: pd.Series) -> dict[str, int]:
    mask = covered & region & helper.core_1D
    sums = helper.mut_df.loc[mask, ["n", "m", "nn"]].sum()
    return {key: int(sums[key]) for key in ["n", "m", "nn"]}


def _region_mutations(helper: IsolateSNVHelper, diff: pd.Series, region: pd.Series) -> dict[str, int]:
    region_snv = region.reindex(helper.snv_index, fill_value=False)
    mut_types = helper.snv_types[diff & region_snv & helper.snv_1D]
    counts = _count_mut_types(mut_types)
    return {
        "n": counts["mut_n"],
        "m": counts["mut_m"],
        "nn": counts["mut_nn"],
    }


def compute_core_pair_counts(
    helper: IsolateSNVHelper,
    sample1: str,
    sample2: str,
    *,
    min_contig_4D_sites: int | None = None,
) -> tuple[dict[str, int], pd.Series, pd.Series]:
    """Return whole-core dN/dS counts plus reusable coverage/diff masks.

    When ``min_contig_4D_sites`` is set, contigs with too few covered 4D sites
    in this pair are masked out of ``covered`` before any counts are computed,
    matching the per-pair contig filter applied by the CP-HMM.
    """
    diff = helper.compute_pairwise_snvs(sample1, sample2)
    covered = helper.compute_pairwise_coverage(sample1, sample2)
    if min_contig_4D_sites:
        keep = short_contig_site_mask(helper, covered, min_contig_4D_sites)
        covered = covered & keep
        keep_snv = keep.reindex(helper.snv_index, fill_value=False)
        diff = diff & keep_snv
    core_site_mask = getattr(
        helper,
        "core_site_mask",
        pd.Series(True, index=helper.index, name="core_site"),
    )
    core_snv_mask = core_site_mask.reindex(helper.snv_index, fill_value=False)
    row = {
        "core_len": int((covered & core_site_mask).sum()),
        "core_diff": int((diff & core_snv_mask).sum()),
        "core_len_4D": int((covered & helper.core_4D).sum()),
        "core_len_1D": int((covered & helper.core_1D).sum()),
        "core_diff_4D": int((diff & helper.snv_4D).sum()),
        "core_diff_1D": int((diff & helper.snv_1D).sum()),
    }
    return row, covered, diff


def add_region_counts(
    row: dict,
    helper: IsolateSNVHelper,
    covered: pd.Series,
    diff: pd.Series,
    region: pd.Series,
    prefix: str,
) -> None:
    """Add length, SNV, opportunity, and 1D mutation counts for one region."""
    region_snv = region.reindex(helper.snv_index, fill_value=False)
    row[f"{prefix}_len_4D"] = int((covered & region & helper.core_4D).sum())
    row[f"{prefix}_len_1D"] = int((covered & region & helper.core_1D).sum())
    row[f"{prefix}_diff_4D"] = int((diff & region_snv & helper.snv_4D).sum())
    row[f"{prefix}_diff_1D"] = int((diff & region_snv & helper.snv_1D).sum())

    opportunities = _region_opportunities(helper, covered, region)
    mutations = _region_mutations(helper, diff, region)
    for key in ["n", "m", "nn"]:
        row[f"{prefix}_{key}"] = opportunities[key]
        row[f"{prefix}_mut_{key}"] = mutations[key]


def compute_close_pair_rows(
    helper: IsolateSNVHelper,
    *,
    min_contig_4D_sites: int | None = None,
) -> pd.DataFrame:
    pairs = helper.recombination_pairs
    if pairs.empty:
        return pd.DataFrame(columns=CLOSE_COLUMNS)
    rows = []
    for pair_row in pairs[pairs["event_count"] > 0].itertuples(index=False):
        sample1, sample2 = _sample_pair_id(pair_row.sample_1, pair_row.sample_2)
        row, covered, diff = compute_core_pair_counts(
            helper, sample1, sample2, min_contig_4D_sites=min_contig_4D_sites,
        )
        row.update(
            {
                "species_name": helper.species,
                "sample 1": sample1,
                "sample 2": sample2,
                "pair_class": "close_recombination",
                "event_count": int(pair_row.event_count),
                "dedup_event_count": int(pair_row.dedup_event_count),
                "clonal_divergence": float(pair_row.clonal_divergence),
                "clonal_fraction": float(pair_row.clonal_fraction),
            }
        )
        recomb = helper.compute_pair_recombination_mask(sample1, sample2)
        add_region_counts(row, helper, covered, diff, recomb, "recomb")
        add_region_counts(row, helper, covered, diff, ~recomb, "clonal")
        rows.append(row)
    return pd.DataFrame(rows, columns=CLOSE_COLUMNS)


def compute_clonal_pair_rows(
    helper: IsolateSNVHelper,
    *,
    min_contig_4D_sites: int | None = None,
) -> pd.DataFrame:
    pairs = helper.recombination_pairs
    if pairs.empty:
        return pd.DataFrame(columns=CLONAL_COLUMNS)
    rows = []
    for pair_row in pairs[pairs["event_count"] == 0].itertuples(index=False):
        sample1, sample2 = _sample_pair_id(pair_row.sample_1, pair_row.sample_2)
        row, covered, diff = compute_core_pair_counts(
            helper, sample1, sample2, min_contig_4D_sites=min_contig_4D_sites,
        )
        row.update(
            {
                "species_name": helper.species,
                "sample 1": sample1,
                "sample 2": sample2,
                "pair_class": "close_no_recombination",
                "event_count": int(pair_row.event_count),
                "dedup_event_count": int(pair_row.dedup_event_count),
                "clonal_divergence": float(pair_row.clonal_divergence),
                "clonal_fraction": float(pair_row.clonal_fraction),
            }
        )
        all_sites = pd.Series(True, index=helper.index)
        opportunities = _region_opportunities(helper, covered, all_sites)
        mutations = _region_mutations(helper, diff, all_sites)
        for key in ["n", "m", "nn"]:
            row[f"clonal_{key}"] = opportunities[key]
            row[f"clonal_mut_{key}"] = mutations[key]
        rows.append(row)
    return pd.DataFrame(rows, columns=CLONAL_COLUMNS)


def load_identical_fraction_matrix(
    helper: IsolateSNVHelper,
    identical_fraction_root: Path | str = DEFAULT_IDENTICAL_FRACTION_ROOT,
) -> pd.DataFrame:
    path = Path(identical_fraction_root) / f"{helper.species}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Identical fraction matrix not found: {path}")
    mat = pd.read_csv(path, header=None)
    if mat.shape != (len(helper.samples), len(helper.samples)):
        raise ValueError(
            f"{path} has shape {mat.shape}, expected {(len(helper.samples), len(helper.samples))}"
        )
    mat.index = helper.samples.astype(str)
    mat.columns = helper.samples.astype(str)
    return mat


def sample_typical_pairs(
    helper: IsolateSNVHelper,
    *,
    identical_fraction_root: Path | str = DEFAULT_IDENTICAL_FRACTION_ROOT,
    threshold: float = 0.05,
    num_pairs: int = 20,
    seed: int = 0,
    exclude_samples: "set[str] | None" = None,
) -> list[tuple[str, str, float]]:
    """Sample ``num_pairs`` typical (fully recombined) pairs (identical_fraction <= threshold).

    Prefers the long-form ``identical_fraction.parquet`` cache beside the SNV
    table (built by ``compute_identical_fraction.py``); falls back to the legacy
    square-matrix CSV at ``identical_fraction_root`` if the long-form cache is
    absent. ``exclude_samples`` drops any pair touching those samples (used to
    keep a misclassified genome out of the typical-pair pool).
    """
    exclude = {str(s) for s in (exclude_samples or set())}
    longform = (
        getattr(helper, "table_paths", None) is not None
        and helper.table_paths.identical_fraction.exists()
    )
    if longform:
        df = helper.identical_fraction
        sub = df[df["identical_fraction"] <= float(threshold)]
        pairs = [
            (str(r.sample_1), str(r.sample_2), float(r.identical_fraction))
            for r in sub.itertuples(index=False)
            if str(r.sample_1) not in exclude and str(r.sample_2) not in exclude
        ]
        if not pairs:
            return []
        rng = np.random.default_rng(seed)
        if len(pairs) > num_pairs:
            idx = rng.choice(len(pairs), size=num_pairs, replace=False)
            pairs = [pairs[i] for i in sorted(idx)]
        return pairs

    matrix = load_identical_fraction_matrix(helper, identical_fraction_root)
    values = matrix.to_numpy(dtype=float)
    upper = np.triu_indices(values.shape[0], k=1)
    eligible = np.flatnonzero(values[upper] <= threshold)
    if len(eligible) == 0:
        return []
    rng = np.random.default_rng(seed)
    if len(eligible) > num_pairs:
        eligible = rng.choice(eligible, size=num_pairs, replace=False)
    samples = matrix.index.to_numpy(dtype=str)
    pairs = []
    for idx in eligible:
        i = int(upper[0][idx])
        j = int(upper[1][idx])
        a, b = samples[i], samples[j]
        if a in exclude or b in exclude:
            continue
        pairs.append((a, b, float(values[i, j])))
    return pairs


def compute_typical_pair_rows(
    helper: IsolateSNVHelper,
    *,
    identical_fraction_root: Path | str = DEFAULT_IDENTICAL_FRACTION_ROOT,
    threshold: float = 0.05,
    num_pairs: int = 20,
    seed: int = 0,
    min_contig_4D_sites: int | None = None,
    exclude_samples: "set[str] | None" = None,
) -> pd.DataFrame:
    rows = []
    for sample1, sample2, identical_fraction in sample_typical_pairs(
        helper,
        identical_fraction_root=identical_fraction_root,
        threshold=threshold,
        num_pairs=num_pairs,
        seed=seed,
        exclude_samples=exclude_samples,
    ):
        row, covered, diff = compute_core_pair_counts(
            helper, sample1, sample2, min_contig_4D_sites=min_contig_4D_sites,
        )
        row.update(
            {
                "species_name": helper.species,
                "sample 1": sample1,
                "sample 2": sample2,
                "pair_class": "typical_fully_recombined",
                "identical_block_fraction": identical_fraction,
            }
        )
        all_sites = pd.Series(True, index=helper.index)
        opportunities = _region_opportunities(helper, covered, all_sites)
        mutations = _region_mutations(helper, diff, all_sites)
        for key in ["n", "m", "nn"]:
            row[f"core_{key}"] = opportunities[key]
            row[f"core_mut_{key}"] = mutations[key]
        rows.append(row)
    return pd.DataFrame(rows, columns=TYPICAL_COLUMNS)


def compute_isolate_dnds(
    accession: str,
    *,
    table_root: Path | str = DEFAULT_SNV_TABLE_ROOT,
    recombination_root: Path | str | None = None,
    output_root: Path | str | None = None,
    identical_fraction_root: Path | str = DEFAULT_IDENTICAL_FRACTION_ROOT,
    typical_threshold: float = 0.05,
    typical_pairs: int = 20,
    seed: int = 0,
    min_contig_4D_sites: int | None = None,
    exclude_samples: "set[str] | None" = None,
) -> DndsResult:
    table_root = Path(table_root)
    if recombination_root is None:
        recombination_root = table_root
    if output_root is None:
        output_dir = table_root / accession / "dnds"
    else:
        # Parity with the default layout so downstream tools (e.g. the
        # stratification grid) discover files via the same glob.
        output_dir = Path(output_root) / accession / "dnds"
    output_dir.mkdir(parents=True, exist_ok=True)

    helper = IsolateSNVHelper(
        accession,
        table_root=table_root,
        recombination_root=recombination_root,
        source="tables",
        compute_bi_snvs=False,
        annotate=True,
    )

    close_df = compute_close_pair_rows(helper, min_contig_4D_sites=min_contig_4D_sites)
    clonal_df = compute_clonal_pair_rows(helper, min_contig_4D_sites=min_contig_4D_sites)
    typical_df = compute_typical_pair_rows(
        helper,
        identical_fraction_root=identical_fraction_root,
        threshold=typical_threshold,
        num_pairs=typical_pairs,
        seed=seed,
        min_contig_4D_sites=min_contig_4D_sites,
        exclude_samples=exclude_samples,
    )

    close_path = output_dir / "close_pairs_with_recombination.csv"
    clonal_path = output_dir / "clonal_pairs_no_recombination.csv"
    typical_path = output_dir / "typical_pairs_fully_recombined.csv"
    summary_path = output_dir / "dnds_summary.json"
    close_df.to_csv(close_path, index=False)
    clonal_df.to_csv(clonal_path, index=False)
    typical_df.to_csv(typical_path, index=False)

    summary = {
        "accession": accession,
        "num_samples": int(len(helper.samples)),
        "num_coverage_sites": int(helper.num_sites),
        "num_core_sites": int(helper.core_site_mask.sum()),
        "num_core_1D_sites": int(helper.core_1D.sum()),
        "num_core_4D_sites": int(helper.core_4D.sum()),
        "num_biallelic_snvs": int(len(helper.snv_index)),
        "close_pairs_with_recombination": int(close_df.shape[0]),
        "clonal_pairs_no_recombination": int(clonal_df.shape[0]),
        "typical_pairs_fully_recombined": int(typical_df.shape[0]),
        "typical_threshold": typical_threshold,
        "typical_seed": seed,
        "min_contig_4D_sites": int(min_contig_4D_sites) if min_contig_4D_sites else None,
        "outputs": {
            "close_pairs_with_recombination": str(close_path),
            "clonal_pairs_no_recombination": str(clonal_path),
            "typical_pairs_fully_recombined": str(typical_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return DndsResult(
        accession=accession,
        output_dir=output_dir,
        close_path=close_path,
        clonal_path=clonal_path,
        typical_path=typical_path,
        summary_path=summary_path,
        summary=summary,
    )
