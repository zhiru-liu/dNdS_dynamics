"""dN/dS counting for LiuGood2024 SNVs with refactored CP-HMM events."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config


DEFAULT_IDENTICAL_FRACTION_ROOT = config.LIUGOOD_CF_BETWEEN_HOSTS
DEFAULT_PUBLISHED_DATA_ROOT = config.DNDS_DYNAMICS_DATA
DEFAULT_BF_CPHMM_RESULTS_DIR = config.CPHMM_BF_RESULTS
DEFAULT_SPECIES = "Bacteroides_fragilis_54507"

# Mirrors close_pair_hmm/cphmm/config.py: HMM_BLOCK_SIZE * HMM_MIN_SEQ_LEN.
# Contigs with fewer covered 4D sites than this in a given pair are skipped by
# the CP-HMM, so the recombination calls don't cover them. Passing this
# threshold to the count functions matches that filter.
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
    "naive_divergence",
    "clonal_fraction",
    "cphmm_genome_len_4D",
    "cphmm_clonal_len_blocks",
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
class LiuGoodCphmmDndsResult:
    species: str
    output_dir: Path
    close_path: Path
    clonal_path: Path
    typical_path: Path
    comparison_path: Path
    summary_path: Path
    summary: dict


@dataclass
class ReferenceIndexLookup:
    """Fast interval-to-mask helper for a sorted-ish (Contig, Location) index."""

    index: pd.MultiIndex
    contig_positions: dict[str, np.ndarray]
    contig_locations: dict[str, np.ndarray]

    @classmethod
    def from_index(cls, index: pd.MultiIndex) -> "ReferenceIndexLookup":
        contigs = index.get_level_values("Contig").astype(str).to_numpy()
        locations = index.get_level_values("Location").to_numpy(dtype=np.int64)
        contig_positions: dict[str, np.ndarray] = {}
        contig_locations: dict[str, np.ndarray] = {}
        for contig in pd.unique(contigs):
            positions = np.flatnonzero(contigs == contig)
            locs = locations[positions]
            order = np.argsort(locs, kind="mergesort")
            contig_positions[str(contig)] = positions[order]
            contig_locations[str(contig)] = locs[order]
        return cls(index=index, contig_positions=contig_positions, contig_locations=contig_locations)

    def interval_mask(self, events: pd.DataFrame) -> pd.Series:
        return pd.Series(self.interval_mask_array(events), index=self.index)

    def interval_mask_array(self, events: pd.DataFrame) -> np.ndarray:
        mask = np.zeros(len(self.index), dtype=bool)
        if events.empty:
            return mask
        for row in events.itertuples(index=False):
            contig = str(getattr(row, "contig"))
            if contig not in self.contig_locations:
                continue
            start_site = int(getattr(row, "start_site"))
            end_site = int(getattr(row, "end_site"))
            start = min(start_site, end_site)
            end = max(start_site, end_site)
            locs = self.contig_locations[contig]
            positions = self.contig_positions[contig]
            left = np.searchsorted(locs, start, side="left")
            right = np.searchsorted(locs, end, side="right")
            if left < right:
                mask[positions[left:right]] = True
        return mask


class PositionalDndsCounter:
    """NumPy-backed pair counter for LiuGood SNVHelper objects."""

    count_keys = [
        "len_4D",
        "len_1D",
        "diff_4D",
        "diff_1D",
        "n",
        "m",
        "nn",
        "mut_n",
        "mut_m",
        "mut_nn",
    ]

    def __init__(self, helper):
        self.helper = helper
        self.samples = helper.samples.astype(str).to_numpy()
        self.sample_to_index = {sample: idx for idx, sample in enumerate(self.samples)}
        self.coverage = helper.coverage.loc[:, helper.samples].to_numpy(dtype=bool, copy=False)
        self.snvs = helper.snvs.loc[:, helper.samples].to_numpy(dtype=np.uint8, copy=False)
        self.core_1d = helper.core_1D.to_numpy(dtype=bool, copy=False)
        self.core_4d = helper.core_4D.to_numpy(dtype=bool, copy=False)
        self.core_to_snvs = np.asarray(helper.core_to_snvs, dtype=bool)
        self.snv_site_positions = np.flatnonzero(self.core_to_snvs)
        if not helper.coverage.index[self.core_to_snvs].equals(helper.snvs.index):
            self.snv_site_positions = helper.coverage.index.get_indexer(helper.snvs.index)
            if (self.snv_site_positions < 0).any():
                raise ValueError("Some SNV sites are absent from the coverage index")
        self.snv_1d = self.core_1d[self.snv_site_positions]
        self.snv_4d = self.core_4d[self.snv_site_positions]
        self.mut_opportunity = helper.mut_df[["n", "m", "nn"]].to_numpy(dtype=np.int64, copy=False)
        self.snv_type_codes = (
            helper.snv_types.map({"n": 0, "m": 1, "nn": 2}).fillna(-1).to_numpy(dtype=np.int8)
        )
        self.all_sites = np.ones(helper.coverage.shape[0], dtype=bool)
        # Per-site integer contig code, used for the optional CP-HMM contig filter.
        contig_values = helper.coverage.index.get_level_values("Contig").astype(str).to_numpy()
        unique_contigs, self.contig_codes = np.unique(contig_values, return_inverse=True)
        self.contig_codes = self.contig_codes.astype(np.int32, copy=False)
        self.n_contigs = int(unique_contigs.size)

    def short_contig_keep_array(
        self,
        covered: np.ndarray,
        min_contig_4D_sites: int,
    ) -> np.ndarray:
        """Boolean site mask that drops contigs with too few covered 4D sites.

        Reproduces the per-pair contig filter applied by the CP-HMM
        (cphmm/recomb_inference.py:170): a contig is dropped if it has fewer
        than ``min_contig_4D_sites`` covered 4D sites in this pair.
        """
        per_contig = np.bincount(
            self.contig_codes,
            weights=(covered & self.core_4d).astype(np.int64),
            minlength=self.n_contigs,
        )
        keep_contigs = per_contig >= int(min_contig_4D_sites)
        return keep_contigs[self.contig_codes]

    def pair_arrays(self, sample1: str, sample2: str) -> tuple[np.ndarray, np.ndarray]:
        idx1 = self.sample_to_index[str(sample1)]
        idx2 = self.sample_to_index[str(sample2)]
        covered = self.coverage[:, idx1] & self.coverage[:, idx2]
        sample1_snvs = self.snvs[:, idx1]
        sample2_snvs = self.snvs[:, idx2]
        diff = (sample1_snvs != sample2_snvs) & (sample1_snvs != 255) & (sample2_snvs != 255)
        return covered, diff

    def count_region(
        self,
        covered: np.ndarray,
        diff: np.ndarray,
        region: np.ndarray,
    ) -> dict[str, int]:
        site_mask = covered & region
        snv_region = region[self.snv_site_positions]
        snv_mask = diff & snv_region

        opportunity_mask = site_mask & self.core_1d
        if opportunity_mask.any():
            opportunities = self.mut_opportunity[opportunity_mask].sum(axis=0)
        else:
            opportunities = np.zeros(3, dtype=np.int64)

        mutation_mask = snv_mask & self.snv_1d
        return {
            "len_4D": int(np.count_nonzero(site_mask & self.core_4d)),
            "len_1D": int(np.count_nonzero(opportunity_mask)),
            "diff_4D": int(np.count_nonzero(snv_mask & self.snv_4d)),
            "diff_1D": int(np.count_nonzero(mutation_mask)),
            "n": int(opportunities[0]),
            "m": int(opportunities[1]),
            "nn": int(opportunities[2]),
            "mut_n": int(np.count_nonzero(mutation_mask & (self.snv_type_codes == 0))),
            "mut_m": int(np.count_nonzero(mutation_mask & (self.snv_type_codes == 1))),
            "mut_nn": int(np.count_nonzero(mutation_mask & (self.snv_type_codes == 2))),
        }

    def core_counts(
        self,
        sample1: str,
        sample2: str,
        *,
        min_contig_4D_sites: int | None = None,
    ) -> tuple[dict[str, int], np.ndarray, np.ndarray, dict[str, int]]:
        covered, diff = self.pair_arrays(sample1, sample2)
        if min_contig_4D_sites:
            keep = self.short_contig_keep_array(covered, min_contig_4D_sites)
            covered = covered & keep
            diff = diff & keep[self.snv_site_positions]
        total = self.count_region(covered, diff, self.all_sites)
        row = {
            "core_len": int(np.count_nonzero(covered)),
            "core_diff": int(np.count_nonzero(diff)),
            "core_len_4D": total["len_4D"],
            "core_len_1D": total["len_1D"],
            "core_diff_4D": total["diff_4D"],
            "core_diff_1D": total["diff_1D"],
        }
        return row, covered, diff, total

    @classmethod
    def subtract_counts(cls, total: dict[str, int], part: dict[str, int]) -> dict[str, int]:
        return {key: int(total[key] - part[key]) for key in cls.count_keys}

    @staticmethod
    def add_prefixed_counts(row: dict, prefix: str, counts: dict[str, int]) -> None:
        row[f"{prefix}_len_4D"] = counts["len_4D"]
        row[f"{prefix}_len_1D"] = counts["len_1D"]
        row[f"{prefix}_diff_4D"] = counts["diff_4D"]
        row[f"{prefix}_diff_1D"] = counts["diff_1D"]
        for key in ["n", "m", "nn"]:
            row[f"{prefix}_{key}"] = counts[key]
            row[f"{prefix}_mut_{key}"] = counts[f"mut_{key}"]


def load_liugood_snv_helper(
    species: str,
    *,
    snv_repo: Path | str | None = None,  # accepted for backward compat; ignored
    annotate: bool = True,
):
    """Load a QP ``SNVHelper`` for ``species``.

    Backward-compatible wrapper around
    :func:`dnds_dynamics.snv_helpers.qp.load_qp_snv_helper`. The QP helper now
    comes bundled in the editable ``cphmm`` package, so ``snv_repo`` is ignored.
    """
    from ..snv_helpers.qp import load_qp_snv_helper

    return load_qp_snv_helper(species, annotate=annotate)


def _pair_key(sample1: object, sample2: object) -> tuple[str, str]:
    return tuple(sorted((str(sample1), str(sample2))))


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0 or pd.isna(denominator):
        return float("nan")
    return float(numerator) / float(denominator)


def _aggregate_dnds(df: pd.DataFrame, prefix: str) -> dict[str, float]:
    d_s = _safe_div(df[f"{prefix}_diff_4D"].sum(), df[f"{prefix}_len_4D"].sum())
    d_n = _safe_div(df[f"{prefix}_diff_1D"].sum(), df[f"{prefix}_len_1D"].sum())
    return {
        f"{prefix}_dS": d_s,
        f"{prefix}_dN": d_n,
        f"{prefix}_dNdS": _safe_div(d_n, d_s),
    }


def _count_mut_types(values: pd.Series) -> dict[str, int]:
    return {
        "n": int((values == "n").sum()),
        "m": int((values == "m").sum()),
        "nn": int((values == "nn").sum()),
    }


def _region_opportunities(helper, covered: pd.Series, region: pd.Series) -> dict[str, int]:
    mask = covered & region & helper.core_1D
    sums = helper.mut_df.loc[mask, ["n", "m", "nn"]].sum()
    return {key: int(sums[key]) for key in ["n", "m", "nn"]}


def _region_mutations(helper, diff: pd.Series, region: pd.Series) -> dict[str, int]:
    region_snv = region.reindex(helper.snvs.index, fill_value=False)
    mut_types = helper.snv_types[diff & region_snv & helper.snv_1D]
    return _count_mut_types(mut_types)


def compute_core_pair_counts(helper, sample1: str, sample2: str) -> tuple[dict[str, int], pd.Series, pd.Series]:
    diff = helper.compute_pairwise_snvs(sample1, sample2)
    covered = helper.compute_pairwise_coverage(sample1, sample2)
    row = {
        "core_len": int(covered.sum()),
        "core_diff": int(diff.sum()),
        "core_len_4D": int((covered & helper.core_4D).sum()),
        "core_len_1D": int((covered & helper.core_1D).sum()),
        "core_diff_4D": int((diff & helper.snv_4D).sum()),
        "core_diff_1D": int((diff & helper.snv_1D).sum()),
    }
    return row, covered, diff


def add_region_counts(
    row: dict,
    helper,
    covered: pd.Series,
    diff: pd.Series,
    region: pd.Series,
    prefix: str,
) -> None:
    region_snv = region.reindex(helper.snvs.index, fill_value=False)
    row[f"{prefix}_len_4D"] = int((covered & region & helper.core_4D).sum())
    row[f"{prefix}_len_1D"] = int((covered & region & helper.core_1D).sum())
    row[f"{prefix}_diff_4D"] = int((diff & region_snv & helper.snv_4D).sum())
    row[f"{prefix}_diff_1D"] = int((diff & region_snv & helper.snv_1D).sum())

    opportunities = _region_opportunities(helper, covered, region)
    mutations = _region_mutations(helper, diff, region)
    for key in ["n", "m", "nn"]:
        row[f"{prefix}_{key}"] = opportunities[key]
        row[f"{prefix}_mut_{key}"] = mutations[key]


def load_cphmm_tables(
    *,
    species: str = DEFAULT_SPECIES,
    results_dir: Path | str = DEFAULT_BF_CPHMM_RESULTS_DIR,
    suffix: str = "all_pairs",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    results_dir = Path(results_dir)
    pair_path = results_dir / f"{species}__{suffix}__inference_summary.csv"
    transfer_path = results_dir / f"{species}__{suffix}__transfer_summary.csv"
    if not pair_path.exists():
        raise FileNotFoundError(f"Missing CP-HMM pair summary: {pair_path}")
    if not transfer_path.exists():
        raise FileNotFoundError(f"Missing CP-HMM transfer summary: {transfer_path}")
    pair_df = pd.read_csv(pair_path, dtype={"genome1": str, "genome2": str})
    transfer_df = pd.read_csv(transfer_path, dtype={"genome1": str, "genome2": str, "contig": str})
    return pair_df, transfer_df


def compute_cphmm_pair_rows(
    helper,
    pair_df: pd.DataFrame,
    transfer_df: pd.DataFrame,
    *,
    min_contig_4D_sites: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lookup = ReferenceIndexLookup.from_index(helper.coverage.index)
    counter = PositionalDndsCounter(helper)
    events_by_pair = {
        key: group.copy()
        for key, group in transfer_df.groupby(
            transfer_df.apply(lambda row: _pair_key(row["genome1"], row["genome2"]), axis=1),
            sort=False,
        )
    }

    close_rows = []
    clonal_rows = []
    for pair in pair_df.itertuples(index=False):
        sample1 = str(pair.genome1)
        sample2 = str(pair.genome2)
        key = _pair_key(sample1, sample2)
        events = events_by_pair.get(key, pd.DataFrame(columns=transfer_df.columns))
        row, covered, diff, total_counts = counter.core_counts(
            sample1, sample2, min_contig_4D_sites=min_contig_4D_sites,
        )
        row.update(
            {
                "species_name": helper.species_name,
                "sample 1": sample1,
                "sample 2": sample2,
                "event_count": int(events.shape[0]),
                "dedup_event_count": int(events.shape[0]),
                "clonal_divergence": float(pair.est_div),
                "naive_divergence": float(pair.naive_div),
                "cphmm_genome_len_4D": int(pair.genome_len),
                "cphmm_clonal_len_blocks": int(pair.clonal_len),
            }
        )

        if events.empty:
            row["pair_class"] = "close_no_new_recombination"
            row["clonal_fraction"] = 1.0
            counter.add_prefixed_counts(row, "clonal", total_counts)
            clonal_rows.append(row)
            continue

        row["pair_class"] = "close_new_recombination"
        recomb = lookup.interval_mask_array(events)
        recomb_counts = counter.count_region(covered, diff, recomb)
        clonal_counts = counter.subtract_counts(total_counts, recomb_counts)
        counter.add_prefixed_counts(row, "recomb", recomb_counts)
        counter.add_prefixed_counts(row, "clonal", clonal_counts)
        row["clonal_fraction"] = _safe_div(row["clonal_len_4D"], row["core_len_4D"])
        close_rows.append(row)

    return (
        pd.DataFrame(close_rows, columns=CLOSE_COLUMNS),
        pd.DataFrame(clonal_rows, columns=[c for c in CLOSE_COLUMNS if not c.startswith("recomb_")]),
    )


def load_identical_fraction_matrix(
    helper,
    identical_fraction_root: Path | str = DEFAULT_IDENTICAL_FRACTION_ROOT,
) -> pd.DataFrame:
    path = Path(identical_fraction_root) / f"{helper.species_name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Identical fraction matrix not found: {path}")
    matrix = pd.read_csv(path, header=None)
    if matrix.shape != (len(helper.samples), len(helper.samples)):
        raise ValueError(
            f"{path} has shape {matrix.shape}, expected {(len(helper.samples), len(helper.samples))}"
        )
    matrix.index = helper.samples.astype(str)
    matrix.columns = helper.samples.astype(str)
    return matrix


def sample_typical_pairs(
    helper,
    *,
    identical_fraction_root: Path | str = DEFAULT_IDENTICAL_FRACTION_ROOT,
    threshold: float = 0.05,
    num_pairs: int = 20,
    seed: int = 0,
) -> list[tuple[str, str, float]]:
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
        pairs.append((samples[i], samples[j], float(values[i, j])))
    return pairs


def compute_typical_pair_rows(
    helper,
    *,
    identical_fraction_root: Path | str = DEFAULT_IDENTICAL_FRACTION_ROOT,
    threshold: float = 0.05,
    num_pairs: int = 20,
    seed: int = 0,
    min_contig_4D_sites: int | None = None,
) -> pd.DataFrame:
    rows = []
    counter = PositionalDndsCounter(helper)
    for sample1, sample2, identical_fraction in sample_typical_pairs(
        helper,
        identical_fraction_root=identical_fraction_root,
        threshold=threshold,
        num_pairs=num_pairs,
        seed=seed,
    ):
        row, _, _, total_counts = counter.core_counts(
            sample1, sample2, min_contig_4D_sites=min_contig_4D_sites,
        )
        row.update(
            {
                "species_name": helper.species_name,
                "sample 1": sample1,
                "sample 2": sample2,
                "pair_class": "typical_fully_recombined",
                "identical_block_fraction": identical_fraction,
            }
        )
        for key in ["n", "m", "nn"]:
            row[f"core_{key}"] = total_counts[key]
            row[f"core_mut_{key}"] = total_counts[f"mut_{key}"]
        rows.append(row)
    return pd.DataFrame(rows, columns=TYPICAL_COLUMNS)


def compare_to_published_close_dnds(
    new_close_df: pd.DataFrame,
    *,
    species: str,
    published_data_root: Path | str = DEFAULT_PUBLISHED_DATA_ROOT,
) -> pd.DataFrame:
    old_path = Path(published_data_root) / "gut_microbiome_close_pair_dNdS" / f"{species}.csv"
    if not old_path.exists() or new_close_df.empty:
        return pd.DataFrame()
    old = pd.read_csv(old_path, dtype={"sample 1": str, "sample 2": str})
    key_cols = ["species_name", "sample 1", "sample 2"]
    compare_cols = [
        "core_diff_4D",
        "core_diff_1D",
        "recomb_len_4D",
        "recomb_len_1D",
        "recomb_diff_4D",
        "recomb_diff_1D",
        "clonal_len_4D",
        "clonal_len_1D",
        "clonal_diff_4D",
        "clonal_diff_1D",
    ]
    merged = new_close_df[key_cols + ["event_count"] + compare_cols].merge(
        old[key_cols + compare_cols],
        on=key_cols,
        how="outer",
        suffixes=("_new_cphmm", "_published"),
        indicator=True,
    )
    for col in compare_cols:
        new_col = f"{col}_new_cphmm"
        old_col = f"{col}_published"
        if new_col in merged and old_col in merged:
            merged[f"{col}_delta"] = merged[new_col] - merged[old_col]
    return merged


def summarize_outputs(
    helper,
    close_df: pd.DataFrame,
    clonal_df: pd.DataFrame,
    typical_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    transfer_df: pd.DataFrame,
) -> dict:
    summary: dict = {
        "species": helper.species_name,
        "num_samples": int(len(helper.samples)),
        "num_coverage_sites": int(helper.coverage.shape[0]),
        "num_core_1D_sites": int(helper.core_1D.sum()),
        "num_core_4D_sites": int(helper.core_4D.sum()),
        "num_biallelic_snvs": int(helper.snvs.shape[0]),
        "new_cphmm_pairs": int(close_df.shape[0] + clonal_df.shape[0]),
        "new_cphmm_pairs_with_events": int(close_df.shape[0]),
        "new_cphmm_pairs_without_events": int(clonal_df.shape[0]),
        "new_cphmm_events": int(transfer_df.shape[0]),
        "typical_pairs": int(typical_df.shape[0]),
    }
    if not close_df.empty:
        summary["new_cphmm_close_aggregate"] = {
            **_aggregate_dnds(close_df, "core"),
            **_aggregate_dnds(close_df, "recomb"),
            **_aggregate_dnds(close_df, "clonal"),
            "total_recomb_len_4D": int(close_df["recomb_len_4D"].sum()),
            "total_recomb_diff_4D": int(close_df["recomb_diff_4D"].sum()),
            "total_recomb_diff_1D": int(close_df["recomb_diff_1D"].sum()),
        }
    if not typical_df.empty:
        d_s = _safe_div(typical_df["core_diff_4D"].sum(), typical_df["core_len_4D"].sum())
        d_n = _safe_div(typical_df["core_diff_1D"].sum(), typical_df["core_len_1D"].sum())
        summary["typical_aggregate"] = {
            "core_dS": d_s,
            "core_dN": d_n,
            "core_dNdS": _safe_div(d_n, d_s),
        }
    if not comparison_df.empty:
        shared = comparison_df[comparison_df["_merge"] == "both"]
        summary["published_comparison"] = {
            "published_shared_pairs": int(shared.shape[0]),
            "mean_recomb_len_4D_delta": float(shared["recomb_len_4D_delta"].mean()),
            "median_recomb_len_4D_delta": float(shared["recomb_len_4D_delta"].median()),
            "total_recomb_len_4D_new_cphmm": int(shared["recomb_len_4D_new_cphmm"].sum()),
            "total_recomb_len_4D_published": int(shared["recomb_len_4D_published"].sum()),
            "total_recomb_diff_4D_new_cphmm": int(shared["recomb_diff_4D_new_cphmm"].sum()),
            "total_recomb_diff_4D_published": int(shared["recomb_diff_4D_published"].sum()),
            "total_recomb_diff_1D_new_cphmm": int(shared["recomb_diff_1D_new_cphmm"].sum()),
            "total_recomb_diff_1D_published": int(shared["recomb_diff_1D_published"].sum()),
        }
    return summary


def compute_liugood_cphmm_dnds(
    *,
    species: str = DEFAULT_SPECIES,
    results_dir: Path | str = DEFAULT_BF_CPHMM_RESULTS_DIR,
    suffix: str = "all_pairs",
    output_dir: Path | str,
    snv_repo: Path | str | None = None,  # accepted for backward compat; ignored
    identical_fraction_root: Path | str = DEFAULT_IDENTICAL_FRACTION_ROOT,
    typical_threshold: float = 0.05,
    typical_pairs: int = 20,
    seed: int = 0,
    published_data_root: Path | str = DEFAULT_PUBLISHED_DATA_ROOT,
    min_contig_4D_sites: int | None = None,
) -> LiuGoodCphmmDndsResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pair_df, transfer_df = load_cphmm_tables(species=species, results_dir=results_dir, suffix=suffix)
    helper = load_liugood_snv_helper(species, snv_repo=snv_repo, annotate=True)
    close_df, clonal_df = compute_cphmm_pair_rows(
        helper, pair_df, transfer_df, min_contig_4D_sites=min_contig_4D_sites,
    )
    typical_df = compute_typical_pair_rows(
        helper,
        identical_fraction_root=identical_fraction_root,
        threshold=typical_threshold,
        num_pairs=typical_pairs,
        seed=seed,
        min_contig_4D_sites=min_contig_4D_sites,
    )
    comparison_df = compare_to_published_close_dnds(
        close_df,
        species=species,
        published_data_root=published_data_root,
    )

    close_path = output_dir / "close_pairs_with_new_cphmm_recombination.csv"
    clonal_path = output_dir / "close_pairs_no_new_cphmm_recombination.csv"
    typical_path = output_dir / "typical_pairs_fully_recombined.csv"
    comparison_path = output_dir / "published_vs_new_cphmm_pair_comparison.csv"
    summary_path = output_dir / "dnds_summary.json"

    close_df.to_csv(close_path, index=False)
    clonal_df.to_csv(clonal_path, index=False)
    typical_df.to_csv(typical_path, index=False)
    comparison_df.to_csv(comparison_path, index=False)

    summary = summarize_outputs(helper, close_df, clonal_df, typical_df, comparison_df, transfer_df)
    summary["inputs"] = {
        "results_dir": str(results_dir),
        "suffix": suffix,
        "snv_repo": str(snv_repo),
        "identical_fraction_root": str(identical_fraction_root),
        "published_data_root": str(published_data_root),
        "min_contig_4D_sites": int(min_contig_4D_sites) if min_contig_4D_sites else None,
    }
    summary["outputs"] = {
        "close_pairs_with_new_cphmm_recombination": str(close_path),
        "close_pairs_no_new_cphmm_recombination": str(clonal_path),
        "typical_pairs_fully_recombined": str(typical_path),
        "published_vs_new_cphmm_pair_comparison": str(comparison_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    return LiuGoodCphmmDndsResult(
        species=species,
        output_dir=output_dir,
        close_path=close_path,
        clonal_path=clonal_path,
        typical_path=typical_path,
        comparison_path=comparison_path,
        summary_path=summary_path,
        summary=summary,
    )
