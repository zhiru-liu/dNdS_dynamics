"""Shared helpers for LiuGood-style SNV tables.

The table contract mirrors the layout used by ``LiuGood2024_data``:

- ``snv_catalog``: one row per SNV coordinate, sample columns are boolean
  non-reference genotype states, plus ``Contig`` and ``Location``.
- ``alleles``: same SNV coordinates, sample columns are base calls, plus
  ``Contig``, ``Location``, and ``Ref``.
- ``coverage``: coordinate table with boolean sample coverage calls, plus
  ``Contig`` and ``Location``.

``compute_biallelic_snvs`` intentionally follows the published helper's allele
polarization rules so that downstream dN/dS code sees ``0=Major``, ``1=Alt``,
and ``255=missing``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


INDEX_COLUMNS = ["Contig", "Location"]
ALLELE_COLUMNS = ["Ref", "Major", "Alt"]
SUPPORTED_FORMATS = {"feather", "parquet"}


def decode_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Decode byte-string object columns produced by old pyarrow/pandas stacks."""
    for col in df.columns:
        if df[col].dtype != "O" or df.empty:
            continue
        first_valid = df[col].dropna().head(1)
        if first_valid.empty:
            continue
        first = first_valid.iloc[0]
        if isinstance(first, bytes):
            df[col] = df[col].str.decode("utf-8")
    return df


def load_snv_dataframe(path: Path | str, format: str | None = None) -> pd.DataFrame:
    """Load a SNV table and set ``(Contig, Location)`` as the index."""
    path = Path(path)
    table_format = format or path.suffix.lstrip(".")
    if table_format not in SUPPORTED_FORMATS:
        raise ValueError(f"format must be one of {sorted(SUPPORTED_FORMATS)}")

    if table_format == "feather":
        df = pd.read_feather(path)
    else:
        try:
            df = pd.read_parquet(path)
        except (NotImplementedError, TypeError, ValueError):
            import pyarrow.parquet as pq

            table = pq.read_table(path).replace_schema_metadata(None)
            df = table.to_pandas()

    df = decode_object_columns(df)
    missing = [col for col in INDEX_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing index columns: {missing}")
    df.set_index(INDEX_COLUMNS, inplace=True)
    return df


def save_snv_dataframe(df: pd.DataFrame, path: Path | str, format: str | None = None) -> None:
    """Save a SNV table with index columns restored as ordinary columns."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table_format = format or path.suffix.lstrip(".")
    if table_format not in SUPPORTED_FORMATS:
        raise ValueError(f"format must be one of {sorted(SUPPORTED_FORMATS)}")

    out = df.reset_index()
    if table_format == "feather":
        out.to_feather(path)
    else:
        out.to_parquet(path, index=False)


def sample_columns(df: pd.DataFrame) -> list[str]:
    """Return sample columns, excluding known allele metadata columns."""
    return [col for col in df.columns if col not in ALLELE_COLUMNS]


def compute_biallelic_snvs(
    snvs: pd.DataFrame,
    alleles: pd.DataFrame,
    coverage: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Compute polarized biallelic SNVs from raw SNV, allele, and coverage tables.

    Parameters
    ----------
    snvs
        Boolean non-reference genotype table indexed by ``(Contig, Location)``.
    alleles
        Base-call table with the same index as ``snvs`` and a ``Ref`` column.
    coverage
        Boolean coverage table indexed by ``(Contig, Location)``.

    Returns
    -------
    bi_snvs
        Biallelic genotype table. Sample columns are ``0=Major``, ``1=Alt``,
        and ``255=missing``; metadata columns are ``Ref``, ``Major``, ``Alt``.
    multi_sites
        Boolean series over the input SNV coordinates marking sites with more
        than two observed alleles among covered samples.
    """
    snvs = snvs.copy()
    alleles = alleles.copy()
    coverage_at_snvs = coverage.loc[snvs.index, snvs.columns]
    refs = alleles.pop("Ref")
    alleles = alleles.loc[:, snvs.columns]

    snv_mask = snvs & coverage_at_snvs
    alleles.where(snv_mask, refs, axis=0, inplace=True)
    alleles.where(coverage_at_snvs, inplace=True)

    allele_counts = alleles.apply(lambda row: row.value_counts(), axis=1)
    observed_allele_counts = allele_counts.notna().sum(axis=1)
    mono_sites = observed_allele_counts == 1
    bi_sites = observed_allele_counts == 2
    multi_sites = observed_allele_counts > 2

    alt_alleles = allele_counts.idxmin(axis=1)
    major_alleles = allele_counts.idxmax(axis=1)
    true_major_alleles = refs.copy()
    true_alt_alleles = alt_alleles.copy()

    true_major_alleles[mono_sites] = major_alleles[mono_sites]
    true_alt_alleles[mono_sites] = refs[mono_sites]
    true_major_alleles[bi_sites] = major_alleles[bi_sites]

    tie_mask = bi_sites & (allele_counts.min(axis=1) == allele_counts.max(axis=1))
    for loc in tie_mask[tie_mask].index:
        tie_alleles = allele_counts.loc[loc].dropna().index.tolist()
        if refs[loc] in tie_alleles:
            tie_alleles.remove(refs[loc])
            true_major_alleles[loc] = refs[loc]
            true_alt_alleles[loc] = tie_alleles[0]
        else:
            true_major_alleles[loc] = tie_alleles[1]
            true_alt_alleles[loc] = tie_alleles[0]

    bi_mask = ~multi_sites
    bi_snvs = snvs.loc[bi_mask].copy().astype(np.uint8)
    bi_ref = refs.loc[bi_mask]
    bi_major = true_major_alleles.loc[bi_mask]
    bi_alt = true_alt_alleles.loc[bi_mask]

    polarize_mask = bi_ref != bi_major
    bi_snvs.loc[polarize_mask] = 1 - bi_snvs.loc[polarize_mask]
    bi_snvs.where(coverage_at_snvs.loc[bi_snvs.index], other=255, inplace=True)

    bi_snvs["Ref"] = bi_ref
    bi_snvs["Major"] = bi_major
    bi_snvs["Alt"] = bi_alt
    return bi_snvs, multi_sites.astype(bool)
