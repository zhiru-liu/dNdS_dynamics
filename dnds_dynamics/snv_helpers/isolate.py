"""Adapter for UHGG isolate-genome SNV tables.

The existing dNdS code works with an SNVHelper-like object.  The preferred input
for this helper is now a LiuGood-style table directory generated directly from
raw UHGG SNV archives:

- ``snv_catalog.parquet``
- ``alleles.parquet``
- ``coverage.parquet``
- optional ``biallelic_snvs.parquet``

The deprecated DH ``.npy`` adapter is kept as a fallback so older smoke tests
and comparisons still run while the new table build path matures.

Mutation-type breakdowns require a future ``site_annotations`` table.  The new
SNV/allele tables do carry ``Ref``, ``Major``, and ``Alt`` via the biallelic
derivation, so they remove the allele-identity blocker from the old DH arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ..recombination.cache import (
    DEFAULT_RECOMBINATION_CACHE_ROOT,
    RecombinationCachePaths,
    events_for_pair,
    load_recombination_events,
    load_recombination_pairs,
    recombination_mask_from_events,
)
from .table_contract import (
    compute_biallelic_snvs,
    load_snv_dataframe,
    sample_columns as table_sample_columns,
    save_snv_dataframe,
)


from .. import config

DEFAULT_DH_ROOT = config.UHGG_DH_FORMAT
DEFAULT_REFERENCE_ROOT = config.UHGG_REFERENCE_GENOMES
DEFAULT_GFF_ROOT = config.UHGG_GENES
DEFAULT_SNV_TABLE_ROOT = config.UHGG_ISOLATE_SNVS


def _decode_array(values: np.ndarray) -> np.ndarray:
    """Decode byte-string arrays from old npy files into plain Python strings."""
    if values.dtype.kind == "S":
        return values.astype(str)
    if values.dtype.kind == "O" and len(values) and isinstance(values[0], bytes):
        return np.asarray([x.decode("utf-8") for x in values])
    return values.astype(str) if values.dtype.kind in {"U", "O"} else values


def _read_fasta(path: Path) -> dict[str, str]:
    records: dict[str, list[str]] = {}
    current: str | None = None
    with path.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                current = line[1:].split()[0]
                records[current] = []
            elif current is not None:
                records[current].append(line.upper())
    return {name: "".join(seq) for name, seq in records.items()}


@dataclass(frozen=True)
class IsolateDHPaths:
    """Resolved paths for one accession's DH tables and optional reference."""

    accession: str
    dh_dir: Path
    reference_fasta: Path | None = None

    @classmethod
    def from_roots(
        cls,
        accession: str,
        dh_root: Path | str = DEFAULT_DH_ROOT,
        reference_root: Path | str | None = DEFAULT_REFERENCE_ROOT,
    ) -> "IsolateDHPaths":
        dh_dir = Path(dh_root) / accession
        reference_fasta = None
        if reference_root is not None:
            candidate = Path(reference_root) / f"{accession}.fna"
            reference_fasta = candidate if candidate.exists() else None
        return cls(accession=accession, dh_dir=dh_dir, reference_fasta=reference_fasta)


@dataclass(frozen=True)
class IsolateTablePaths:
    """Resolved paths for one accession's LiuGood-style isolate SNV tables."""

    accession: str
    table_dir: Path
    snv_catalog: Path
    alleles: Path
    coverage: Path
    biallelic_snvs: Path
    site_annotations: Path
    core_genes: Path
    core_gene_stats: Path
    snv_coverage: Path
    gene_files: Path
    metadata: Path
    identical_fraction: Path
    identical_fraction_metadata: Path

    @classmethod
    def from_root(
        cls,
        accession: str,
        table_root: Path | str = DEFAULT_SNV_TABLE_ROOT,
        table_format: str = "parquet",
    ) -> "IsolateTablePaths":
        table_dir = Path(table_root) / accession
        return cls(
            accession=accession,
            table_dir=table_dir,
            snv_catalog=table_dir / f"snv_catalog.{table_format}",
            alleles=table_dir / f"alleles.{table_format}",
            coverage=table_dir / f"coverage.{table_format}",
            biallelic_snvs=table_dir / f"biallelic_snvs.{table_format}",
            site_annotations=table_dir / f"site_annotations.{table_format}",
            core_genes=table_dir / "core_genes.json",
            core_gene_stats=table_dir / f"core_gene_stats.{table_format}",
            snv_coverage=table_dir / "SNV_coverage.npy",
            gene_files=table_dir / "gene_files.json",
            metadata=table_dir / "metadata.json",
            identical_fraction=table_dir / f"identical_fraction.{table_format}",
            identical_fraction_metadata=table_dir / "identical_fraction_metadata.json",
        )

    def core_tables_exist(self) -> bool:
        return self.snv_catalog.exists() and self.alleles.exists() and self.coverage.exists()


class IsolateSNVHelper:
    """SNVHelper-compatible adapter for UHGG isolate SNV tables.

    Attributes intentionally matching the old helper:
    - species/species_name
    - samples
    - coverage (lazy DataFrame)
    - core_to_snvs
    - core_1D/core_4D and snv_1D/snv_4D masks
    - compute_pairwise_snvs()
    - compute_pairwise_coverage()

    In ``source='tables'`` mode the helper follows the LiuGood-style table
    contract. In ``source='dh'`` mode it keeps the old memmapped-array adapter
    for backwards compatibility.
    """

    mutation_breakdown_available = False

    def __init__(
        self,
        accession: str,
        dh_root: Path | str = DEFAULT_DH_ROOT,
        reference_root: Path | str | None = DEFAULT_REFERENCE_ROOT,
        chunk_size: int = 100_000,
        *,
        table_root: Path | str | None = DEFAULT_SNV_TABLE_ROOT,
        snv_format: str = "parquet",
        source: str = "auto",
        compute_bi_snvs: bool = True,
        save_bi_snvs: bool = False,
        annotate: bool = False,
        mask_multi_sites: bool = True,
        recombination_root: Path | str | None = None,
        recombination_format: str = "parquet",
    ) -> None:
        self.species = accession
        self.species_name = accession
        self.paths = IsolateDHPaths.from_roots(accession, dh_root, reference_root)
        self.format = snv_format
        self.table_paths = (
            IsolateTablePaths.from_root(accession, table_root, snv_format)
            if table_root is not None
            else None
        )
        if recombination_root is None:
            recombination_root = table_root if table_root is not None else DEFAULT_RECOMBINATION_CACHE_ROOT
        self.recombination_paths = RecombinationCachePaths.from_root(
            accession,
            recombination_root,
            table_format=recombination_format,
        )
        self._recombination_events: pd.DataFrame | None = None
        self._recombination_pairs: pd.DataFrame | None = None
        if source not in {"auto", "tables", "dh"}:
            raise ValueError("source must be one of 'auto', 'tables', or 'dh'")

        if source == "auto":
            source = "tables" if self.table_paths is not None and self.table_paths.core_tables_exist() else "dh"

        if source == "tables":
            if self.table_paths is None:
                raise ValueError("table_root cannot be None when source='tables'")
            self._init_from_tables(
                compute_bi_snvs=compute_bi_snvs,
                save_bi_snvs=save_bi_snvs,
                annotate=annotate,
                mask_multi_sites=mask_multi_sites,
            )
        else:
            self._init_from_dh(chunk_size=chunk_size)

    def _init_from_dh(self, chunk_size: int) -> None:
        self.source = "dh"
        if not self.paths.dh_dir.exists():
            raise FileNotFoundError(f"DH directory not found: {self.paths.dh_dir}")

        self.chromosomes = _decode_array(np.load(self.paths.dh_dir / "chromosomes.npy", mmap_mode="r"))
        self.locations = np.load(self.paths.dh_dir / "locations.npy", mmap_mode="r")
        self.variants = _decode_array(np.load(self.paths.dh_dir / "variants.npy", mmap_mode="r"))
        self.gene_names = np.load(self.paths.dh_dir / "gene_names.npy", mmap_mode="r")
        self.samples = pd.Index(
            _decode_array(np.load(self.paths.dh_dir / "good_genomes.npy", allow_pickle=True)),
            name="sample",
        )

        self._snp_array = np.load(self.paths.dh_dir / "snp_array.npy", mmap_mode="r")
        self._covered_array = np.load(self.paths.dh_dir / "covered_array.npy", mmap_mode="r")
        self._sample_to_col = {sample: i for i, sample in enumerate(self.samples)}
        self._coverage_df: pd.DataFrame | None = None
        self._snvs_df: pd.DataFrame | None = None
        self._reference_bases: pd.Series | None = None

        self.index = pd.MultiIndex.from_arrays(
            [self.chromosomes, self.locations],
            names=["Contig", "Location"],
        )
        self.num_sites = len(self.index)
        self.core_to_snvs = self._compute_polymorphic_mask(chunk_size=chunk_size)
        self.snv_index = self.index[self.core_to_snvs]

        self.core_1D = pd.Series(self.variants == "1D", index=self.index, name="core_1D")
        self.core_4D = pd.Series(self.variants == "4D", index=self.index, name="core_4D")
        self.core_site_mask = pd.Series(True, index=self.index, name="core_site")
        self.snv_1D = self.core_1D.loc[self.snv_index]
        self.snv_4D = self.core_4D.loc[self.snv_index]
        self.has_site_annotations = True
        self.annotated = False
        self.genome_len = int(self.core_4D.sum())

    def _init_from_tables(
        self,
        compute_bi_snvs: bool,
        save_bi_snvs: bool,
        annotate: bool,
        mask_multi_sites: bool,
    ) -> None:
        self.source = "tables"
        assert self.table_paths is not None
        if not self.table_paths.core_tables_exist():
            raise FileNotFoundError(
                "Missing one or more core SNV tables: "
                f"{self.table_paths.snv_catalog}, {self.table_paths.alleles}, {self.table_paths.coverage}"
            )

        self.full_snvs = load_snv_dataframe(self.table_paths.snv_catalog, self.format)
        self._coverage_df = load_snv_dataframe(self.table_paths.coverage, self.format)
        self.samples = pd.Index(table_sample_columns(self._coverage_df), name="sample")
        self.full_snvs = self.full_snvs.loc[:, self.samples]
        self._sample_to_col = {sample: i for i, sample in enumerate(self.samples)}

        if compute_bi_snvs or not self.table_paths.biallelic_snvs.exists():
            alleles = load_snv_dataframe(self.table_paths.alleles, self.format)
            self._snvs_df, self.multi_sites = compute_biallelic_snvs(
                self.full_snvs,
                alleles,
                self._coverage_df,
            )
            if save_bi_snvs:
                save_snv_dataframe(self._snvs_df, self.table_paths.biallelic_snvs, self.format)
        else:
            self._snvs_df = load_snv_dataframe(self.table_paths.biallelic_snvs, self.format)
            multi_mask = pd.Series(
                ~self.full_snvs.index.isin(self._snvs_df.index),
                index=self.full_snvs.index,
                name="multi_sites",
            )
            self.multi_sites = multi_mask

        missing_snv_coverage = self._snvs_df.index.difference(self._coverage_df.index)
        if len(missing_snv_coverage):
            raise ValueError(
                f"{self.species}: {len(missing_snv_coverage)} biallelic SNV rows are absent "
                "from coverage. Rebuild coverage with a scope that includes all SNV coordinates."
            )
        ordered_snv_index = self._coverage_df.index[self._coverage_df.index.isin(self._snvs_df.index)]
        self._snvs_df = self._snvs_df.reindex(ordered_snv_index)

        if mask_multi_sites:
            multi_site_index = self.multi_sites[self.multi_sites].index
            shared_multi = self._coverage_df.index.intersection(multi_site_index)
            self._coverage_df.loc[shared_multi, self.samples] = False

        self.index = self._coverage_df.index
        self.num_sites = len(self.index)
        self.snv_index = self._snvs_df.index
        self.core_to_snvs = self.index.isin(self.snv_index)
        self.chromosomes = self.index.get_level_values("Contig").to_numpy(dtype=str)
        self.locations = self.index.get_level_values("Location").to_numpy(dtype=int)
        self.gene_names = None
        self._snp_array = None
        self._covered_array = None
        self._reference_bases: pd.Series | None = None

        self.annotated = False
        self.has_site_annotations = False
        self._core_genes: list[int] | None = None
        self._core_gene_stats: pd.DataFrame | None = None
        if self.table_paths.site_annotations.exists():
            self.mut_df = load_snv_dataframe(self.table_paths.site_annotations, self.format)
            self.mut_df = self.mut_df.reindex(self.index)
            self._set_annotation_masks_from_mut_df()
            self.annotated = True
            self.has_site_annotations = True
        elif annotate:
            raise FileNotFoundError(
                f"Site annotation table not found: {self.table_paths.site_annotations}. "
                "Generate it before requesting annotate=True."
            )
        else:
            self.core_site_mask = pd.Series(True, index=self.index, name="core_site")
            self.core_1D = pd.Series(False, index=self.index, name="core_1D")
            self.core_4D = pd.Series(False, index=self.index, name="core_4D")
            self.snv_1D = self.core_1D.reindex(self.snv_index, fill_value=False)
            self.snv_4D = self.core_4D.reindex(self.snv_index, fill_value=False)
        self.genome_len = int(self.core_4D.sum()) if self.has_site_annotations else int(self.num_sites)

    @property
    def core_genes(self) -> list[str]:
        """Core gene IDs for this accession, if computed.

        Kept as strings so both MIDAS feature IDs (e.g. ``445970.5.peg.2``) and
        legacy numeric UHGG gene indices are supported without coercion.
        """
        if self.source != "tables" or self.table_paths is None:
            raise RuntimeError("core_genes are only available for source='tables'")
        if self._core_genes is None:
            if not self.table_paths.core_genes.exists():
                raise FileNotFoundError(f"Core gene list not found: {self.table_paths.core_genes}")
            self._core_genes = [str(gene) for gene in json.loads(self.table_paths.core_genes.read_text())]
        return self._core_genes

    @property
    def core_gene_stats(self) -> pd.DataFrame:
        """Per-feature core-gene coverage/prevalence stats, if computed."""
        if self.source != "tables" or self.table_paths is None:
            raise RuntimeError("core_gene_stats are only available for source='tables'")
        if self._core_gene_stats is None:
            if not self.table_paths.core_gene_stats.exists():
                raise FileNotFoundError(f"Core gene stats not found: {self.table_paths.core_gene_stats}")
            self._core_gene_stats = pd.read_parquet(self.table_paths.core_gene_stats)
        return self._core_gene_stats

    @property
    def identical_fraction(self) -> pd.DataFrame:
        """Long-form pairwise identical-fraction table for this accession.

        Reads ``identical_fraction.<fmt>`` under the SNV table root. See
        ``analysis/compute_identical_fraction.py`` for how the cache is built.
        """
        if self.source != "tables" or self.table_paths is None:
            raise RuntimeError("identical_fraction is only available for source='tables'")
        if not self.table_paths.identical_fraction.exists():
            raise FileNotFoundError(
                f"Identical-fraction cache not found: {self.table_paths.identical_fraction}. "
                "Generate it with analysis/compute_identical_fraction.py."
            )
        path = self.table_paths.identical_fraction
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        if path.suffix == ".feather":
            return pd.read_feather(path)
        raise ValueError(f"Unsupported identical_fraction format: {path.suffix}")

    def get_close_pairs(
        self,
        cutoff: float = 0.5,
        *,
        block_size: int = 1000,
        site_class: str = "4D",
    ) -> list[tuple[str, str]]:
        """Sample-name pairs with ``identical_fraction > cutoff``.

        The cached metadata is validated against ``block_size`` and
        ``site_class`` so a mismatched artifact raises rather than silently
        returning the wrong pair set.
        """
        if self.source != "tables" or self.table_paths is None:
            raise RuntimeError("get_close_pairs is only available for source='tables'")
        meta_path = self.table_paths.identical_fraction_metadata
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Identical-fraction metadata not found: {meta_path}. "
                "Generate it with analysis/compute_identical_fraction.py."
            )
        metadata = json.loads(meta_path.read_text())
        cached_block = int(metadata.get("block_size", -1))
        cached_class = str(metadata.get("site_class", ""))
        if cached_block != int(block_size):
            raise ValueError(
                f"Cached identical_fraction was computed with block_size={cached_block}, "
                f"but get_close_pairs was called with block_size={block_size}."
            )
        if cached_class != site_class:
            raise ValueError(
                f"Cached identical_fraction was computed with site_class={cached_class!r}, "
                f"but get_close_pairs was called with site_class={site_class!r}."
            )
        df = self.identical_fraction
        close = df.loc[df["identical_fraction"] > float(cutoff), ["sample_1", "sample_2"]]
        return [(str(a), str(b)) for a, b in close.itertuples(index=False)]

    @property
    def has_recombination_cache(self) -> bool:
        """Whether this accession has a recombination event cache on disk."""
        return self.recombination_paths.exists()

    @property
    def recombination_events(self) -> pd.DataFrame:
        """Pair-specific recombination event intervals for this accession."""
        if self._recombination_events is None:
            if not self.recombination_paths.events.exists():
                raise FileNotFoundError(
                    f"Recombination event cache not found: {self.recombination_paths.events}"
                )
            self._recombination_events = load_recombination_events(self.recombination_paths.events)
        return self._recombination_events

    @property
    def recombination_pairs(self) -> pd.DataFrame:
        """Pair-level recombination summary table for this accession."""
        if self._recombination_pairs is None:
            if not self.recombination_paths.pairs.exists():
                raise FileNotFoundError(
                    f"Recombination pair cache not found: {self.recombination_paths.pairs}"
                )
            self._recombination_pairs = load_recombination_pairs(self.recombination_paths.pairs)
        return self._recombination_pairs

    def get_pair_recombination_events(
        self,
        sample1: str,
        sample2: str,
        *,
        include_reverse: bool = True,
        dedup_only: bool = False,
    ) -> pd.DataFrame:
        """Return cached recombination events for one pair."""
        return events_for_pair(
            self.recombination_events,
            str(sample1),
            str(sample2),
            include_reverse=include_reverse,
            dedup_only=dedup_only,
        )

    def compute_pair_recombination_mask(
        self,
        sample1: str,
        sample2: str,
        *,
        include_reverse: bool = True,
        dedup_only: bool = False,
    ) -> pd.Series:
        """Return a genome-coordinate recombination mask for one pair.

        The mask is built from inclusive reference contig intervals in the cache
        and is indexed exactly like ``self.coverage``.
        """
        events = self.get_pair_recombination_events(
            sample1,
            sample2,
            include_reverse=include_reverse,
            dedup_only=dedup_only,
        )
        return recombination_mask_from_events(self.index, events)

    def _set_annotation_masks_from_mut_df(self) -> None:
        site_types = self.mut_df["Site Type"]
        core_gene_mask = pd.Series(True, index=self.index, name="core_gene")
        if self.source == "tables" and self.table_paths is not None and self.table_paths.core_genes.exists():
            core_gene_ids = {str(g) for g in self.core_genes}
            # Compare as strings: MIDAS Gene Name is "<genome>.peg.N"; legacy UHGG
            # tables store a numeric index (possibly float after NaN coercion).
            gene_names = self.mut_df["Gene Name"]
            if pd.api.types.is_float_dtype(gene_names):
                gene_names = gene_names.astype("Int64")
            gene_ids = gene_names.astype(str)
            core_gene_mask = gene_ids.isin(core_gene_ids)
            core_gene_mask.index = self.index
        self.core_site_mask = core_gene_mask & site_types.isin(["1D", "2D", "3D", "4D"])
        self.core_1D = self.core_site_mask & (site_types == "1D")
        self.core_4D = self.core_site_mask & (site_types == "4D")
        self.core_site_mask.name = "core_site"
        self.core_1D.name = "core_1D"
        self.core_4D.name = "core_4D"
        self.snv_1D = self.core_1D.reindex(self.snv_index, fill_value=False)
        self.snv_4D = self.core_4D.reindex(self.snv_index, fill_value=False)
        if all(col in self.mut_df.columns for col in ["A", "C", "G", "T"]):
            snv_types = [self.mut_df.at[idx, col] for idx, col in zip(self.snv_index, self.snvs["Alt"])]
            self.snv_types = pd.Series(snv_types, index=self.snv_index)

    @property
    def coverage(self) -> pd.DataFrame:
        """Boolean coverage matrix indexed like the original SNVHelper.coverage."""
        if self._coverage_df is None:
            self._coverage_df = pd.DataFrame(
                self._covered_array,
                index=self.index,
                columns=self.samples,
                copy=False,
            )
        return self._coverage_df

    @property
    def snvs(self) -> pd.DataFrame:
        """Polymorphic genotype table with 0/1 alleles and 255 for missing."""
        if self._snvs_df is None:
            genotypes = self._snp_array[self.core_to_snvs].astype(bool).astype(np.uint8)
            covered = self._covered_array[self.core_to_snvs]
            genotypes[~covered] = 255
            self._snvs_df = pd.DataFrame(genotypes, index=self.snv_index, columns=self.samples)
        return self._snvs_df

    @property
    def reference_bases(self) -> pd.Series:
        """Reference base at each DH coordinate, loaded lazily from the FASTA."""
        if self._reference_bases is None:
            if self.paths.reference_fasta is None:
                raise FileNotFoundError(
                    "No reference FASTA was found. Pass reference_root or create "
                    f"{DEFAULT_REFERENCE_ROOT / (self.species + '.fna')}"
                )
            records = _read_fasta(self.paths.reference_fasta)
            bases = np.empty(self.num_sites, dtype="U1")
            for contig in pd.unique(self.chromosomes):
                mask = self.chromosomes == contig
                if contig not in records:
                    raise KeyError(f"Contig {contig!r} missing from {self.paths.reference_fasta}")
                seq = records[contig]
                locs = self.locations[mask].astype(int)
                if locs.max(initial=0) > len(seq):
                    raise ValueError(f"DH locations exceed FASTA length for contig {contig!r}")
                bases[mask] = np.fromiter((seq[i - 1] for i in locs), dtype="U1", count=len(locs))
            self._reference_bases = pd.Series(bases, index=self.index, name="Ref")
        return self._reference_bases

    def _compute_polymorphic_mask(self, chunk_size: int) -> np.ndarray:
        """Rows where at least one covered isolate has 0 and one has 1."""
        mask = np.zeros(self._snp_array.shape[0], dtype=bool)
        for start in range(0, self._snp_array.shape[0], chunk_size):
            stop = min(start + chunk_size, self._snp_array.shape[0])
            covered = self._covered_array[start:stop]
            ones = ((self._snp_array[start:stop] == 1) & covered).sum(axis=1)
            covered_counts = covered.sum(axis=1)
            zeros = covered_counts - ones
            mask[start:stop] = (ones > 0) & (zeros > 0)
        return mask

    def _col(self, sample: str) -> int:
        try:
            return self._sample_to_col[str(sample)]
        except KeyError as exc:
            raise KeyError(f"Sample {sample!r} is not present for {self.species}") from exc

    def compute_pairwise_snvs(self, sample1: str, sample2: str) -> pd.Series:
        """Return polymorphic sites where two isolates differ and both are covered."""
        if self.source == "tables":
            sample1_alleles = self.snvs[str(sample1)]
            sample2_alleles = self.snvs[str(sample2)]
            diff_sites = (
                (sample1_alleles != sample2_alleles)
                & (sample1_alleles != 255)
                & (sample2_alleles != 255)
            )
            return pd.Series(diff_sites.to_numpy(dtype=bool), index=self.snv_index, name="diff")

        i = self._col(sample1)
        j = self._col(sample2)
        g1 = self._snp_array[self.core_to_snvs, i].astype(bool)
        g2 = self._snp_array[self.core_to_snvs, j].astype(bool)
        covered = self._covered_array[self.core_to_snvs, i] & self._covered_array[self.core_to_snvs, j]
        return pd.Series((g1 != g2) & covered, index=self.snv_index, name="diff")

    def compute_pairwise_coverage(self, sample1: str, sample2: str) -> pd.Series:
        """Return genome-coordinate sites covered in both isolates."""
        if self.source == "tables":
            covered = self.coverage[str(sample1)] & self.coverage[str(sample2)]
            return pd.Series(covered.to_numpy(dtype=bool), index=self.index, name="covered")

        i = self._col(sample1)
        j = self._col(sample2)
        covered = self._covered_array[:, i] & self._covered_array[:, j]
        return pd.Series(covered, index=self.index, name="covered")

    def get_pair_snp_info(self, pair: Iterable[str], site_class: str = "4D"):
        """Return CP-HMM input: SNP vector, contig names, and contig locations.

        By default this mirrors the PLoS Bio 2024 wrapper and uses covered 4D
        sites only.
        """
        if site_class in {"4D", "1D"} and not self.has_site_annotations:
            raise RuntimeError(
                f"{self.species}: site_class={site_class!r} requires a site_annotations table. "
                "Use site_class='all' for raw table coordinates or generate annotations first."
            )
        sample1, sample2 = tuple(pair)
        diffs = self.compute_pairwise_snvs(sample1, sample2)
        coverage = self.compute_pairwise_coverage(sample1, sample2).to_numpy()

        diff_core = np.zeros(self.num_sites, dtype=bool)
        diff_core[self.core_to_snvs] = diffs.to_numpy()

        if site_class == "4D":
            site_mask = self.core_4D.to_numpy()
        elif site_class == "1D":
            site_mask = self.core_1D.to_numpy()
        elif site_class in {"all", "covered"}:
            site_mask = np.ones(self.num_sites, dtype=bool)
        else:
            raise ValueError("site_class must be one of '4D', '1D', or 'all'")

        keep = coverage & site_mask
        return diff_core[keep], self.chromosomes[keep].astype(str), self.locations[keep].astype(int)

    def compute_basic_pair_counts(
        self,
        sample1: str,
        sample2: str,
        recombination_mask: np.ndarray | pd.Series | None = None,
    ) -> dict[str, int]:
        """Compute dN/dS count columns that do not require Alt-base identities."""
        if not self.has_site_annotations:
            raise RuntimeError(
                f"{self.species}: basic 1D/4D counts require a site_annotations table. "
                "The raw SNV/allele/coverage tables are available, but degeneracy masks are not."
            )
        diffs = self.compute_pairwise_snvs(sample1, sample2)
        covered = self.compute_pairwise_coverage(sample1, sample2).to_numpy()
        diff_core = np.zeros(self.num_sites, dtype=bool)
        diff_core[self.core_to_snvs] = diffs.to_numpy()
        core_1d = self.core_1D.to_numpy()
        core_4d = self.core_4D.to_numpy()

        counts = {
            "core_len": int(covered.sum()),
            "core_diff": int(diff_core.sum()),
            "core_len_4D": int((covered & core_4d).sum()),
            "core_len_1D": int((covered & core_1d).sum()),
            "core_diff_4D": int((diff_core & core_4d).sum()),
            "core_diff_1D": int((diff_core & core_1d).sum()),
        }

        if recombination_mask is not None:
            recomb = self._coerce_site_mask(recombination_mask)
            clonal = ~recomb
            counts.update(
                {
                    "recomb_len_4D": int((covered & recomb & core_4d).sum()),
                    "recomb_len_1D": int((covered & recomb & core_1d).sum()),
                    "recomb_diff_4D": int((diff_core & recomb & core_4d).sum()),
                    "recomb_diff_1D": int((diff_core & recomb & core_1d).sum()),
                    "clonal_len_4D": int((covered & clonal & core_4d).sum()),
                    "clonal_len_1D": int((covered & clonal & core_1d).sum()),
                    "clonal_diff_4D": int((diff_core & clonal & core_4d).sum()),
                    "clonal_diff_1D": int((diff_core & clonal & core_1d).sum()),
                }
            )
        return counts

    def _coerce_site_mask(self, mask: np.ndarray | pd.Series) -> np.ndarray:
        if isinstance(mask, pd.Series):
            return mask.reindex(self.index, fill_value=False).to_numpy(dtype=bool)
        arr = np.asarray(mask, dtype=bool)
        if arr.shape != (self.num_sites,):
            raise ValueError(f"Expected mask of shape {(self.num_sites,)}, got {arr.shape}")
        return arr
