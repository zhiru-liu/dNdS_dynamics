"""Site and mutation annotation cache for UHGG isolate SNV tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.Seq import Seq

from . import codon_annotation as annotation_utils
from .isolate import DEFAULT_GFF_ROOT, DEFAULT_REFERENCE_ROOT, IsolateSNVHelper
from .table_contract import save_snv_dataframe


@dataclass(frozen=True)
class SiteAnnotationResult:
    accession: str
    output_path: Path
    num_sites: int
    site_type_counts: dict[str, int]


def read_fasta(path: Path) -> dict[str, str]:
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


def gff_to_gene_df(path: Path) -> pd.DataFrame:
    rows = []
    with path.open() as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            rows.append(line.rstrip("\n").split("\t")[:9])
    df = pd.DataFrame(
        rows,
        columns=[
            "Contig",
            "Source",
            "Type",
            "Start",
            "End",
            "Score",
            "Strand",
            "Frame",
            "Attribute",
        ],
    )
    df["Start"] = df["Start"].astype(int)
    df["End"] = df["End"].astype(int)
    df["Gene ID"] = np.arange(df.shape[0], dtype=np.int32)
    return df


def polarize_reference_records(
    records: dict[str, str],
    biallelic_snvs: pd.DataFrame,
) -> dict[str, Seq]:
    """Return reference contigs with SNV coordinates changed to the major allele."""
    polarized = {contig: np.asarray(list(seq), dtype="U1") for contig, seq in records.items()}
    if biallelic_snvs.empty:
        return {contig: Seq("".join(seq)) for contig, seq in polarized.items()}

    for contig, contig_snvs in biallelic_snvs.groupby(level="Contig", sort=False):
        contig = str(contig)
        if contig not in polarized:
            raise KeyError(f"Contig {contig!r} missing from reference FASTA")
        locs = contig_snvs.index.get_level_values("Location").to_numpy(dtype=int)
        seq = polarized[contig]
        if locs.max(initial=0) > len(seq):
            raise ValueError(f"SNV location exceeds contig length for {contig!r}")
        observed_ref = seq[locs - 1]
        expected_ref = contig_snvs["Ref"].astype(str).to_numpy()
        if not np.all(observed_ref == expected_ref):
            bad = np.flatnonzero(observed_ref != expected_ref)[0]
            raise ValueError(
                f"Reference base mismatch for {contig}:{locs[bad]} "
                f"({observed_ref[bad]} != {expected_ref[bad]})"
            )
        seq[locs - 1] = contig_snvs["Major"].astype(str).to_numpy()

    return {contig: Seq("".join(seq)) for contig, seq in polarized.items()}


def annotate_sites_for_helper(
    helper: IsolateSNVHelper,
    ref_fasta_path: Path,
    gff_path: Path,
) -> pd.DataFrame:
    """Annotate helper coverage coordinates using major-polarized codons."""
    records = read_fasta(ref_fasta_path)
    gene_df = gff_to_gene_df(gff_path)
    polarized_records = polarize_reference_records(records, helper.snvs)

    per_contig = []
    covered_contigs = pd.Index(helper.index.get_level_values("Contig")).unique()
    for contig in covered_contigs:
        contig = str(contig)
        if contig not in polarized_records:
            raise KeyError(f"Contig {contig!r} missing from polarized reference")
        contig_gene_df = gene_df[gene_df["Contig"] == contig]
        annotated = annotation_utils.annotate_sequence_site_types_to_df(
            polarized_records[contig],
            contig_gene_df,
        )
        annotated["Contig"] = contig
        per_contig.append(annotated)

    mut_df = pd.concat(per_contig, ignore_index=True)
    mut_df = mut_df.rename(columns={f"{base} Mut": base for base in ["A", "C", "G", "T"]})
    mut_df.set_index(["Contig", "Location"], inplace=True)
    mut_df = mut_df.reindex(helper.index)

    for col in ["Site Type", "Gene Name", "A", "C", "G", "T", "Ref Base"]:
        if col in mut_df.columns:
            mut_df[col] = mut_df[col].where(mut_df[col].notna(), "NA").astype(str)
    if "Gene Count" in mut_df.columns:
        mut_df["Gene Count"] = pd.to_numeric(mut_df["Gene Count"], errors="coerce").fillna(0).astype(int)

    mut_df["s"] = (mut_df.loc[:, ["A", "T", "C", "G"]] == "s").sum(axis=1) - 1
    mut_df["n"] = (mut_df.loc[:, ["A", "T", "C", "G"]] == "n").sum(axis=1)
    mut_df["m"] = (mut_df.loc[:, ["A", "T", "C", "G"]] == "m").sum(axis=1)
    mut_df["nn"] = (mut_df.loc[:, ["A", "T", "C", "G"]] == "nn").sum(axis=1)
    return mut_df


def build_site_annotation_cache(
    accession: str,
    *,
    table_root: Path,
    ref_root: Path = DEFAULT_REFERENCE_ROOT,
    gff_root: Path = DEFAULT_GFF_ROOT,
    table_format: str = "parquet",
    overwrite: bool = False,
    save_biallelic_snvs: bool = True,
) -> SiteAnnotationResult:
    """Build ``site_annotations.parquet`` for one isolate accession."""
    helper = IsolateSNVHelper(
        accession,
        table_root=table_root,
        snv_format=table_format,
        source="tables",
        compute_bi_snvs=True,
        save_bi_snvs=save_biallelic_snvs,
    )
    assert helper.table_paths is not None
    output_path = helper.table_paths.site_annotations
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} exists; pass overwrite=True to replace it")

    mut_df = annotate_sites_for_helper(
        helper,
        ref_fasta_path=Path(ref_root) / f"{accession}.fna",
        gff_path=Path(gff_root) / f"{accession}.gff",
    )
    save_snv_dataframe(mut_df, output_path, table_format)
    counts = mut_df["Site Type"].value_counts(dropna=False).to_dict()
    return SiteAnnotationResult(
        accession=accession,
        output_path=output_path,
        num_sites=int(mut_df.shape[0]),
        site_type_counts={str(key): int(value) for key, value in counts.items()},
    )
