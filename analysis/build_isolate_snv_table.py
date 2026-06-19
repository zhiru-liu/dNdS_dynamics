"""Build UHGG-style SNV tables for NCBI isolate assemblies of any species.

For each vhq/hq/complete isolate, align the assembly to the MIDAS reference for
``--midas-species`` with nucmer, derive a per-isolate coverage mask and
base-call array on reference coordinates (with deletion->missing and frameshift
masking), then aggregate across isolates into the canonical UHGG isolate SNV
table contract:

- ``snv_catalog.parquet``: polymorphic sites, per-isolate non-ref boolean
- ``alleles.parquet``: polymorphic sites, per-isolate base call (+ Ref column)
- ``coverage.parquet``: all reference sites, per-isolate boolean coverage
- ``biallelic_snvs.parquet``: polarized 0/1/255 calls with Ref/Major/Alt
- ``site_annotations.parquet``: site type / gene / mutation type per position
- ``metadata.json``: provenance + alignment parameters

Reference choice (the alignment target). By default the reference is the MIDAS
rep genome for ``--midas-species`` (``<rep-genome-root>/<midas-species>/
genome.fna.gz`` + ``genome.features.gz``), which is the same reference the
LiuGood2024 metagenome catalog was called against — required if you want to
compare isolates to the metagenome (QP) data. To use **any other reference**,
pass ``--ref-fna`` + ``--ref-gff`` (FASTA + GFF); everything downstream
(annotation, SNV table, recombination, dN/dS) is coordinate-based and works with
any reference, so only the alignment target and the gene annotation change.
Isolate FASTAs: ``<ncbi-root>/<isolate-dir>/fasta/*.fna.gz``.

Core genes (``--core-gene-source``). Site-type/dN/dS work on the whole reference,
but recombination detection restricts to *core* genes. "Core" is reference- and
annotation-dependent: with ``qp`` (default) a gene is core if its reference sites
appear in the matching MIDAS/QP catalog — which only exists for MIDAS references.
For a non-MIDAS reference there is no QP catalog, so use ``--core-gene-source none``
and supply ``core_genes.json`` yourself (e.g. genes present across your isolate
panel, or from the reference's pangenome); see analysis/README.md.

This is the single isolate SNV-table builder for all species.

Examples:
  # MIDAS reference (comparable to the metagenome catalog)
  python build_isolate_snv_table.py --midas-species Alistipes_putredinis_61533
  python build_isolate_snv_table.py --midas-species Bacteroides_vulgatus_57955
  # Arbitrary reference (FASTA + GFF), no QP/MIDAS core
  python build_isolate_snv_table.py --species My_species \\
      --ref-fna ref.fna --ref-gff ref.gff --core-gene-source none
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from dnds_dynamics import config  # noqa: E402

REP_GENOME_ROOT = config.GG2019_REP_GENOMES
NCBI_ROOT = config.NCBI_ISOLATES_ROOT

# These path globals are SET BY main() via configure_paths() before any worker
# function that reads them runs. Defaults reproduce the A. putredinis run.
SPECIES = "Alistipes_putredinis_61533"            # MIDAS rep-genome species id
REF_FNA_GZ = REP_GENOME_ROOT / SPECIES / "genome.fna.gz"
REF_FEATURES_GZ = REP_GENOME_ROOT / SPECIES / "genome.features.gz"
REF_ANNOTATION_FORMAT = "features"  # "features" (MIDAS .features.gz) or "gff"
ISOLATE_ROOT = NCBI_ROOT / "Alistipes_putredinis"  # NCBI download dir
MANIFEST = ISOLATE_ROOT / "manifest_all_isolates.tsv"
FASTA_DIR = ISOLATE_ROOT / "fasta"
DEFAULT_OUT_DIR = ISOLATE_ROOT / "snv_table"


def _default_isolate_dir(midas_species: str) -> str:
    """`Bacteroides_fragilis_54507` -> `Bacteroides_fragilis` (drop numeric MIDAS id)."""
    parts = midas_species.split("_")
    if parts and parts[-1].isdigit():
        parts = parts[:-1]
    return "_".join(parts)


def configure_paths(species: str, midas_species: str | None = None,
                    isolate_dir: str | None = None, out_dir: Path | None = None,
                    ref_fna: Path | None = None, ref_gff: Path | None = None) -> None:
    """Set module-level path globals (read by worker fns).

    ``species`` is the label/slug for outputs and the default isolate download
    directory. The alignment reference is either an explicit ``ref_fna`` +
    ``ref_gff`` (any genome + annotation), or — if those are omitted — the MIDAS
    rep genome for ``midas_species`` (then required), the reference the metagenome
    catalog was called against.
    """
    global SPECIES, REF_FNA_GZ, REF_FEATURES_GZ, REF_ANNOTATION_FORMAT
    global ISOLATE_ROOT, MANIFEST, FASTA_DIR, DEFAULT_OUT_DIR
    SPECIES = species
    if ref_fna is not None:
        REF_FNA_GZ = Path(ref_fna)
        REF_FEATURES_GZ = Path(ref_gff)
        REF_ANNOTATION_FORMAT = "gff"
    else:
        if not midas_species:
            raise ValueError("midas_species is required when ref_fna/ref_gff are not given")
        REF_FNA_GZ = REP_GENOME_ROOT / midas_species / "genome.fna.gz"
        REF_FEATURES_GZ = REP_GENOME_ROOT / midas_species / "genome.features.gz"
        REF_ANNOTATION_FORMAT = "features"
    ISOLATE_ROOT = NCBI_ROOT / (isolate_dir or _default_isolate_dir(species))
    MANIFEST = ISOLATE_ROOT / "manifest_all_isolates.tsv"
    FASTA_DIR = ISOLATE_ROOT / "fasta"
    DEFAULT_OUT_DIR = out_dir or (ISOLATE_ROOT / "snv_table")


def _load_gene_df() -> pd.DataFrame:
    """Read the reference gene table as columns [Gene ID, Contig, Start, End, Strand, Type]."""
    if REF_ANNOTATION_FORMAT == "gff":
        rows = []
        opener = gzip.open if str(REF_FEATURES_GZ).endswith(".gz") else open
        with opener(REF_FEATURES_GZ, "rt") as fh:
            for line in fh:
                if line.startswith("#") or not line.strip():
                    continue
                f = line.rstrip("\n").split("\t")
                if len(f) < 8:
                    continue
                rows.append({"Contig": f[0], "Type": f[2], "Start": int(f[3]),
                             "End": int(f[4]), "Strand": f[6]})
        gene_df = pd.DataFrame(rows)
        gene_df["Gene ID"] = np.arange(len(gene_df), dtype=np.int32)
        return gene_df
    gene_df = pd.read_csv(REF_FEATURES_GZ, sep="\t")
    gene_df.columns = ["Gene ID", "Contig", "Start", "End", "Strand", "Type", "Info"]
    return gene_df

NUCMER = str(config.NUCMER_BIN)
DELTA_FILTER = str(config.DELTA_FILTER_BIN)
SHOW_SNPS = str(config.SHOW_SNPS_BIN)
SHOW_COORDS = str(config.SHOW_COORDS_BIN)

ALLOWED_TIERS = ("vhq", "hq", "complete")
MIN_ALIGN_IDENT = 80  # %identity for delta-filter
MIN_CONTIG_LEN = 2000  # ignore tiny ref contigs
# Frameshift masking: indels within INDEL_MERGE_GAP bp are treated as one event;
# events whose net length is not a multiple of 3 (frameshifts) get a flanking
# window of FRAMESHIFT_MASK_BP masked as missing on each side.
INDEL_MERGE_GAP = 10
FRAMESHIFT_MASK_BP = 30


def read_fasta_gz(path: Path) -> dict[str, str]:
    records: dict[str, list[str]] = {}
    current = None
    open_fn = gzip.open if str(path).endswith(".gz") else open
    with open_fn(path, "rt") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                current = line[1:].split()[0]
                records[current] = []
            elif current is not None:
                records[current].append(line.upper())
    return {k: "".join(v) for k, v in records.items()}


def write_fasta(records: dict[str, str], path: Path) -> None:
    with open(path, "w") as handle:
        for cid, seq in records.items():
            handle.write(f">{cid}\n")
            for i in range(0, len(seq), 80):
                handle.write(seq[i:i + 80] + "\n")


def run_alignment(ref_path: Path, isolate_path: Path, work_dir: Path, prefix: str) -> tuple[Path, Path, Path]:
    """Run nucmer + delta-filter + show-snps + show-coords, return paths to results."""
    delta = work_dir / f"{prefix}.delta"
    delta_filt = work_dir / f"{prefix}.filt.delta"
    snps = work_dir / f"{prefix}.snps"
    coords = work_dir / f"{prefix}.coords"

    subprocess.run(
        [NUCMER, "--prefix", str(work_dir / prefix), str(ref_path), str(isolate_path)],
        check=True, capture_output=True,
    )
    with open(delta_filt, "wb") as out:
        subprocess.run(
            [DELTA_FILTER, "-1", "-i", str(MIN_ALIGN_IDENT), str(delta)],
            check=True, stdout=out, stderr=subprocess.PIPE,
        )
    # NOTE: indels are intentionally REPORTED (no -I) so we can (a) mark deleted
    # reference positions as missing rather than reference, and (b) mask windows
    # around frameshift indels. -C still excludes ambiguous (repeat) alignments.
    subprocess.run(
        [SHOW_SNPS, "-C", "-l", "-r", "-T", "-H", str(delta_filt)],
        check=True, stdout=open(snps, "wb"),
    )
    subprocess.run(
        [SHOW_COORDS, "-T", "-H", "-r", "-l", "-c", str(delta_filt)],
        check=True, stdout=open(coords, "wb"),
    )
    return delta_filt, snps, coords


def parse_coords(coords_path: Path) -> pd.DataFrame:
    """show-coords -THrlc: cols S1 E1 S2 E2 LEN1 LEN2 %IDY LENR LENQ COVR COVQ TAGS_R TAGS_Q."""
    if coords_path.stat().st_size == 0:
        return pd.DataFrame(columns=["s1", "e1", "s2", "e2", "len1", "len2", "ident", "ref_contig", "q_contig"])
    rows = []
    with open(coords_path) as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            # nucmer-4 -THrlc output has 13 fields
            if len(parts) < 13:
                continue
            s1, e1, s2, e2 = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            len1, len2 = int(parts[4]), int(parts[5])
            ident = float(parts[6])
            ref_contig = parts[11]
            q_contig = parts[12]
            rows.append(
                {
                    "s1": s1, "e1": e1, "s2": s2, "e2": e2,
                    "len1": len1, "len2": len2, "ident": ident,
                    "ref_contig": ref_contig, "q_contig": q_contig,
                }
            )
    return pd.DataFrame(rows)


def parse_snps(snps_path: Path) -> pd.DataFrame:
    """show-snps -CTrlHI: cols P1 SUB_R SUB_Q P2 BUFF DIST LEN_R LEN_Q FRM_R FRM_Q TAGS_R TAGS_Q."""
    if snps_path.stat().st_size == 0:
        return pd.DataFrame(columns=["ref_pos", "ref_base", "alt_base", "ref_contig"])
    rows = []
    with open(snps_path) as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 12:
                continue
            p1 = int(parts[0])
            sub_r = parts[1]
            sub_q = parts[2]
            ref_contig = parts[10]
            rows.append({"ref_pos": p1, "ref_base": sub_r, "alt_base": sub_q, "ref_contig": ref_contig})
    return pd.DataFrame(rows)


def isolate_arrays(coords_df: pd.DataFrame, snps_df: pd.DataFrame,
                   ref_records: dict[str, str]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Build per-contig coverage (bool) and allele (U1) arrays for one isolate.

    Coverage = positions inside any 1-to-1 filtered alignment interval.
    Allele = ref base where covered and not in a substitution; alt base at SNV
    positions; '.' (treated as not covered) at indel/gap-only positions.
    """
    coverage: dict[str, np.ndarray] = {}
    alleles: dict[str, np.ndarray] = {}
    for contig, seq in ref_records.items():
        n = len(seq)
        cov = np.zeros(n, dtype=bool)
        ale = np.full(n, "N", dtype="U1")
        coverage[contig] = cov
        alleles[contig] = ale

    if not coords_df.empty:
        for _, row in coords_df.iterrows():
            c = row["ref_contig"]
            if c not in coverage:
                continue
            a = int(min(row["s1"], row["e1"])) - 1
            b = int(max(row["s1"], row["e1"]))  # exclusive
            coverage[c][a:b] = True
            ale = alleles[c]
            seq = ref_records[c]
            # default the aligned region to the reference base
            ale[a:b] = np.frombuffer(seq[a:b].encode("ascii"), dtype="U1") if False else \
                np.asarray(list(seq[a:b]), dtype="U1")

    if snps_df.empty:
        return coverage, alleles

    # Pass 1: apply substitutions; collect indel rows for frameshift handling.
    # show-snps rows: ref_base=='.' -> insertion in isolate (no ref base consumed);
    # alt_base=='.' -> deletion in isolate (a reference base is missing in isolate).
    indel_rows = []  # (contig, ref_pos_0based, kind) where kind in {'ins','del'}
    for _, row in snps_df.iterrows():
        c = row["ref_contig"]
        if c not in coverage:
            continue
        p = int(row["ref_pos"]) - 1
        rb = row["ref_base"]
        ab = row["alt_base"]
        if rb == "." or ab == ".":
            kind = "ins" if rb == "." else "del"
            # clamp ref pos for insertions (P1 is the flanking ref position)
            p_clamped = min(max(p, 0), len(coverage[c]) - 1)
            indel_rows.append((c, p_clamped, kind))
            continue
        if not coverage[c][p]:
            continue
        # substitution
        if ab in "ACGT":
            alleles[c][p] = ab
        else:
            # ambiguous base call -> treat as missing
            coverage[c][p] = False
            alleles[c][p] = "N"

    # Pass 2: deletions -> always missing (a deleted base has no allele on the
    # reference position). Group consecutive indel positions into events; if an
    # event's net length is not a multiple of 3 (frameshift), mask a flanking
    # window so spurious boundary substitutions don't inflate dN.
    for c, p, kind in indel_rows:
        if kind == "del":
            coverage[c][p] = False
            alleles[c][p] = "N"

    # group indels per contig into events of adjacent reference positions
    by_contig: dict[str, list[tuple[int, str]]] = {}
    for c, p, kind in indel_rows:
        by_contig.setdefault(c, []).append((p, kind))
    for c, events in by_contig.items():
        events.sort()
        i = 0
        n = len(events)
        while i < n:
            j = i
            # extend the event while positions are within INDEL_MERGE_GAP bp
            while j + 1 < n and events[j + 1][0] - events[j][0] <= INDEL_MERGE_GAP:
                j += 1
            span = events[i:j + 1]
            # net indel length: +1 per inserted base, -1 per deleted base
            net = sum(1 if k == "ins" else -1 for _, k in span)
            if net % 3 != 0:  # frameshift
                lo = max(span[0][0] - FRAMESHIFT_MASK_BP, 0)
                hi = min(span[-1][0] + FRAMESHIFT_MASK_BP + 1, len(coverage[c]))
                coverage[c][lo:hi] = False
                alleles[c][lo:hi] = "N"
            i = j + 1

    return coverage, alleles


# --------------------------------------------------------------------------- #
# Memory-efficient aggregation.                                               #
#                                                                             #
# The old aggregate()/derive_snv_catalog() pair materialized a whole-genome   #
# x all-isolates *string* (U1 -> pandas object) allele matrix, which for a    #
# 5 Mb genome x ~370 isolates peaked well over a 24 GB machine's RAM and got  #
# OOM-killed. We instead:                                                     #
#   1. Pre-allocate two dense arrays once: coverage (bool) and base calls      #
#      encoded as int8 codes (A/C/G/T -> 0..3, missing/ambiguous -> 4).        #
#   2. Stream each isolate into a single column, discarding its per-contig     #
#      arrays immediately (no all-isolates dict held in memory).               #
#   3. Materialize the (Contig, Location)-indexed *string* allele table only   #
#      at polymorphic sites, which are a tiny fraction of the genome.          #
# Coverage stays genome-wide (bool, ~N x n_iso bytes) because dN/dS needs      #
# opportunity denominators at every covered site.                             #
# --------------------------------------------------------------------------- #

# Base <-> int8 code mapping; code 4 == missing / ambiguous / 'N'.
_CODE_OF = {"A": 0, "C": 1, "G": 2, "T": 3}
_BASES = np.array(["A", "C", "G", "T", "N"], dtype="U1")  # index by code 0..4


def build_site_index(ref_records: dict[str, str]):
    """Build the (Contig, Location) site index + per-site reference base codes.

    Only contigs with >= MIN_CONTIG_LEN bp are kept (same filter the old
    aggregate() used). Returns ``(offsets, n_sites, index, ref_codes)`` where
    ``offsets[contig] = (start_row, length)`` locates each contig's block in the
    global per-isolate arrays, and ``ref_codes`` holds the int8 code of the
    reference base at every site.
    """
    offsets: dict[str, tuple[int, int]] = {}
    contig_cols: list[np.ndarray] = []
    loc_cols: list[np.ndarray] = []
    ref_code_parts: list[np.ndarray] = []
    off = 0
    for contig, seq in ref_records.items():
        n = len(seq)
        if n < MIN_CONTIG_LEN:
            continue
        offsets[contig] = (off, n)
        contig_cols.append(np.full(n, contig, dtype=object))
        loc_cols.append(np.arange(1, n + 1, dtype=np.int32))
        ascii_codes = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
        codes = np.full(n, 4, dtype=np.int8)
        for base, k in _CODE_OF.items():
            codes[ascii_codes == ord(base)] = k
        ref_code_parts.append(codes)
        off += n
    if not offsets:
        raise SystemExit(f"no reference contigs >= MIN_CONTIG_LEN ({MIN_CONTIG_LEN})")
    index = pd.MultiIndex.from_arrays(
        [np.concatenate(contig_cols), np.concatenate(loc_cols)],
        names=["Contig", "Location"],
    )
    ref_codes = np.concatenate(ref_code_parts)
    return offsets, off, index, ref_codes


def write_isolate_column(base_mat: np.ndarray, cov_mat: np.ndarray, j: int,
                         offsets: dict[str, tuple[int, int]],
                         cov: dict[str, np.ndarray], ale: dict[str, np.ndarray]) -> None:
    """Fold one isolate's per-contig coverage/allele arrays into column ``j``."""
    for contig, (off, n) in offsets.items():
        cov_mat[off:off + n, j] = cov[contig]
        ale_c = ale[contig]
        codes = np.full(n, 4, dtype=np.int8)
        for base, k in _CODE_OF.items():
            codes[ale_c == base] = k
        base_mat[off:off + n, j] = codes


def derive_tables(base_mat: np.ndarray, cov_mat: np.ndarray, ref_codes: np.ndarray,
                  index: pd.MultiIndex, sample_ids: list[str]
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Find polymorphic sites; build the (small) snv_catalog + poly-site alleles.

    A site is polymorphic if any isolate has a confident A/C/G/T call that
    differs from the reference. Only those rows are turned into a string allele
    table (with a leading ``Ref`` column and ``'NA'`` for uncovered isolates),
    matching the contract ``compute_biallelic_snvs`` expects.
    """
    non_ref = (base_mat != ref_codes[:, None]) & cov_mat & (base_mat < 4)
    poly_mask = non_ref.any(axis=1)
    poly_index = index[poly_mask]
    snv_catalog = pd.DataFrame(non_ref[poly_mask], index=poly_index, columns=sample_ids)
    del non_ref

    ale_str = _BASES[base_mat[poly_mask]]  # (P, n_iso) U1 string array
    poly_alleles = pd.DataFrame(ale_str, index=poly_index, columns=sample_ids)
    # uncovered isolates get 'NA' rather than a base call
    poly_alleles = poly_alleles.where(cov_mat[poly_mask], other="NA")
    poly_alleles.insert(0, "Ref", _BASES[ref_codes[poly_mask]])
    return snv_catalog, poly_alleles


def compute_biallelic(snv_catalog: pd.DataFrame, alleles: pd.DataFrame, coverage: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Polarize biallelic SNVs into 0/1/255 calls with Ref/Major/Alt metadata."""
    from dnds_dynamics.snv_helpers.qp import compute_biallelic_snvs

    poly_alleles = alleles.loc[snv_catalog.index].copy()
    bi_snvs, multi = compute_biallelic_snvs(
        snv_catalog.copy(), poly_alleles.copy(), coverage.copy()
    )
    return bi_snvs, multi


def annotate_sites(bi_snvs: pd.DataFrame, coverage: pd.DataFrame, ref_records: dict[str, str]) -> pd.DataFrame:
    """Annotate every covered position with site type & predicted mutation effects."""
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord

    from dnds_dynamics.snv_helpers.qp import polarize_reference_seq
    from dnds_dynamics.snv_helpers import codon_annotation as annotation_utils

    # build SeqRecord-like list for polarization
    seq_records = []
    for cid, seq in ref_records.items():
        if len(seq) < MIN_CONTIG_LEN:
            continue
        rec = SeqRecord(Seq(seq), id=cid)
        seq_records.append(rec)
    polarized = polarize_reference_seq(seq_records, snv_df=bi_snvs)

    gene_df = _load_gene_df()

    per_contig = []
    for rec in polarized:
        contig = rec.id
        contig_gene_df = gene_df[gene_df["Contig"] == contig]
        df = annotation_utils.annotate_sequence_site_types_to_df(rec.seq, contig_gene_df)
        df["Contig"] = contig
        per_contig.append(df)
    mut_df = pd.concat(per_contig, ignore_index=True)
    mut_df = mut_df.rename(columns={f"{b} Mut": b for b in ["A", "C", "G", "T"]})
    mut_df.set_index(["Contig", "Location"], inplace=True)
    mut_df = mut_df.reindex(coverage.index)
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


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.reset_index() if df.index.names != [None] else df
    out.to_parquet(path, index=False)


def write_core_genes_from_qp(mut_df: pd.DataFrame, species: str, qp_catalog_dir: Path,
                             out_dir: Path, frac: float = 0.5) -> "list | None":
    """Define core genes by **reusing the QP/MIDAS core** (same reference genome).

    A gene is core if at least ``frac`` of its reference sites appear in the matching
    QP catalog's coverage table (= the MIDAS core genome). Writes ``core_genes.json``
    (list of MIDAS gene IDs). Skips gracefully if no QP catalog exists for ``species``.
    """
    qp_cov_path = Path(qp_catalog_dir) / species / "coverage.feather"
    if not qp_cov_path.exists():
        print(f"[core_genes] no QP catalog at {qp_cov_path}; skipping core_genes.json")
        return None
    qp_cov = pd.read_feather(qp_cov_path)
    qp_sites = set(zip(qp_cov["Contig"].astype(str), qp_cov["Location"].astype(int)))
    con = mut_df.index.get_level_values("Contig").to_numpy().astype(str)
    loc = mut_df.index.get_level_values("Location").to_numpy()
    is_qp = np.fromiter(((c, int(l)) in qp_sites for c, l in zip(con, loc)),
                        dtype=bool, count=len(mut_df))
    df = pd.DataFrame({"gene": mut_df["Gene Name"].to_numpy(), "is_qp": is_qp})
    df = df[df["gene"] != "NA"]
    fr = df.groupby("gene")["is_qp"].mean()
    core = sorted(fr.index[fr >= frac].tolist())
    (Path(out_dir) / "core_genes.json").write_text(json.dumps(core))
    print(f"core_genes.json: {len(core)} / {fr.size} genes (>= {frac} of sites in QP catalog)")
    return core


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--species", default=None,
                        help="Species label/slug for outputs and the default isolate dir. "
                             "Defaults to --midas-species; required when using --ref-fna.")
    parser.add_argument("--midas-species", default=None,
                        help="MIDAS rep-genome id; used as the alignment reference when "
                             "--ref-fna is not given (and as the default species label).")
    parser.add_argument("--isolate-dir", default=None,
                        help="NCBI download subdir under /Volumes/Botein/ncbi_isolates "
                             "(default: midas-species with numeric id stripped).")
    parser.add_argument("--ref-fna", type=Path, default=None,
                        help="Explicit reference FASTA (e.g. UHGG reference_genomes/<MGYG>.fna). "
                             "Overrides the MIDAS rep genome; requires --ref-gff.")
    parser.add_argument("--ref-gff", type=Path, default=None,
                        help="GFF annotation matching --ref-fna (e.g. UHGG genes/<MGYG>.gff).")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output table dir (default: <isolate-dir>/snv_table).")
    parser.add_argument("--max-isolates", type=int, default=0, help="0 = all")
    parser.add_argument("--qp-catalog-dir", type=Path,
                        default=Path("/Volumes/Botein/GarudGood2019_snvs/snvs_feather"),
                        help="QP catalog dir; its coverage table defines the reused MIDAS core genes "
                             "(only used when --core-gene-source qp).")
    parser.add_argument("--core-gene-source", choices=("qp", "none"), default="qp",
                        help="How to define core_genes.json (reference-dependent): 'qp' reuses the "
                             "MIDAS/QP catalog core (default; MIDAS references only); 'none' skips it "
                             "(supply core_genes.json yourself for a non-MIDAS reference).")
    args = parser.parse_args()

    if (args.ref_fna is None) != (args.ref_gff is None):
        raise SystemExit("--ref-fna and --ref-gff must be given together.")
    if args.ref_fna is None and not args.midas_species:
        raise SystemExit("need a reference: pass --midas-species (MIDAS rep genome), "
                         "or --ref-fna + --ref-gff for any other reference.")
    species = args.species or args.midas_species
    if not species:
        raise SystemExit("provide --species (the label) when using --ref-fna.")
    configure_paths(species, args.midas_species, args.isolate_dir, args.out_dir,
                    ref_fna=args.ref_fna, ref_gff=args.ref_gff)
    out_dir = args.out_dir or DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"species={SPECIES}  isolates={ISOLATE_ROOT}  out={out_dir}")
    if not REF_FNA_GZ.exists():
        raise SystemExit(f"reference not found: {REF_FNA_GZ}")
    if not MANIFEST.exists():
        raise SystemExit(f"manifest not found: {MANIFEST} (run download_ncbi_isolates.py first)")
    args.out_dir = out_dir

    # Prefer manifest_to_download.tsv (what was actually downloaded) over the
    # full manifest_all_isolates.tsv — for prolific species (e.g. S. enterica,
    # 31k HQ assemblies) the full manifest is orders of magnitude larger than
    # the downloaded subset, and only downloaded FASTAs can be built anyway.
    manifest_dl = ISOLATE_ROOT / "manifest_to_download.tsv"
    manifest_path = manifest_dl if manifest_dl.exists() else MANIFEST
    manifest = pd.read_csv(manifest_path, sep="\t")
    isolates = manifest[manifest["tier"].isin(ALLOWED_TIERS)].copy()
    if args.max_isolates:
        isolates = isolates.head(args.max_isolates)
    print(f"using {len(isolates)} isolates from {manifest_path.name} "
          f"({sorted(isolates['tier'].unique())})")

    ref_records = read_fasta_gz(REF_FNA_GZ)
    print(f"reference: {len(ref_records)} contigs, {sum(len(s) for s in ref_records.values())} bp")

    offsets, n_sites, index, ref_codes = build_site_index(ref_records)

    # Determine the matrix columns (isolates with a FASTA on disk) up front so
    # we can pre-allocate and stream into them rather than holding all isolates.
    present: list[tuple[str, Path, str]] = []
    for row in isolates.itertuples(index=False):
        acc = row.accession
        cand = list(FASTA_DIR.glob(f"{acc}.fna.gz"))
        if not cand:
            print(f"  {acc}: FASTA missing; skipping")
            continue
        present.append((acc, cand[0], row.tier))
    sample_ids = [acc for acc, _, _ in present]
    n_iso = len(sample_ids)
    if n_iso == 0:
        raise SystemExit("no isolate FASTAs found to align")
    print(f"aligning {n_iso} isolates over {n_sites:,} reference sites "
          f"(~{n_sites * n_iso * 2 / 1e9:.1f} GB for coverage+base matrices)")

    # Two dense arrays, allocated once and filled column-by-column.
    cov_mat = np.zeros((n_sites, n_iso), dtype=bool)
    base_mat = np.full((n_sites, n_iso), 4, dtype=np.int8)  # 4 == missing

    # write decompressed reference once
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ref_path = tmp / "ref.fna"
        write_fasta(ref_records, ref_path)

        align_stats = []
        for j, (acc, fasta_path, tier) in enumerate(present):
            t0 = time.perf_counter()
            iso_records = read_fasta_gz(fasta_path)
            iso_path = tmp / f"{acc}.fna"
            write_fasta(iso_records, iso_path)
            delta_filt, snps, coords = run_alignment(ref_path, iso_path, tmp, prefix=acc.replace(".", "_"))
            snps_df = parse_snps(snps)
            coords_df = parse_coords(coords)
            cov, ale = isolate_arrays(coords_df, snps_df, ref_records)
            write_isolate_column(base_mat, cov_mat, j, offsets, cov, ale)
            dt = time.perf_counter() - t0
            cov_total = sum(int(cov[c].sum()) for c in cov)
            n_snps = int(len(snps_df))
            align_stats.append({"accession": acc, "tier": tier, "covered_bp": cov_total, "n_snps": n_snps, "secs": round(dt, 2)})
            # release this isolate's arrays before moving on
            del cov, ale, iso_records, snps_df, coords_df
            if (j + 1) % 5 == 0 or (j + 1) == n_iso:
                print(f"  {j + 1}/{n_iso}: {acc} aligned ({cov_total} bp covered, {n_snps} snps, {dt:.1f}s)")
            iso_path.unlink(missing_ok=True)

    pd.DataFrame(align_stats).to_csv(args.out_dir / "alignment_stats.csv", index=False)

    # Polymorphic sites (small) materialized as strings; coverage stays dense.
    snv_catalog, poly_alleles = derive_tables(base_mat, cov_mat, ref_codes, index, sample_ids)
    print(f"snv_catalog: {snv_catalog.shape} polymorphic sites")

    coverage = pd.DataFrame(cov_mat, index=index, columns=sample_ids)
    del base_mat, cov_mat
    print(f"coverage: {coverage.shape}")

    bi_snvs, multi_sites = compute_biallelic(snv_catalog, poly_alleles, coverage)
    print(f"biallelic_snvs: {bi_snvs.shape}; multi-allelic sites: {int(multi_sites.sum())}")

    save_parquet(coverage, args.out_dir / "coverage.parquet")
    save_parquet(poly_alleles, args.out_dir / "alleles.parquet")
    save_parquet(snv_catalog, args.out_dir / "snv_catalog.parquet")
    save_parquet(bi_snvs, args.out_dir / "biallelic_snvs.parquet")

    mut_df = annotate_sites(bi_snvs, coverage, ref_records)
    save_parquet(mut_df, args.out_dir / "site_annotations.parquet")
    print(f"site_annotations: {mut_df.shape}; site type counts: "
          f"{mut_df['Site Type'].value_counts().to_dict()}")

    # Core genes are reference- and annotation-dependent (see --core-gene-source).
    # 'qp' reuses the MIDAS/QP core (same reference, keyed by the MIDAS id), which
    # excludes type-strain accessory content most isolates lack, otherwise
    # inflating coverage gaps.
    if args.core_gene_source == "qp":
        if not args.midas_species:
            raise SystemExit("--core-gene-source qp needs --midas-species (the QP catalog key); "
                             "use --core-gene-source none for a non-MIDAS reference.")
        write_core_genes_from_qp(mut_df, args.midas_species, args.qp_catalog_dir, args.out_dir)
    else:
        print("[core_genes] --core-gene-source none: not writing core_genes.json. "
              "Supply it yourself (list of Gene Name values from site_annotations) "
              "before running recombination on a non-MIDAS reference.")

    metadata = {
        "species": SPECIES,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reference_fna": str(REF_FNA_GZ),
        "reference_features": str(REF_FEATURES_GZ),
        "isolate_root": str(ISOLATE_ROOT),
        "n_isolates": n_iso,
        "isolate_tiers": pd.Series([t for _, _, t in present]).value_counts().to_dict(),
        "aligner": "nucmer (mummer-4.0.0rc1) + delta-filter -1",
        "min_align_identity": MIN_ALIGN_IDENT,
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
