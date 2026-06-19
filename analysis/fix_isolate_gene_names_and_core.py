"""Repair the per-site ``Gene Name`` annotation and write a QP/MIDAS core-gene list.

Older isolate-table builds stored a broken constant ``Gene Name`` (e.g. '44') for
all coding sites, so the gene-level core mask in ``IsolateSNVHelper`` could not work.
This recomputes ``Gene Name`` positionally from the MIDAS reference features (the
same CDS / complete-reading-frame rule the annotator uses), patches
``site_annotations.parquet`` in place (after a .bak backup), and derives
``core_genes.json`` by **reusing the QP/MIDAS core** -- genes whose reference sites
are present in the matching QP catalog (same reference genome).

Usage:
    python analysis/fix_isolate_gene_names_and_core.py \
        --table-root /Volumes/Botein/ncbi_isolates/Alistipes_putredinis \
        --midas-species Alistipes_putredinis_61533
"""
from __future__ import annotations

import argparse
import gzip
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

QP_DEFAULT = Path("/Volumes/Botein/GarudGood2019_snvs/snvs_feather")
REF_DEFAULT = Path("/Volumes/Botein/GarudGood2019_snvs/midas_db_data/rep_genomes")
MIN_CONTIG_LEN = 2000
CORE_GENE_QP_FRAC = 0.5  # gene is core if >= this fraction of its sites are QP-core


def load_gene_df(features_gz: Path) -> pd.DataFrame:
    g = pd.read_csv(features_gz, sep="\t")
    g.columns = ["Gene ID", "Contig", "Start", "End", "Strand", "Type", "Info"]
    g["Gene ID"] = g["Gene ID"].astype(str)
    return g


def contig_gene_name_array(gene_df: pd.DataFrame, contig: str, contig_len: int) -> np.ndarray:
    """Per-position gene-ID array for one contig (CDS, complete frame, last wins)."""
    names = np.full(contig_len, "NA", dtype=object)
    cds = gene_df[(gene_df["Contig"] == contig) & (gene_df["Type"] == "CDS")]
    for _, row in cds.iterrows():
        start, end = int(row["Start"]), int(row["End"])
        if (end - start + 1) % 3 != 0:
            continue
        names[start - 1:end] = row["Gene ID"]
    return names


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table-root", required=True, type=Path)
    ap.add_argument("--midas-species", required=True)
    ap.add_argument("--snv-table", default="snv_table")
    ap.add_argument("--qp-catalog-dir", type=Path, default=QP_DEFAULT)
    ap.add_argument("--reference-dir", type=Path, default=REF_DEFAULT)
    ap.add_argument("--dry-run", action="store_true", help="report only; don't write")
    args = ap.parse_args()

    table_dir = args.table_root / args.snv_table
    sa_path = table_dir / "site_annotations.parquet"
    features = args.reference_dir / args.midas_species / "genome.features.gz"
    ref_fna = args.reference_dir / args.midas_species / "genome.fna.gz"

    gene_df = load_gene_df(features)

    # contig lengths from the reference fasta
    from Bio import SeqIO
    contig_len = {r.id: len(r.seq) for r in SeqIO.parse(gzip.open(ref_fna, "rt"), "fasta")}

    # --- 1. recompute Gene Name per site -------------------------------------
    sa = pd.read_parquet(sa_path)
    old_unique = sa["Gene Name"].nunique()
    new_names = np.full(len(sa), "NA", dtype=object)
    loc = sa["Location"].to_numpy()
    con = sa["Contig"].to_numpy().astype(str)
    for c in pd.unique(con):
        if c not in contig_len or contig_len[c] < MIN_CONTIG_LEN:
            continue
        arr = contig_gene_name_array(gene_df, c, contig_len[c])
        sel = np.flatnonzero(con == c)
        idx = loc[sel] - 1
        ok = (idx >= 0) & (idx < len(arr))
        new_names[sel[ok]] = arr[idx[ok]]
    sa_new_gene = pd.Series(new_names, index=sa.index)
    coding = sa["Site Type"].isin(["1D", "2D", "3D", "4D"])
    print(f"Gene Name: {old_unique} -> {sa_new_gene.nunique()} distinct "
          f"({(sa_new_gene[coding] != 'NA').sum()}/{coding.sum()} coding sites now have a gene ID)")

    # --- 2. QP/MIDAS core gene set -------------------------------------------
    qp_cov = pd.read_feather(args.qp_catalog_dir / args.midas_species / "coverage.feather")
    qp_sites = set(zip(qp_cov["Contig"].astype(str), qp_cov["Location"].astype(int)))
    print(f"QP catalog core sites: {len(qp_sites)}")
    sa_is_qp = np.fromiter(((cc, int(ll)) in qp_sites for cc, ll in zip(con, loc)),
                           dtype=bool, count=len(sa))
    df = pd.DataFrame({"gene": sa_new_gene.to_numpy(), "is_qp": sa_is_qp})
    df = df[df["gene"] != "NA"]
    frac = df.groupby("gene")["is_qp"].mean()
    core_genes = sorted(frac.index[frac >= CORE_GENE_QP_FRAC].tolist())
    print(f"core genes (>= {CORE_GENE_QP_FRAC} of sites in QP catalog): {len(core_genes)} / {frac.size}")

    if args.dry_run:
        print("[dry-run] not writing")
        return

    # --- 3. write (with backup) ----------------------------------------------
    bak = sa_path.with_suffix(".parquet.broken_gene_name_bak")
    if not bak.exists():
        shutil.copy2(sa_path, bak)
        print(f"backed up original site_annotations -> {bak.name}")
    sa["Gene Name"] = sa_new_gene.to_numpy()
    sa.to_parquet(sa_path, index=False)
    core_path = table_dir / "core_genes.json"
    core_path.write_text(json.dumps(core_genes))
    print(f"wrote {core_path.name} ({len(core_genes)} genes) and patched {sa_path.name}")


if __name__ == "__main__":
    main()
