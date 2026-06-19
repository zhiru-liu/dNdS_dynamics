"""Simple genome-wide view of A. putredinis clonal 1D (nonsynonymous) hotspots.

Per 1-kb window, the pooled number of clonal nonsynonymous (1D) differences
across all QP close pairs (after removing CP-HMM-detected recombination). Top
peaks are annotated with their RefSeq gene product.
"""
from __future__ import annotations
import os, re, sys
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO / ".cache" / "matplotlib"))
LIUGOOD = Path("/Users/Device6/Documents/Research/bgoodlab/LiuGood2024_data")
FEATHER = Path("/Volumes/Botein/GarudGood2019_snvs/snvs_feather")
GFF = Path("/tmp/gffq/ncbi_dataset/data/GCF_000154465.1/genomic.gff")
COUNTER = Path("/tmp/qp_counter.npy")  # per-snv-site clonal recurrence over close pairs
FIG = REPO / "figures" / "aputredinis_1D_hotspot_genomeview"
WIN = 1000
N_LABEL = 10


def load_qp():
    cwd = os.getcwd()
    try:
        os.chdir(LIUGOOD); sys.path.insert(0, str(LIUGOOD))
        from snv_utils import SNVHelper
    finally:
        os.chdir(cwd)
    return SNVHelper("Alistipes_putredinis_61533", snv_path=FEATHER, snv_format="feather",
                     compute_bi_snvs=False, annotate=True, mask_multi_sites=True)


def refseq_product(cds, contig, lo, hi):
    ov = cds[(cds.contig == contig + ".1") & (cds.end >= lo) & (cds.start <= hi)].copy()
    if ov.empty:
        return "(intergenic)"
    # gene with the largest overlap with the window
    ov["ovl"] = np.minimum(ov.end, hi) - np.maximum(ov.start, lo)
    return ov.sort_values("ovl", ascending=False)["product"].iloc[0]


def main():
    qp = load_qp()
    snv = qp.snvs.index
    sc = snv.get_level_values("Contig").to_numpy().astype(str)
    sp = snv.get_level_values("Location").to_numpy().astype(int)
    s1 = qp.snv_1D.to_numpy()
    counter = np.load(COUNTER)
    assert len(counter) == len(snv)

    # per-contig 1-kb windows (k = pos//WIN), summing clonal 1D recurrence
    contigs = list(dict.fromkeys(sc))
    off, cum = {}, 0
    for c in contigs:
        off[c] = cum; cum += sp[sc == c].max() + 20000
    d1 = pd.DataFrame({"contig": sc[s1], "k": (sp[s1] // WIN), "w": counter[s1]})
    wd = d1.groupby(["contig", "k"], as_index=False)["w"].sum()
    wd["gx"] = wd["k"] * WIN + WIN / 2 + wd["contig"].map(off)
    wd = wd.sort_values("gx").reset_index(drop=True)
    centers = wd["gx"].to_numpy()
    dens = wd["w"].to_numpy(float)

    # RefSeq CDS
    rows = []
    for line in open(GFF):
        if line.startswith("#"):
            continue
        f = line.rstrip("\n").split("\t")
        if len(f) < 9 or f[2] != "CDS":
            continue
        p = re.search(r"product=([^;]+)", f[8])
        rows.append((f[0], int(f[3]), int(f[4]), p.group(1) if p else "?"))
    cds = pd.DataFrame(rows, columns=["contig", "start", "end", "product"])

    # pick top peaks, dedup by gene product (label the highest window per product)
    order = np.argsort(dens)[::-1]
    labels, seen = [], set()
    for wi in order:
        if dens[wi] <= 0:
            continue
        row = wd.iloc[wi]
        c = row["contig"]; lo = int(row["k"]) * WIN; hi = lo + WIN
        prod = refseq_product(cds, c, lo, hi)
        key = prod.lower()
        if key in seen:
            continue
        seen.add(key)
        labels.append((centers[wi], dens[wi], prod))
        if len(labels) >= N_LABEL:
            break

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(13, 4.5), dpi=300)
    # contig boundaries
    for c in contigs:
        ax.axvline(off[c], color="0.9", lw=0.5, zorder=0)
    ax.vlines(centers, 0, dens, color="#c0392b", lw=0.5, zorder=2)
    ymax = dens.max()
    for gx_, y, prod in labels:
        ax.annotate(prod, xy=(gx_, y), xytext=(gx_, y + ymax * 0.04),
                    rotation=90, ha="center", va="bottom", fontsize=7, color="0.15",
                    arrowprops=dict(arrowstyle="-", color="0.5", lw=0.5))
    ax.set_ylim(0, ymax * 1.6)
    ax.set_xlim(0, cum)
    ax.set_xlabel("position along concatenated core genome (bp)")
    ax.set_ylabel("pooled clonal 1D (nonsyn)\ndifferences per 1 kb")
    ax.set_title("A. putredinis clonal nonsynonymous (1D) hotspots across the genome (QP cohort)")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG.with_suffix(f".{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIG.with_suffix('.png')}")
    print("\ntop labelled peaks:")
    for gx_, y, prod in labels:
        print(f"  {int(y):6d}  {prod}")


if __name__ == "__main__":
    main()
