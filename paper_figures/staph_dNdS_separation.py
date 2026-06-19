"""Supplementary figure: S. aureus dN/dS separation (Full / Recombined / Clonal).

Faithful port of the figure-producing cells of
dNdS_dynamics/exploratory/2023-10-28_staph/dNdS_analysis.ipynb (-> Staph_dNdS.pdf).
The isolate analog of Fig 2: three panels of dN/dS vs full-genome dS for S. aureus
isolate pairs, with recombination masked HEURISTICALLY from hardcoded ST-lineage
event coordinates (eyeballed from SNP-density plots in the source notebook).

Data (large; not vendored) lives in config.STAPH_DATA_DIR:
  variants.npy (site types), Saureus.fasta (MSA), div_mat.npy, cf_mat.npy.
Writes config.fig_path / 'Staph_dNdS.pdf'.
"""
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from Bio import SeqIO

from dnds_dynamics import config
from dnds_dynamics.figures.dynamics import computed_poisson_thinning

base_dir = config.STAPH_DATA_DIR

# --- load alignment + annotations ---
variants = np.load(base_dir / "variants.npy").astype(str)
records = list(SeqIO.parse(str(base_dir / "Saureus.fasta"), "fasta"))
seqs = [np.array(record.seq) for record in records]
seq_ids = np.array([record.id for record in records])
genome_len = len(seqs[0])

div_mat = np.load(base_dir / "div_mat.npy")
clonal_frac_mat = np.load(base_dir / "cf_mat.npy")

# --- select close (cf>0.5) and typical (cf<0.5, 500 sampled) pairs ---
# NOTE: the source notebook left random.sample unseeded; seed here so the script
# is reproducible. The figure is statistical, so this does not change conclusions.
random.seed(0)
idxs = np.where(clonal_frac_mat > 0.5)
close_pairs = [(int(a), int(b)) for a, b in zip(idxs[0], idxs[1]) if a < b]
idxs = np.where(clonal_frac_mat < 0.5)
typical_pairs = [(int(a), int(b)) for a, b in zip(idxs[0], idxs[1]) if a < b]
typical_pairs = random.sample(typical_pairs, 500)

species_name = "Staph"

# --- heuristic per-ST-lineage recombination masks (eyeballed in source notebook) ---
ST34 = ["C1129", "C1115", "C1102", "C1100", "C1090", "C1122"]
ST582 = ["C1142", "C1158"]
ST239 = ["C9669"]


def prepare_clonal_mask_from_recomb(genome_len, events):
    mask = np.ones(genome_len).astype(bool)
    for start, end in events:
        mask[int(start):int(end)] = False
    return mask


ST34_mask = prepare_clonal_mask_from_recomb(genome_len, [(0, 1.4e5), (genome_len - 1.15e5, genome_len)])
ST239_mask = prepare_clonal_mask_from_recomb(genome_len, [(0, 4e5), (genome_len - 2.35e5, genome_len)])
ST582_mask = prepare_clonal_mask_from_recomb(genome_len, [(845000, 1155000)])

id_to_type = dict([(x, "ST34") for x in ST34] + [(x, "ST582") for x in ST582] + [(x, "ST239") for x in ST239])
type_to_mask = {"ST34": ST34_mask, "ST582": ST582_mask, "ST239": ST239_mask, "other": np.ones(genome_len).astype(bool)}


def get_clonal_mask(id1, id2):
    type1 = id_to_type.get(id1, "other")
    type2 = id_to_type.get(id2, "other")
    if type1 != type2:
        return type_to_mask[type1] & type_to_mask[type2]
    return type_to_mask["other"]


def pairwise_snps(seq1, seq2):
    covered1 = seq1 != "-"
    covered2 = seq2 != "-"
    return seq1 != seq2, covered1 & covered2


# --- compute per-pair dN/dS (full / recomb / clonal) ---
pairs_to_process = close_pairs + typical_pairs
dat = []
for pair in pairs_to_process:
    seq1, seq2 = seqs[pair[0]], seqs[pair[1]]
    snp_vec, covered = pairwise_snps(seq1, seq2)
    snp_vec = snp_vec[covered]
    covered_variants = variants[covered]

    mask_4D = covered_variants == "4D"
    mask_1D = covered_variants == "1D"
    core_div = snp_vec.mean()
    core_len = len(snp_vec)
    core_len_4D = len(snp_vec[mask_4D])
    core_len_1D = len(snp_vec[mask_1D])
    core_diff_4D = np.sum(snp_vec[mask_4D])
    core_diff_1D = np.sum(snp_vec[mask_1D])

    if pair in close_pairs:
        clonal_mask = get_clonal_mask(seq_ids[pair[0]], seq_ids[pair[1]])[covered]
        recomb_mask = ~clonal_mask
        recomb_len_4D = len(snp_vec[mask_4D & recomb_mask])
        recomb_diff_4D = np.sum(snp_vec[mask_4D & recomb_mask])
        recomb_len_1D = len(snp_vec[mask_1D & recomb_mask])
        recomb_diff_1D = np.sum(snp_vec[mask_1D & recomb_mask])
        clonal_len_4D = len(snp_vec[mask_4D & clonal_mask])
        clonal_diff_4D = np.sum(snp_vec[mask_4D & clonal_mask])
        clonal_len_1D = len(snp_vec[mask_1D & clonal_mask])
        clonal_diff_1D = np.sum(snp_vec[mask_1D & clonal_mask])
        dat.append((species_name, pair, core_div, core_len,
                    core_len_4D, core_len_1D, core_diff_4D, core_diff_1D,
                    recomb_len_4D, recomb_len_1D, recomb_diff_4D, recomb_diff_1D,
                    clonal_len_4D, clonal_len_1D, clonal_diff_4D, clonal_diff_1D))
    else:
        dat.append((species_name, pair, core_div, core_len,
                    core_len_4D, core_len_1D, core_diff_4D, core_diff_1D,
                    np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))

dnds_df = pd.DataFrame(dat)
dnds_df.columns = ["species_name", "pair", "core_div", "core_len",
                   "core_len_4D", "core_len_1D", "core_diff_4D", "core_diff_1D",
                   "recomb_len_4D", "recomb_len_1D", "recomb_diff_4D", "recomb_diff_1D",
                   "clonal_len_4D", "clonal_len_1D", "clonal_diff_4D", "clonal_diff_1D"]

# --- three-panel figure ---
mpl.rcParams["font.size"] = 7
mpl.rcParams["lines.linewidth"] = 1
mpl.rcParams["legend.fontsize"] = "small"
mpl.rcParams["legend.frameon"] = False

fig, axes = plt.subplots(1, 3, figsize=(8, 1.8), dpi=200)

close_mask = dnds_df["pair"].isin(close_pairs)
close_df = dnds_df[close_mask]

dS1, dS2 = computed_poisson_thinning(dnds_df["core_diff_4D"], dnds_df["core_len_4D"])
naive_dS = dnds_df["core_diff_4D"] / dnds_df["core_len_4D"].astype(float)
dN = dnds_df["core_diff_1D"] / dnds_df["core_len_1D"].astype(float)
axes[0].scatter(dS1, dN / dS2, s=1, alpha=0.2, color="tab:grey", rasterized=True)

recomb_mask = close_df["recomb_len_4D"] > 0
naive_recomb_dS = close_df["recomb_diff_4D"] / close_df["recomb_len_4D"].astype(float)
naive_recomb_dS = naive_recomb_dS[recomb_mask]
recomb_dN = close_df["recomb_diff_1D"] / close_df["recomb_len_1D"].astype(float)
recomb_dN = recomb_dN[recomb_mask]
axes[1].scatter(dS1[close_mask & recomb_mask], recomb_dN / naive_recomb_dS, s=1, alpha=0.3, color="#FF968D", rasterized=True)

naive_clonal_dS = close_df["clonal_diff_4D"] / close_df["clonal_len_4D"].astype(float)
clonal_dN = close_df["clonal_diff_1D"] / close_df["clonal_len_1D"].astype(float)
axes[2].scatter(dS1[close_mask], clonal_dN / naive_clonal_dS, s=1, alpha=0.3, color="#AECDE1", rasterized=True)

typical_df = dnds_df[~close_mask]
dS1t, dS2t = computed_poisson_thinning(typical_df["core_diff_4D"], typical_df["core_len_4D"])
dNt = typical_df["core_diff_1D"] / typical_df["core_len_1D"].astype(float)
dNdS = np.mean(dNt / dS2t)

for ax in axes:
    ax.set_ylim([1e-2, 1e1])
    ax.set_xlim([2e-6, 5e-2])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$dS$ (Full genome)")
    ax.axhline(1, linewidth=0.5, linestyle="--", color="grey", zorder=-1)
    ax.axhline(dNdS, linewidth=0.5, linestyle="-.", color="k", zorder=-1)

axes[0].set_ylabel("$dN/dS$")
axes[0].set_title("Full genome")
axes[1].set_title("Recombined regions")
axes[2].set_title("Clonal regions")
config.fig_path.mkdir(parents=True, exist_ok=True)
fig.savefig(config.fig_path / "Staph_dNdS.pdf", bbox_inches="tight")
