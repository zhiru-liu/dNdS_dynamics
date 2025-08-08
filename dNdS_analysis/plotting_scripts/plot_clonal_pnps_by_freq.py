import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import seaborn as sns

import dNdS_analysis.config as config
from dNdS_analysis.utils import dynamics_utils
from dNdS_analysis.utils import snv_utils

def dedup_clonal_pairs(pairs_df, snv_helper=None, identical_frac_threshold=0.2):
    """
    Function to keep only one pair of clonal samples from each group of closely related samples.

    For example, if strains 1,2,3 are all within a clonal complex, there will be
    three clonal pairs: (1,2), (1,3), (2,3). This function will keep only one of them.
    """
    if snv_helper is None:
        snv_helper = snv_utils.SNVHelper(pairs_df.index.get_level_values('species_name').unique()[0], compute_bi_snvs=False, annotate=True)
        snv_helper.load_identical_block()
    included_pairs = []
    for pair, row in pairs_df.iterrows():
        # check if pair is close to any of the included pairs
        close_to_included = False
        for included_pair in included_pairs:
            sample1 = included_pair[0]
            sample2 = pair[0]
            dist = snv_helper.identical_block_frac.loc[sample1, sample2]
            if dist > identical_frac_threshold:
                close_to_included = True
                break
        # if not close to any included pair, add it
        if not close_to_included:
            included_pairs.append(pair)
        else:
            continue
    return included_pairs


complete_df = dynamics_utils.load_dNdS_data()

# filter for pairs that are clonal (i.e. no detected recombination)
clonal_mask = complete_df['recomb_len_4D'].isna()
passed_pairs = complete_df[clonal_mask]

# focus on a single example species
species_name = 'Bacteroides_vulgatus_57955'
species_clonal_pairs = passed_pairs.loc[species_name]

snv_helper = snv_utils.SNVHelper(species_name, compute_bi_snvs=False, annotate=True)
snv_helper.load_identical_block()

# then filter to only one pair per clonal cluster 
dedup_pairs = dedup_clonal_pairs(species_clonal_pairs, snv_helper)

# This species has two subspecies / clades (vulgatus and dorei), so we will focus on the major clade
clades = pd.read_csv(config.data_path / 'Bv_clades.txt', sep='\t',index_col=0, header=None, names=['sample','clade'])
major_clade_samples = clades[clades['clade'] == 'major'].index

# now select the samples to be included when calculating the population prevalence
# first start with the clonal pairs
samples = [x for tup in dedup_pairs for x in tup]
unique_samples = list(set(samples))

# then include additional samples iteratively, making sure that they are not closely
# related to the already included samples
included_samples = [sample for sample in unique_samples if sample in major_clade_samples]
for sample in major_clade_samples:
    if sample in included_samples:
        continue
    # then check the distance to the included samples
    # access the identical block fraction matrix directly
    similarity = snv_helper.identical_block_frac.loc[included_samples, sample]
    # if has >50 identical fraction with any included sample, skip
    if similarity.max() < 0.5:
        included_samples.append(sample)
        continue

# make a identical fraction clustermap for quick sanity check
sns.clustermap(snv_helper.identical_block_frac.loc[included_samples, included_samples],)
plt.savefig(config.fig_path / f'Bv_clonal_identical_fraction_clustermap.pdf', bbox_inches='tight')

# Now start to analyze the population prevalence of clonal SNVs

# First identify clonal SNVs
all_clonal_diffs = pd.DataFrame(index=snv_helper.snvs.index)
# dedup pairs are unique clonal pairs with no other clonal strains
for pair in dedup_pairs:
    if pair[0] not in major_clade_samples:
        continue
    pair_snvs = snv_helper.compute_pairwise_snvs(pair[0], pair[1])
    all_clonal_diffs[pair] = pair_snvs
clonal_snvs = all_clonal_diffs[all_clonal_diffs.sum(axis=1)>0]

# Next filter the haplotype matrix to clonal SNVs and included samples
haps_all = snv_helper.snvs.loc[:, included_samples]
num_alt_all = (haps_all==1).sum(axis=1)
num_covered_all = (haps_all!=255).sum(axis=1)
snv_filter = (num_alt_all > 0) & (num_alt_all < num_covered_all)
haps_all = haps_all.loc[snv_filter, :]
num_alt_all = num_alt_all[snv_filter]
num_covered_all = num_covered_all[snv_filter]

haps = haps_all.loc[clonal_snvs.index, :]
num_alt = (haps==1).sum(axis=1)
num_covered = (haps!=255).sum(axis=1)
num_host_sampled = haps.shape[1]

# Now calculate for 1D and 4D sites separately
def repolarize_counts(alts, covs):
    m = alts > (covs / 2)
    alts = alts.copy()
    alts[m] = covs[m] - alts[m]
    return alts

core_1D_sites = snv_helper.core_1D[snv_helper.core_1D].index
core_4D_sites = snv_helper.core_4D[snv_helper.core_4D].index

mask_1d = num_alt_all.index.isin(core_1D_sites)
mask_4d = num_alt_all.index.isin(core_4D_sites)
num_alt_all_1D = num_alt_all.loc[mask_1d]
num_alt_all_4D = num_alt_all.loc[mask_4d]
num_covered_all_1D = num_covered_all.loc[mask_1d]
num_covered_all_4D = num_covered_all.loc[mask_4d]

mask_1d = num_alt.index.isin(core_1D_sites)
mask_4d = num_alt.index.isin(core_4D_sites)
num_alt_1D = num_alt.loc[mask_1d]
num_alt_4D = num_alt.loc[mask_4d]
num_covered_1D = num_covered.loc[mask_1d]
num_covered_4D = num_covered.loc[mask_4d]

# repolarize counts (because we are using only major clade samples, some minor alleles are flipped)
num_alt_all_1D = repolarize_counts(num_alt_all_1D, num_covered_all_1D)
num_alt_all_4D = repolarize_counts(num_alt_all_4D, num_covered_all_4D)
num_alt_1D = repolarize_counts(num_alt_1D, num_covered_1D)
num_alt_4D = repolarize_counts(num_alt_4D, num_covered_4D)

# calculate the total number of eligible 1D and 4D sites
L1d = snv_helper.core_1D.sum()
L4d = snv_helper.core_4D.sum()


# Codes for binning the SNVs into singletons, low frequency, and high frequency
freq_edges  = np.array([0.00, 0.10, 0.50])  # right closed
freq_labels = ["<10%", "10-50%"]    # len = len(freq_edges)‑1

def label_sites(alt, cov):
    """
    Return a categorical Series whose values are
       'singleton', '<5 %', '5–10 %', or '10–50 %'.
    """
    # Avoid divide‑by‑zero; keep only sites with coverage ≥1
    m = cov > 0
    alt, cov = alt[m], cov[m]

    # start with a numpy array of empty strings
    out = np.empty(alt.size, dtype=object)

    # ① singletons
    is_singleton = alt == 1
    out[is_singleton] = "singleton"

    # ② everything else → place by frequency
    freq = alt[~is_singleton] / cov[~is_singleton]
    cat  = pd.cut(freq,
                  bins=freq_edges,
                  labels=freq_labels,
                  include_lowest=False,   # 0 belongs only to singletons
                  right=True)            # right-closed buckets
    out[~is_singleton] = cat.astype(str)

    return pd.Series(out, dtype="category")
# ------------------------------------------------------------
# 1D all
cats_1d_all    = label_sites(num_alt_all_1D,    num_covered_all_1D)
# 4D all
cats_4d_all    = label_sites(num_alt_all_4D,    num_covered_all_4D)
# 1D clonal
cats_1d_clonal = label_sites(num_alt_1D, num_covered_1D)
# 4D clonal
cats_4d_clonal = label_sites(num_alt_4D, num_covered_4D)

# Finally, plotting
from scipy.stats import poisson

# ------------------------------------------------------------------
# 0️⃣  Recompute *counts* and *proportions*
# ------------------------------------------------------------------
order = ['singleton'] + freq_labels                # the bin order you already use

def counts_and_prop(series, total_sites):
    """Return (counts, prop) arrays in the fixed order."""
    counts = series.value_counts().reindex(order, fill_value=0).to_numpy()
    prop   = counts / total_sites
    return counts, prop

cnt_1d_clonal, prop_1d_clonal = counts_and_prop(cats_1d_clonal, L1d)
cnt_4d_clonal, prop_4d_clonal = counts_and_prop(cats_4d_clonal, L4d)
cnt_1d_all,    prop_1d_all    = counts_and_prop(cats_1d_all,    L1d)
cnt_4d_all,    prop_4d_all    = counts_and_prop(cats_4d_all,    L4d)

# ------------------------------------------------------------------
# 1️⃣  Helper: Poisson proportion error bars
# ------------------------------------------------------------------
def poisson_prop_err(counts, total, conf=0.95):
    """Return a (2, n) array of asymmetric errors for plt.errorbar."""
    lo, hi = poisson.interval(conf, counts)       # exact Poisson CI on counts
    prop   = counts / total
    lo_p   = prop - lo / total
    hi_p   = hi   / total - prop
    return np.vstack([lo_p, hi_p])

err_1d_clonal = poisson_prop_err(cnt_1d_clonal, L1d)
err_4d_clonal = poisson_prop_err(cnt_4d_clonal, L4d)
err_1d_all    = poisson_prop_err(cnt_1d_all,    L1d)
err_4d_all    = poisson_prop_err(cnt_4d_all,    L4d)

# ------------------------------------------------------------------
# 2️⃣  Plot – two stacked panels with error bars
# ------------------------------------------------------------------
x      = np.arange(len(order))
width  = 0.35

fig, axes = plt.subplots(
    nrows=2, ncols=1, figsize=(7, 6),
    sharex=True, gridspec_kw={'hspace': 0.2}
)

# ---- Panel A: Clonal ----
ax = axes[0]
ax.bar(x - width/2, prop_1d_clonal, width,
       label="1D", color="tab:blue", alpha=0.9)
ax.bar(x + width/2, prop_4d_clonal, width,
       label="4D", color="tab:blue", alpha=0.4)

# Add error bars
ax.errorbar(x - width/2, prop_1d_clonal, yerr=err_1d_clonal,
            fmt='none', ecolor='k', capsize=3, lw=1)
ax.errorbar(x + width/2, prop_4d_clonal, yerr=err_4d_clonal,
            fmt='none', ecolor='k', capsize=3, lw=1)

ax.set_ylabel("Fraction of sites")
ax.set_title("Clonal SNVs")
ax.legend(frameon=False)

# ---- Panel B: All ----
ax = axes[1]
ax.bar(x - width/2, prop_1d_all, width,
       label="1D", color="tab:orange", alpha=0.9)
ax.bar(x + width/2, prop_4d_all, width,
       label="4D", color="tab:orange", alpha=0.4)

ax.errorbar(x - width/2, prop_1d_all, yerr=err_1d_all,
            fmt='none', ecolor='k', capsize=3, lw=1)
ax.errorbar(x + width/2, prop_4d_all, yerr=err_4d_all,
            fmt='none', ecolor='k', capsize=3, lw=1)

ax.set_ylabel("Fraction of sites")
ax.set_xlabel("Allele prevalence across sampled hosts (n={})".format(num_host_sampled))
ax.set_title("All SNVs")
ax.legend(frameon=False)
ax.set_ylim(ymax=ax.get_ylim()[1] * 1.2)  # extend y‑axis to fit the legend

# ---- Cosmetics & show ----
axes[1].set_xticks(x)
axes[1].set_xticklabels(order)

# change y ticks
import matplotlib.ticker as mtick

for ax in axes:                           # axes = [axes[0], axes[1]]
    ax.ticklabel_format(axis='y',
                        style='sci',      # scientific notation
                        scilimits=(-3,3)) # always use 10^n for 1e‑3 … 1e3
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    ax.yaxis.get_offset_text().set_fontsize(8)  # shrink the “×10⁻³” offset

fig.suptitle("Phocaeicola vulgatus", y=0.98, fontsize=12, weight="bold")
fig.tight_layout()
plt.show()

fig.savefig(config.fig_path / 'Bv_pnps_by_freq.pdf', bbox_inches='tight')