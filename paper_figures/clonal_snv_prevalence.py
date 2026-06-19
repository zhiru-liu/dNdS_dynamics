"""Figure 5: prevalence of clonal SNVs (B. vulgatus detail + per-species grid).

Loads raw QP SNV catalogs per species via dnds_dynamics.snv_helpers.qp
(load_qp_snv_helper / QPSNVHelper).
Outputs (to config.fig_path): Bv_pnps_by_freq.pdf, clonal_snv_prevalence_grid.pdf,
and per-species <species>_clonal_samples_clustermap.pdf.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")

# ===== cell 0 =====
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import seaborn as sns


from dnds_dynamics.figures import dynamics as dynamics_utils
from dnds_dynamics import config

# ===== cell 1 =====
complete_df = dynamics_utils.load_dNdS_data()

# filter for pairs that are clonal
clonal_mask = complete_df['recomb_len_4D'].isna()
passed_pairs = complete_df[clonal_mask]

# ===== cell 2 =====
passed_pairs.index.get_level_values('species_name').value_counts()

# ===== cell 3 =====
from dnds_dynamics.snv_helpers.qp import load_qp_snv_helper

def dedup_clonal_pairs(pairs_df, snv_helper=None, identical_frac_threshold=0.2):
    if snv_helper is None:
        snv_helper = load_qp_snv_helper(pairs_df.index.get_level_values('species_name').unique()[0], compute_bi_snvs=False, annotate=True)
        snv_helper.load_identical_block()
    included_pairs = []
    for pair, row in pairs_df.iterrows():
        # check if pair is close to any of the included pairs
        close_to_included = False
        pair = (str(pair[0]), str(pair[1]))
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

# ===== cell 4 =====
# find clonal pairs per species
species_name = 'Bacteroides_vulgatus_57955'
species_clonal_pairs = passed_pairs.loc[species_name]

# then keep one pair per clonal cluster
snv_helper = load_qp_snv_helper(species_name, compute_bi_snvs=False, annotate=True)
snv_helper.load_identical_block()

# then filter to only one pair per clonal cluster 
dedup_pairs = dedup_clonal_pairs(species_clonal_pairs, snv_helper)

# ===== cell 5 =====
clades = pd.read_csv(config.data_path / 'Bv_clades.txt', sep='\t',index_col=0, header=None, names=['sample','clade'])
major_clade_samples = clades[clades['clade'] == 'major'].index

dedup_pairs = [pair for pair in dedup_pairs if pair[0] in major_clade_samples]

# ===== cell 7 =====
# start with the clonal samples
samples1 = set(species_clonal_pairs.index.get_level_values('sample 1'))
samples2 = set(species_clonal_pairs.index.get_level_values('sample 2'))
clonal_samples = list(samples1.union(samples2))
print(len)

# ===== cell 8 =====
samples = [x for tup in dedup_pairs for x in tup]
unique_samples = list(set(samples))

sns.clustermap(snv_helper.identical_block_frac.loc[unique_samples, unique_samples],)

# ===== cell 9 =====
included_samples = [sample for sample in unique_samples if sample in major_clade_samples]
for sample in major_clade_samples:
    if sample in included_samples:
        continue
    # then check the distance to the included samples
    # access the identical block fraction matrix directly
    similarity = snv_helper.identical_block_frac.loc[included_samples, sample]
    # if any similarity is greater than 0.5, skip the sample
    if similarity.max() < 0.5:
        included_samples.append(sample)
        continue

print(len(dedup_pairs))
print(len(included_samples))

# ===== cell 10 =====
sns.clustermap(snv_helper.identical_block_frac.loc[included_samples, included_samples],)

# ===== cell 12 =====
def repolarize_counts(alts, covs):
    m = alts > (covs / 2)
    alts = alts.copy()
    alts[m] = covs[m] - alts[m]
    return alts

# ===== cell 13 =====
def compute_snv_stats(snv_helper, clonal_pairs, included_samples):
    # Identify clonal SNVs
    all_clonal_diffs = pd.DataFrame(index=snv_helper.snvs.index)
    # dedup pairs are unique clonal pairs with no other clonal strains
    for pair in clonal_pairs:
        pair_snvs = snv_helper.compute_pairwise_snvs(pair[0], pair[1])
        all_clonal_diffs[pair] = pair_snvs
    clonal_snvs = all_clonal_diffs[all_clonal_diffs.sum(axis=1)>0]

    haps_all = snv_helper.snvs.loc[:, included_samples]
    num_alt_all = (haps_all==1).sum(axis=1)
    num_covered_all = (haps_all!=255).sum(axis=1)
    snv_filter = (num_alt_all > 0) & (num_alt_all < num_covered_all)
    haps_all = haps_all.loc[snv_filter, :]
    num_alt_all = num_alt_all[snv_filter]
    num_covered_all = num_covered_all[snv_filter]

    # next focusing on subset of snvs that are detected between clonal pairs
    haps = haps_all.loc[clonal_snvs.index, :]
    num_alt = (haps==1).sum(axis=1)
    num_covered = (haps!=255).sum(axis=1)

    # Next refine to only core 1D and 4D sites
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

    L1d = snv_helper.core_1D.sum()
    L4d = snv_helper.core_4D.sum()

    return (num_alt_all_1D, num_alt_all_4D, num_covered_all_1D, num_covered_all_4D,
            num_alt_1D, num_alt_4D, num_covered_1D, num_covered_4D, L1d, L4d)

# ===== cell 14 =====
# freq_edges  = np.array([0.00, 0.05, 0.10, 0.25, 0.50])  # left‑closed, right‑open
# freq_labels = ["<5%", "5-10%", "10-25%", "25-50%"]    # len = len(freq_edges)‑1
freq_edges  = np.array([0.00, 0.10, 0.50])  # left‑closed, right‑open
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

# ===== cell 15 =====
num_alt_all_1D, num_alt_all_4D, num_covered_all_1D, num_covered_all_4D, num_alt_1D, num_alt_4D, num_covered_1D, num_covered_4D, L1d, L4d = compute_snv_stats(snv_helper, dedup_pairs, included_samples)
# 1D all
cats_1d_all    = label_sites(num_alt_all_1D,    num_covered_all_1D)
# 4D all
cats_4d_all    = label_sites(num_alt_all_4D,    num_covered_all_4D)
# 1D clonal
cats_1d_clonal = label_sites(num_alt_1D, num_covered_1D)
# 4D clonal
cats_4d_clonal = label_sites(num_alt_4D, num_covered_4D)

# ===== cell 16 =====
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.stats import poisson
from statsmodels.stats.proportion import proportion_confint   # optional if you want Wilson

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
    """Return a (2, n) array of asymmetric errors for plt.errorbar."""
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
    sharex=True, gridspec_kw={'hspace': 0.15}
)

# ---- Panel A: Clonal ----
ax = axes[0]
ax.bar(x - width/2, prop_1d_clonal, width,
       label="Nonsynonymous (1D)", color="tab:blue", alpha=0.9)
ax.bar(x + width/2, prop_4d_clonal, width,
       label="Synonymous (4D)", color="tab:blue", alpha=0.4)

# Add error bars
ax.errorbar(x - width/2, prop_1d_clonal, yerr=err_1d_clonal,
            fmt='none', ecolor='k', capsize=3, lw=1)
ax.errorbar(x + width/2, prop_4d_clonal, yerr=err_4d_clonal,
            fmt='none', ecolor='k', capsize=3, lw=1)
# ax.errorbar(x - width/2, prop_1d_clonal, yerr=15/L1d,
#             fmt='none', ecolor='k', capsize=3, lw=1)
# ax.errorbar(x + width/2, prop_4d_clonal, yerr=7/L4d,
#             fmt='none', ecolor='k', capsize=3, lw=1)

ax.set_title("Clonal SNVs")
ax.legend(frameon=False)

# ---- Panel B: All ----
ax = axes[1]
ax.bar(x - width/2, prop_1d_all, width,
       label="Nonsynonymous (1D)", color="tab:orange", alpha=0.9)
ax.bar(x + width/2, prop_4d_all, width,
       label="Synonymous (4D)", color="tab:orange", alpha=0.4)

ax.errorbar(x - width/2, prop_1d_all, yerr=err_1d_all,
            fmt='none', ecolor='k', capsize=3, lw=1)
ax.errorbar(x + width/2, prop_4d_all, yerr=err_4d_all,
            fmt='none', ecolor='k', capsize=3, lw=1)

ax.set_xlabel("Allele prevalence across sampled hosts")
ax.set_title("All SNVs")
ax.legend(loc='upper right', frameon=False)

# ---- Cosmetics & show ----
axes[1].set_xticks(x)
axes[1].set_xticklabels(order)
axes[1].set_ylim(0, 0.028)

# Custom ytick labels
axes[0].set_yticks(np.array([0, 0.5, 1, 1.5, 2])*1e-4)
axes[0].set_yticklabels(['0', '0.5', '1', '1.5', '2'])
axes[0].set_ylabel("Fraction of sites ($\\times 10^{-4}$)")
axes[1].set_yticks(np.array([0, 0.5, 1, 1.5, 2, 2.5])*1e-2)
axes[1].set_yticklabels(['0', '0.5', '1', '1.5', '2', '2.5'])
axes[1].set_ylabel("Fraction of sites ($\\times 10^{-2}$)")

# Add panel labels
labels = ["A", "B"]
for ax, label in zip(axes, labels):
    ax.text(
        -0.1, 1.05, label, transform=ax.transAxes,
        fontsize=14, fontweight="bold", va="top", ha="right"
    )

fig.suptitle("Phocaeicola vulgatus", y=0.98, fontsize=15, fontstyle='italic')
fig.tight_layout()
fig.savefig(config.fig_path / 'Bv_pnps_by_freq.pdf', bbox_inches='tight')

# ===== cell 18 =====
species_res = {}
for species_name, grouped in passed_pairs.groupby("species_name"):
    if len(grouped) < 5:
        continue
    print(f"Processing {species_name} with {len(grouped)} pairs")
    species_clonal_pairs = passed_pairs.loc[species_name]

    # then keep one pair per clonal cluster
    snv_helper = load_qp_snv_helper(species_name, compute_bi_snvs=False, annotate=True)
    snv_helper.load_identical_block()

    # then filter to only one pair per clonal cluster 
    clonal_pairs = dedup_clonal_pairs(species_clonal_pairs, snv_helper)

    # find clonal SNVs
    samples = [x for tup in clonal_pairs for x in tup]
    unique_samples = list(set(samples))
    species_samples = snv_helper.samples
    
    included_samples = [sample for sample in unique_samples]
    for sample in species_samples:
        if sample in included_samples:
            continue
        # then check the distance to the included samples
        # access the identical block fraction matrix directly
        similarity = snv_helper.identical_block_frac.loc[included_samples, sample]
        # if any similarity is greater than 0.5, skip the sample
        if similarity.max() < 0.5:
            included_samples.append(sample)
            continue

    print("Num clonal pairs:", len(clonal_pairs))
    if len(clonal_pairs) < 5:
        print(f"Skipping {species_name} due to insufficient samples")
        continue
    print("Num included samples:", len(included_samples))
    sns.clustermap(snv_helper.identical_block_frac.loc[unique_samples, unique_samples])
    plt.savefig(config.fig_path / f"{species_name}_clonal_samples_clustermap.pdf", dpi=600, bbox_inches='tight')
    plt.close()

    # compute SNV stats
    species_res[species_name] = compute_snv_stats(snv_helper, clonal_pairs, included_samples)

# ===== cell 19 =====
import numpy as np
import matplotlib.pyplot as plt
sns.set_style("white")


def _prep_one_species(stats):
    """Generalize the Bv single-species pipeline (label_sites -> counts_and_prop
    -> poisson_prop_err) to any species' compute_snv_stats() tuple, returning the
    per-prevalence-bin proportions and Poisson error bars the grid expects.
    """
    (num_alt_all_1D, num_alt_all_4D, num_covered_all_1D, num_covered_all_4D,
     num_alt_1D, num_alt_4D, num_covered_1D, num_covered_4D, L1d, L4d) = stats

    cats_1d_clonal = label_sites(num_alt_1D, num_covered_1D)
    cats_4d_clonal = label_sites(num_alt_4D, num_covered_4D)
    cats_1d_all    = label_sites(num_alt_all_1D, num_covered_all_1D)
    cats_4d_all    = label_sites(num_alt_all_4D, num_covered_all_4D)

    cnt_1d_clonal, prop_1d_clonal = counts_and_prop(cats_1d_clonal, L1d)
    cnt_4d_clonal, prop_4d_clonal = counts_and_prop(cats_4d_clonal, L4d)
    cnt_1d_all,    prop_1d_all    = counts_and_prop(cats_1d_all,    L1d)
    cnt_4d_all,    prop_4d_all    = counts_and_prop(cats_4d_all,    L4d)

    return {
        "prop_1d_clonal": prop_1d_clonal, "prop_4d_clonal": prop_4d_clonal,
        "prop_1d_all":    prop_1d_all,    "prop_4d_all":    prop_4d_all,
        "err_1d_clonal":  poisson_prop_err(cnt_1d_clonal, L1d),
        "err_4d_clonal":  poisson_prop_err(cnt_4d_clonal, L4d),
        "err_1d_all":     poisson_prop_err(cnt_1d_all, L1d),
        "err_4d_all":     poisson_prop_err(cnt_4d_all, L4d),
    }


def plot_snv_prevalence_grid_6x4(species_res, species_order=None, figsize=(16, 18)):
    """
    Plot a fixed 6x4 grid: two species per row, each species has two panels
    (Clonal, All). Any unused panels are hidden.
    """
    if species_order is None:
        species_order = sorted(species_res.keys())

    # Precompute everything
    prepped = {sp: _prep_one_species(species_res[sp]) for sp in species_order}

    # Common y-limit across all panels
    all_props = []
    for d in prepped.values():
        all_props.extend(d["prop_1d_clonal"])
        all_props.extend(d["prop_4d_clonal"])
        all_props.extend(d["prop_1d_all"])
        all_props.extend(d["prop_4d_all"])
    ymax = max(all_props) if all_props else 0.0
    ymax = 0.05 if ymax == 0 else min(1.0, ymax * 1.25)

    nrows, ncols = 6, 4  # fixed grid
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=True)
    x = np.arange(len(order))
    width = 0.40

    # Helper to draw one species into a column pair (c0 and c0+1) on row r
    def draw_species_at(sp, r, c0, show_xticks=False, show_legend=False):
        d = prepped[sp]
        # Left: Clonal
        ax = axes[r, c0]
        ax.bar(x - width/2, d["prop_1d_clonal"], width, label="1D", color="tab:blue",  alpha=0.90)
        ax.bar(x + width/2, d["prop_4d_clonal"], width, label="4D", color="tab:blue",  alpha=0.40)
        ax.errorbar(x - width/2, d["prop_1d_clonal"], yerr=d["err_1d_clonal"], fmt='none', ecolor='k', capsize=3, lw=1)
        ax.errorbar(x + width/2, d["prop_4d_clonal"], yerr=d["err_4d_clonal"], fmt='none', ecolor='k', capsize=3, lw=1)
        # if not np.isnan(d["private_1d_frac"]):
        #     ax.text(0.98, 0.92, f"private (1D): {d['private_1d_frac']:.1%}",
        #             transform=ax.transAxes, ha="right", va="top")
        ax.set_title(f"{sp}")

        # Right: All
        ax = axes[r, c0+1]
        ax.bar(x - width/2, d["prop_1d_all"], width, label="1D", color="tab:orange", alpha=0.90)
        ax.bar(x + width/2, d["prop_4d_all"], width, label="4D", color="tab:orange", alpha=0.40)
        ax.errorbar(x - width/2, d["prop_1d_all"], yerr=d["err_1d_all"], fmt='none', ecolor='k', capsize=3, lw=1)
        ax.errorbar(x + width/2, d["prop_4d_all"], yerr=d["err_4d_all"], fmt='none', ecolor='k', capsize=3, lw=1)
        # ax.set_title(f"{sp}\nAll SNVs")

        # X ticks
        if show_xticks:
            axes[r, c0].set_xticks(x);     axes[r, c0].set_xticklabels(order, rotation=0)
            axes[r, c0+1].set_xticks(x);   axes[r, c0+1].set_xticklabels(order, rotation=0)
            # axes[r, c0+1].set_xlabel("Allele prevalence across sampled hosts")
        else:
            axes[r, c0].set_xticks(x);     axes[r, c0].set_xticklabels([])
            axes[r, c0+1].set_xticks(x);   axes[r, c0+1].set_xticklabels([])

        if show_legend:
            axes[r, c0].set_title(f"{sp}\nClonal SNVs")
            axes[r, c0+1].set_title(f"All SNVs")
            for ax in (axes[r, c0], axes[r, c0+1]):
                ax.legend(frameon=False, loc="best")

    # Place species: two per row
    max_species_slots = nrows * (ncols // 2)  # 12
    used_slots = min(len(species_order), max_species_slots)
    for idx in range(used_slots):
        sp = species_order[idx]
        row = idx // 2
        colpair_start = (idx % 2) * 2  # (0,1) or (2,3)
        show_xticks = (row == nrows - 1)
        # show_xticks = (idx == 0)
        show_legend = (idx == 0) 
        draw_species_at(sp, row, colpair_start, show_xticks=show_xticks, show_legend=show_legend)

    # Hide any unused axes (including the last two panels when you have 11 species)
    total_panels = nrows * ncols  # 24
    used_panels = used_slots * 2  # two panels per species
    to_hide = total_panels - used_panels
    if to_hide > 0:
        # flatten in row-major order, hide from the end backwards
        flat_axes = axes.ravel()
        for ax in flat_axes[-to_hide:]:
            ax.axis('off')

    plt.tight_layout()
    return fig, axes

# ---- Example:
species_order = sorted(species_res.keys())  # or your custom order
fig, axes = plot_snv_prevalence_grid_6x4(species_res, species_order=species_order, figsize=(16, 18))
# axes[-1, 0].set_xlabel("Allele prevalence across sampled hosts")
fig.text(0.5, -0.02, "Clonal SNV Prevalence Across Species", ha='center', fontsize=25)
fig.text(-0.02, 0.5, "Fraction of Sites", va='center', rotation='vertical', fontsize=25)
plt.show()

fig.savefig(config.fig_path / "clonal_snv_prevalence_grid.pdf", dpi=600, bbox_inches='tight')


