import matplotlib.pyplot as plt
import os
import pandas as pd
import scipy.stats
import dNdS_analysis.config as config
from dNdS_analysis.utils.dynamics_utils import computed_poisson_thinning
from dNdS_analysis.utils.figure_utils import get_pretty_species_name

# dnds_basepath = os.path.join(config.data_path, 'gut_microbiome_dNdS')
dnds_basepath = os.path.join(config.data_path, 'gut_microbiome_close_pair_dNdS')
dnds_dfs = []

transfer_df = pd.read_csv(os.path.join(config.data_path, 'gut_microbiome_transfers.csv'), index_col=0)
pair_to_cf = transfer_df.groupby(['Species name', 'Sample 1', 'Sample 2'])['Clonal fraction'].mean()
pair_to_div = transfer_df.groupby(['Species name', 'Sample 1', 'Sample 2'])['Clonal divergence'].mean()

for file in os.listdir(dnds_basepath):
    if file.startswith('.'):
        continue
    if file.endswith('mut_opportunity.csv'):
        continue
    species_name = file.split('.')[0]
    dnds_df = pd.read_csv(os.path.join(dnds_basepath, species_name + '.csv'))
    dnds_df.set_index(['species_name', 'sample 1', 'sample 2'], inplace=True)
    dnds_df['Clonal fraction'] = pair_to_cf
    dnds_df['Clonal divergence'] = pair_to_div
    # note: some of the values will be NaN because they have no recombination events
    dnds_dfs.append(dnds_df)

big_df = pd.concat(dnds_dfs)
big_df.reset_index(inplace=True)
species_list = big_df['species_name'].unique()
species_list = sorted(species_list)

fig, axes = plt.subplots(6, 5, figsize=(20, 16))
plt.subplots_adjust(wspace=0.4, hspace=0.6)
count = 0
spearmanr_list = []
spearmanp_list = []
pearsonr_list = []
pearsonp_list = []

for i in range(6):
    for j in range(5):
        if count >= len(species_list):
            break
        species_name = species_list[count]
        ax = axes[i, j]

        sub_df = big_df[big_df['species_name'] == species_name]
        clonal_dS1, clonal_dS2 = computed_poisson_thinning(sub_df['clonal_diff_4D'], sub_df['clonal_len_4D'])
        naive_clonal_dS = sub_df['clonal_diff_4D'] / sub_df['clonal_len_4D'].astype(float)
        clonal_dN = sub_df['clonal_diff_1D'] / sub_df['clonal_len_1D'].astype(float)

        ax.scatter(clonal_dS1,
                   clonal_dN / clonal_dS2, s=2, alpha=0.7, color='#AECDE1', rasterized=True)
        
        # TODO: 240510: add the typical pair results

        nan_mask = clonal_dS2!=0
        res = scipy.stats.spearmanr(clonal_dS1[nan_mask], (clonal_dN / clonal_dS2)[nan_mask])
        spearmanr_list.append(res.correlation)
        spearmanp_list.append(res.pvalue)
        # also compute Pearson correlation
        res = scipy.stats.pearsonr(clonal_dS1[nan_mask], (clonal_dN / clonal_dS2)[nan_mask])
        pearsonr_list.append(res.correlation)
        pearsonp_list.append(res.pvalue)

        ax.set_xlim([2e-6, 1e-3])
        ax.axhline(1, linewidth=1, linestyle='--', color='grey')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim([0.8e-1, 3e1])
        ax.set_title(get_pretty_species_name(species_name))

        if i==5:
            ax.set_xlabel('$dS$ (Clonal region)')
        if j==0:
            ax.set_ylabel('$dN/dS$ (Clonal region)')

        count += 1

fig.savefig(config.fig_path / 'clonal_dNdS_all_species_test.pdf', dpi=300, bbox_inches='tight')

df = pd.DataFrame({'species': species_list, 'spearmanr': spearmanr_list, 'spearmanp': spearmanp_list, 'pearsonr': pearsonr_list, 'pearsonp': pearsonp_list})
# df.to_csv(os.path.join(config.table_path, 'clonal_dNdS_correlation.csv'), index=False)