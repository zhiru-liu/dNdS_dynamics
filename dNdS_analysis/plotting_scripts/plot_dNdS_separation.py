import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from dNdS_analysis.utils.dynamics_utils import computed_poisson_thinning, load_dNdS_data
from dNdS_analysis.utils.theory_utils import logistic_accumulation, compute_dNdS_rec_model
import dNdS_analysis.utils.figure_utils as figure_utils
import dNdS_analysis.config as config

# dnds_basepath = os.path.join(config.data_path, 'gut_microbiome_dNdS')



# row = full_dnds_df[(full_dnds_df['species_name']=='Bacteroides_vulgatus_57955') & (full_dnds_df['pair']==(3,50))]

# naive_recomb_dS = row['recomb_diff_4D'] / row['recomb_len_4D'].astype(float)
# recomb_dN = row['recomb_diff_1D'] / row['recomb_len_1D'].astype(float)
# print(recomb_dN / naive_recomb_dS)

# naive_clonal_dS = row['clonal_diff_4D'] / row['clonal_len_4D'].astype(float)
# clonal_dN = row['clonal_diff_1D'] / row['clonal_len_1D'].astype(float)
# print(clonal_dN / naive_clonal_dS)

mpl.rcParams['font.size'] = 8
fig, axes = plt.subplots(1, 3, figsize=(7, 2), dpi=300)
plt.subplots_adjust(wspace=0.08)

complete_df = load_dNdS_data()

dS1, dS2 = computed_poisson_thinning(complete_df['core_diff_4D'], complete_df['core_len_4D'])
naive_dS = complete_df['core_diff_4D'] / complete_df['core_len_4D'].astype(float)
dN = complete_df['core_diff_1D'] / complete_df['core_len_1D'].astype(float)
zero_mask = dS2 == 0
print(dN[~zero_mask].shape, np.sum(complete_df['core_diff_4D']==0))
axes[0].scatter(dS1[~zero_mask], dN[~zero_mask] / dS2[~zero_mask], s=1, alpha=0.2, color='tab:grey', rasterized=True)

naive_recomb_dS = complete_df['recomb_diff_4D'] / complete_df['recomb_len_4D'].astype(float)
zero_mask = naive_recomb_dS== 0
recomb_dN = complete_df['recomb_diff_1D'] / complete_df['recomb_len_1D'].astype(float)
axes[1].scatter(dS1[~zero_mask], recomb_dN[~zero_mask] / naive_recomb_dS[~zero_mask], s=1, alpha=0.3, color='#FF968D', rasterized=True)

naive_clonal_dS = complete_df['clonal_diff_4D'] / complete_df['clonal_len_4D'].astype(float)
zero_mask = naive_clonal_dS == 0
clonal_dN = complete_df['clonal_diff_1D'] / complete_df['clonal_len_1D'].astype(float)
axes[2].scatter(dS1[~zero_mask], clonal_dN[~zero_mask] / naive_clonal_dS[~zero_mask], s=1, alpha=0.3, color='#AECDE1', rasterized=True)

# plot recombination theory curve
theta = 3e-2
dNdS_c = 1
dNdS_r = 1e-1
ds_mid = 10**(-3.8)
k = 10
dsc_arr = np.logspace(-6, -3)
fr_arr = logistic_accumulation(dsc_arr, ds_mid, k=k)
dS_arr, dNdS_arr = compute_dNdS_rec_model(dsc_arr, fr_arr, theta, dNdS_c, dNdS_r)
# axes[0].plot(dS_arr, dNdS_arr, linestyle='-', color='k', linewidth=0.5, label='recombination theory')

for ax in axes:
    ax.set_ylim([1e-2, 1e1])
    ax.set_xlim([2e-6, 2e-2])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('$dS$ (Full genome)')
    ax.axhline(1, linewidth=0.5, linestyle='--', color='grey')

axes[0].set_ylabel('$dN/dS$')
axes[0].set_title('Full genome')
axes[1].set_title('Recombined regions')
axes[2].set_title('Clonal regions')
axes[1].set_yticklabels([])
axes[2].set_yticklabels([])
axes[0].legend()
# fig.savefig(config.fig_path / 'dNdS.pdf', bbox_inches='tight')
fig.savefig(config.fig_path / 'dNdS_no_curve.pdf', bbox_inches='tight')