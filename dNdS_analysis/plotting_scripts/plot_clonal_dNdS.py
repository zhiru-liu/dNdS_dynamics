import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pandas as pd
import scipy.stats
import dNdS_analysis.config as config
from dNdS_analysis.utils.dynamics_utils import computed_poisson_thinning
import dNdS_analysis.utils.figure_utils as figure_utils
from dNdS_analysis.utils.theory_utils import dNdS_purify_curve

def plot_scatter(dat, ax):
    vals = dat.values
    kernel = scipy.stats.gaussian_kde(vals.T)
    colors = kernel(vals.T)
    cmin, cmax = colors.min(), colors.max()

    ax = sns.scatterplot(data=dat, x="dS", y="dNdS", hue=colors, hue_norm=(-3000, cmax), ax=ax, palette='Blues', rasterized=True, s=3, legend=False)

    # now plot the purifying selection curve
    fd = 0.9
    xs = np.logspace(-6, -1)
    purify_curve_data = pd.DataFrame()
    purify_curve_data['dS'] = xs
    purify_curve_data.set_index('dS', inplace=True)
    for sbymu in [1e5, 1e4, 1e3]:
        dNdS_val = dNdS_purify_curve(xs, fd, sbymu)
        name = '$s/\mu=10^{0:.0f}$'.format(np.log10(sbymu))
        purify_curve_data[name] = dNdS_val

    sns.lineplot(purify_curve_data, palette='deep', ax=ax)

    ax.set_xlim([1e-6, 1e-1])
    ax.set_ylim([0.3e-1, 2e1])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel('$dN/dS$ (Clonal region)')
    ax.set_xlabel('$dS$ (Clonal region)')


def plot_corr_barh(plot_df, ax):
    significant_mask = plot_df['Pearson p'] < 0.001

    ax.barh(plot_df[significant_mask].index, plot_df[significant_mask]['Pearson r'], linewidth=0.7, facecolor='tab:blue', edgecolor='k', fill=True, label=None)
    ax.barh(plot_df[~significant_mask].index, plot_df[~significant_mask]['Pearson r'],linewidth=0.7, color='black', fill=False,)
    ax.set_yticks(np.arange(plot_df.shape[0]))
    ax.set_yticklabels(plot_df['Species'],fontsize=6)
    # ax.barh(xs[~significant_mask], corr_df['pearsonr'][~significant_mask], label=None)
    # ax.set_yticks(np.arange(corr_df.shape[0]+1))
    # ax.set_yticklabels(corr_df['species'].to_list()+['Combined'],fontsize=6)
    # plt.xlim(-1, 1)
    # corr_df.head()
    ax.set_xlabel('Pearson r')
    axes[1].set_ylim(-1, len(plot_df))
    axes[1].yaxis.tick_right()
    axes[1].axvline(plot_df.iloc[-1]['Pearson r'], linewidth=1, color='k', linestyle='--')

if __name__ == '__main__':
    # ax = sns.scatterplot(data=dat, x="dS", y="dNdS", c=colors, cmap="Reds",rasterized=True, s=2, vmin=cmin/10, vmax=cmax)

    # TODO: 24/05/10: update to close pair dNdS once everything is finished
    dnds_basepath = os.path.join(config.data_path, 'gut_microbiome_dNdS')
    dnds_dfs = []

    transfer_df = pd.read_csv(os.path.join(config.data_path, 'gut_microbiome_transfers.csv'), index_col=0)
    pair_to_cf = transfer_df.groupby(['Species name', 'Sample 1', 'Sample 2'])['Clonal fraction'].mean()
    pair_to_div = transfer_df.groupby(['Species name', 'Sample 1', 'Sample 2'])['Clonal divergence'].mean()

    for file in os.listdir(dnds_basepath):
        if file.startswith('.'):
            continue
        species_name = file.split('.')[0]
        dnds_df = pd.read_csv(os.path.join(dnds_basepath, species_name + '.csv'))
        dnds_df.set_index(['species_name', 'sample 1', 'sample 2'], inplace=True)
        dnds_df['Clonal fraction'] = pair_to_cf
        dnds_df['Clonal divergence'] = pair_to_div
        # note: some of the values will be NaN because they have no recombination events
        dnds_dfs.append(dnds_df)

    full_dnds_df = pd.concat(dnds_dfs)
    full_dnds_df.reset_index(inplace=True)

    clonal_dS1, clonal_dS2 = computed_poisson_thinning(full_dnds_df['clonal_diff_4D'], full_dnds_df['clonal_len_4D'])
    naive_clonal_dS = full_dnds_df['clonal_diff_4D'] / full_dnds_df['clonal_len_4D'].astype(float)
    clonal_dN = full_dnds_df['clonal_diff_1D'] / full_dnds_df['clonal_len_1D'].astype(float)

    val1 = clonal_dS1
    val2 = clonal_dN / clonal_dS2
    vals = np.vstack([val1, val2]).T
    # filter rows with nan
    vals = vals[~(np.isinf(vals).any(axis=1))]
    close_res = pd.DataFrame(vals, columns=['dS', 'dNdS'])

    # next add the typical pairs
    typical_pair_dnds_basepath = os.path.join(config.data_path, 'gut_microbiome_typical_pair_dNdS')
    dnds_dfs = []
    for file in os.listdir(typical_pair_dnds_basepath):
        if file.startswith('.'):
            continue
        species_name = file.split('.')[0]
        dnds_df = pd.read_csv(os.path.join(typical_pair_dnds_basepath, species_name + '.csv'))
        dnds_df.set_index(['species_name', 'sample 1', 'sample 2'], inplace=True)
        # note: some of the values will be NaN because they have no recombination events
        dnds_dfs.append(dnds_df)
    typical_dnds_dfs = pd.concat(dnds_dfs)
    typical_dnds_dfs.reset_index(inplace=True)
    typical_dnds_dfs.drop(['sample 1', 'sample 2'], axis=1, inplace=True)
    # take the median of each species
    typical_dnds_dfs = typical_dnds_dfs.groupby('species_name')[['core_diff_4D', 'core_len_4D', 'core_diff_1D', 'core_len_1D']].median()

    typical_dS1, typical_dS2 = computed_poisson_thinning(typical_dnds_dfs['core_diff_4D'], typical_dnds_dfs['core_len_4D'])
    typical_dN = typical_dnds_dfs['core_diff_1D'] / typical_dnds_dfs['core_len_1D'].astype(float)
    vals = np.vstack([typical_dS1, typical_dN / typical_dS2]).T
    # filter rows with nan
    vals = vals[~(np.isinf(vals).any(axis=1))]
    typical_res = pd.DataFrame(vals, columns=['dS', 'dNdS'])


    # zhiru: save the scatter data in case of publication need
    # dat.to_csv(os.path.join(config.fig_dat_dir, 'clonal_dNdS_scatter.csv'), index=False)

    # fig, ax = plt.subplots(figsize=(4, 3))
    fig = plt.figure(figsize=(5, 3))
    grid = plt.GridSpec(1, 2, wspace=0.25, hspace=0.3, width_ratios=[5, 1], figure=fig)
    axes = [None, None]
    axes[0] = plt.subplot(grid[0, 0])
    axes[1] = plt.subplot(grid[0, 1])

    plot_scatter(close_res, axes[0])
    # adding the typical pair results
    # use x marker
    sns.scatterplot(data=typical_res, x="dS", y="dNdS", ax=axes[0], rasterized=True, s=3, legend=False, color='tab:grey', marker='x')


    # if also plotting correlation coefficients
    # need to be computed in plot_clonal_dNdS_all_species first
    corr_df = pd.read_csv(os.path.join(config.table_path, 'clonal_dNdS_correlation.csv'))
    corr_df['species'] = corr_df['species'].apply(figure_utils.get_pretty_species_name)
    corr_df.sort_values('species', inplace=True, ascending=False)

    # also compute the correlation coefficient for the combined data
    pearson_res = scipy.stats.pearsonr(close_res['dS'], close_res['dNdS'])
    spearman_res = scipy.stats.spearmanr(close_res['dS'], close_res['dNdS'])
    plot_df = pd.DataFrame()
    plot_df['Species'] = corr_df['species'].to_list() + ['All species']
    plot_df['Pearson r'] = corr_df['pearsonr'].to_list() + [pearson_res[0]]
    plot_df['Pearson p'] = corr_df['pearsonp'].to_list() + [pearson_res[1]]
    plot_corr_barh(plot_df, axes[1])

    plt.savefig(config.fig_path / 'clonal_dNdS_test.pdf', dpi=300, bbox_inches='tight')