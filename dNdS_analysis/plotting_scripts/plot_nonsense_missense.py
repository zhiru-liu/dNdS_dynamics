import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from dNdS_analysis.utils import figure_utils, dynamics_utils
import dNdS_analysis.config as config


# first load close pair dN/dS data
dat_path = config.data_path / 'gut_microbiome_close_pair_dNdS'
species_list = [f.name.split('.')[0] for f in dat_path.glob('*.csv')]

res = []
for species in species_list:
    close_pair_path = config.data_path / 'gut_microbiome_close_pair_dNdS' / f'{species}.csv'
    close_pair = pd.read_csv(close_pair_path, index_col=0).reset_index()
    res.append(close_pair)

close_pair_dNdS = pd.concat(res)
close_pair_dNdS.drop(columns=['index', 'species_name', 'sample 1', 'sample 2'], inplace=True)
close_pair_dNdS = close_pair_dNdS.astype(float)

# then load clonal pair dN/dS data
dat_path = config.data_path / 'gut_microbiome_clonal_pair_dNdS'
species_list = [f.name.split('.')[0] for f in dat_path.glob('*.csv')]
res = []
for species in species_list:
    if species in config.blacklist_species:
        # some species are blacklisted since recombination detection is not reliable
        continue
    clonal_pair_path = config.data_path / 'gut_microbiome_clonal_pair_dNdS' / f'{species}.csv'
    clonal_pair = pd.read_csv(clonal_pair_path, index_col=0).reset_index()
    res.append(clonal_pair)
clonal_pair_dNdS = pd.concat(res)
clonal_pair_dNdS.drop(columns=['species_name', 'sample 1', 'sample 2', 'core_diff', 'core_len'], inplace=True)
clonal_pair_dNdS.columns = [col.replace('core', 'clonal') for col in clonal_pair_dNdS.columns]
clonal_pair_dNdS = clonal_pair_dNdS.astype(float)

# compute rates for clonal regions
clonal_full = pd.concat([close_pair_dNdS[clonal_pair_dNdS.columns], clonal_pair_dNdS])
clonal_res = dynamics_utils.fit_clonal_region_results(clonal_full)

# compute rates for recombination regions
recomb_res = dynamics_utils.fit_recomb_region_results(close_pair_dNdS)

# compute rates for typical pairs
typical_res = dynamics_utils.load_and_fit_typical_pair_results()


# plotting
fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))
plt.subplots_adjust(wspace=0.33)

ax = axes[0]
xp, yp, xerrp, yerrp = figure_utils.translate_errorbars(clonal_res['syn rates'], clonal_res['ms rates'], clonal_res['syn SE'], clonal_res['ms SE'])
ax.errorbar(xp, yp, xerr=xerrp, yerr=yerrp, markersize=5, label='clonal missense',
            color=figure_utils.clonal_ms_color, fmt='-o')

xp, yp, xerrp, yerrp = figure_utils.translate_errorbars(clonal_res['syn rates'], clonal_res['ns rates'], clonal_res['syn SE'], clonal_res['ns SE'])
ax.errorbar(xp, yp, xerr=xerrp, yerr=yerrp, markersize=6, label='clonal nonsense', 
            color=figure_utils.clonal_ns_color, fmt='-^')

ax.plot([1e-6, 1e-3], [1e-6, 1e-3], color='black', linestyle='--', label='dN=dS')
ax.legend()

ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlim(0.2e-5, 1e-3)
ax.set_ylim(0.2e-5, 1e-3)
ax.set_xlabel('$dS$')
ax.set_ylabel('$dN$')

ax = axes[1]

yerr = np.sqrt(clonal_res['ns SE']**2 + clonal_res['ms SE']**2)
xp, yp, xerrp, yerrp = figure_utils.translate_errorbars(clonal_res['syn rates'], clonal_res['ns rates'] - clonal_res['ms rates'], clonal_res['syn SE'], yerr)
ax.errorbar(xp, yp, xerr=xerrp, yerr=yerrp, markersize=5,
            color=figure_utils.clonal_color, fmt='-o', label='Clonal regions')

clonal_res.to_csv(config.fig_dat_path / 'clonal_region_ns_ms.csv')

yerr = np.sqrt(recomb_res['recomb ns SE']**2 + recomb_res['recomb ms SE']**2)
xp, yp, xerrp, yerrp = figure_utils.translate_errorbars(recomb_res['syn rates'], recomb_res['recomb ns rates'] - recomb_res['recomb ms rates'], recomb_res['syn SE'], yerr)
ax.errorbar(xp, yp, xerr=xerrp, yerr=yerrp, markersize=5,
            color=figure_utils.recomb_color, fmt='-o', label='Recombined regions')

average_res = typical_res.mean(axis=0)
ax.axhline(np.power(10, average_res['ns rates'] - average_res['ms rates']), color='black', linestyle='--',
           label='Average across\nunrelated pairs')
ax.legend(loc='upper right')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(0.2e-5, 1e-3)
ax.set_ylim(0.5e-2, 10)
ax.set_xlabel('$dS$')
# ax.set_ylabel(r'$\frac{\text{Nonsense rate}}{\text{Missense rate}}$')
ax.set_ylabel(r'Nonsense / Missense')

clonal_res.to_csv(config.fig_dat_path / 'clonal_region_ns_ms.csv')
typical_res.to_csv(config.fig_dat_path / 'typical_pair_ns_ms.csv')
recomb_res.to_csv(config.fig_dat_path / 'recomb_region_ns_ms.csv')

fig.savefig(config.fig_path / 'nonsense_vs_missense.pdf', bbox_inches='tight')