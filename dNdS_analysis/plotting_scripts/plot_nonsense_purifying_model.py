import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from dNdS_analysis.utils import figure_utils, dynamics_utils
from dNdS_analysis.utils.theory_utils import dNdS_purify_curve
import dNdS_analysis.config as config

# load cached results after running plot_nonsense_missense.py
clonal_res = pd.read_csv(config.fig_dat_path / 'clonal_region_ns_ms.csv')
typical_res = pd.read_csv(config.fig_dat_path / 'typical_pair_ns_ms.csv')

fig, ax = plt.subplots(figsize=(6, 4))

# first plot the purifying curve with different s/ns and s/ms ratios
# assuming the same selection coefficient for missense and nonsense mutations
xs = np.logspace(-6, -1)
sbymu = 1e4
fs = 0.1
dNdS_val_mis = dNdS_purify_curve(xs, 1-fs, sbymu)


# a lower ratio of secretly synonymous mutations in nonsense mutations
fs = 0.1 * 0.07
dNdS_val_non = dNdS_purify_curve(xs, 1-fs, sbymu)
ax.loglog(xs, dNdS_val_non / dNdS_val_mis, ':', color='k', label='$s_{ns}=s_{ms}$')


# then plot the case where nonsense mutations are more deleterious 
sbymu = 4e4
fs = 0.1 * 0.07
dNdS_val_non = dNdS_purify_curve(xs, 1-fs, sbymu)

ax.loglog(xs, dNdS_val_non / dNdS_val_mis, '--', color='k', label='$s_{ns}=4s_{ms}$')


# now plot the empirical data for reference
yerr = np.sqrt(clonal_res['ns SE']**2 + clonal_res['ms SE']**2)
xp, yp, xerrp, yerrp = figure_utils.translate_errorbars(clonal_res['syn rates'], clonal_res['ns rates'] - clonal_res['ms rates'], clonal_res['syn SE'], yerr)
ax.errorbar(xp, yp, xerr=xerrp, yerr=yerrp, markersize=5,
            color=figure_utils.clonal_color, fmt='-o', label='Clonal regions')

# plot each species average separately
yerr = np.sqrt(typical_res['ns SE']**2 + typical_res['ms SE']**2)
xp, yp, xerrp, yerrp = figure_utils.translate_errorbars(typical_res['syn rates'], typical_res['ns rates'] - typical_res['ms rates'], typical_res['syn SE'], yerr)
ax.errorbar(xp, yp, xerr=xerrp, yerr=yerrp, markersize=5,
            color='grey', fmt='x', label='Species average')

ax.legend(loc='lower left')
ax.set_xlabel('$dS$')
ax.set_ylabel(r'Nonsense / Missense')
ax.set_xlim(1e-6, 1e-1)
plt.savefig(config.fig_path / 'dN_nsms_purify_curve.pdf')