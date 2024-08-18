import matplotlib as mpl
import numpy as np
import seaborn as sns

mpl.rcParams['font.size'] = 9
mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['legend.fontsize'] = 7
mpl.rcParams['legend.frameon'] = False

recomb_color = '#FF968D'
clonal_color = '#AECDE1'
pal = sns.color_palette("Paired", 6).as_hex()
clonal_ns_color = pal[0]
clonal_ms_color = pal[1]
recomb_ns_color = pal[4]
recomb_ms_color = pal[5]

def get_pretty_species_name(species_name, include_number=False, manual=False):
    # convert gut microbiome species name to pretty name for plotting
    items = species_name.split("_")
    
    pretty_name = "%s %s" % (items[0], items[1])
    
    if include_number:
        pretty_name += (" (%s)" % (items[2]))

    # manually matching GarudGood et al convention
    if manual:
        if species_name=='Faecalibacterium_prausnitzii_57453':
            return pretty_name + ' 3'
        elif species_name == 'Faecalibacterium_prausnitzii_62201':
            return pretty_name + ' 2'
    return pretty_name

def translate_errorbars(x, y, xerr, yerr, logbase=10, num_se=2):
    x_plot = np.power(logbase, x)
    y_plot = np.power(logbase, y)
    xerr_max = np.power(logbase, x + xerr * num_se) - x_plot
    xerr_min = x_plot - np.power(logbase, x - xerr * num_se)
    xerr_plot = np.stack([xerr_min, xerr_max])
    yerr_max = np.power(logbase, y + yerr * num_se) - y_plot
    yerr_min = y_plot - np.power(logbase, y - yerr * num_se)
    yerr_plot = np.stack([yerr_min, yerr_max])
    return x_plot, y_plot, xerr_plot, yerr_plot