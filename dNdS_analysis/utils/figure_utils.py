import matplotlib as mpl

mpl.rcParams['font.size'] = 7
mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['legend.frameon'] = False

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