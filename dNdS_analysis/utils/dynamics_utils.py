import numpy as np

def computed_poisson_thinning(diffs, opportunities):
    # apply this to calculation of all dN/dS
    # specifically when calculating dS
    thinned_diffs_1 = np.random.binomial(diffs, 0.5)
    thinned_diffs_2 = diffs - thinned_diffs_1
    d1 = thinned_diffs_1 / (opportunities.astype(float) / 2)
    d2 = thinned_diffs_2 / (opportunities.astype(float) / 2)
    return d1, d2