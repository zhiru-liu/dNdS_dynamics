import numpy as np
def sample_fr(dsc, nsamples, rbymu, L, lr):
    n_events = np.random.poisson(dsc * rbymu * L, size=nsamples)
#     print(n_events.mean())
    fr = n_events.copy()
    mask = fr > 0
    fr[mask] = np.random.gamma(shape=n_events[mask], scale=lr)
    fr = fr / L
    fr = np.minimum(fr, 1)
    return fr

def sample_fr_semi(dsc, nsamples, rbymu, L, lr):
    num_event = (L * (1 - np.exp(-rbymu * dsc))).astype(int)
    if num_event==0:
        lr = np.zeros(nsamples)
    else:
        lr = np.random.gamma(shape=num_event, scale=1e4, size=nsamples)
    fr_arr = np.minimum(num_event * lr / L, 1)
    return fr_arr

def logistic_accumulation(dsc, ds_mid, k):
    return 1 / (1 + np.exp(-k * (np.log10(dsc) - np.log10(ds_mid))))

def compute_dNdS_rec_model(ds_clonal, fr, theta, dNdS_clonal, dNdS_rec):
    dS = (1 - fr) * ds_clonal + fr*theta
    dN = ((1 - fr) * dNdS_clonal * ds_clonal + fr * dNdS_rec * theta) 
    dNdS = dN / dS
    return dS, dNdS

def dNdS_purify_curve(dS, fd, sbymu):
    """
    Purifying selection model for dN/dS ratio.
    fd: fraction of deleterious mutations
    sbymu: selection coefficient divided by mutation rate
    dS: synonymous divergence
    Returns the predicted dN/dS ratio.
    """
    return (1-fd) + fd * (1-np.exp(-sbymu * dS / 2)) / (sbymu * dS / 2)

def dNdS_purify_curve_three_class(dS, alpha0, alpha1, sbymu1, sbymu2):
    """
    Three class DFE model: neutral, weakly deleterious, and strongly deleterious.
    alpha0: neutral fraction
    alpha1: weakly deleterious fraction
    sbymu1: selection coefficient for weakly deleterious mutations
    sbymu2: selection coefficient for strongly deleterious mutations
    dS: synonymous divergence
    Returns the predicted dN/dS ratio averaging over three classes.
    """
    alpha2 = 1 - alpha0 - alpha1
    if alpha2 < 0:
        raise ValueError("Invalid alpha values: alpha0 + alpha1 must be <= 1")
    
    weak_contri = (1-np.exp(-sbymu1 * dS / 2)) / (sbymu1 * dS / 2)
    strong_contri = (1-np.exp(-sbymu2 * dS / 2)) / (sbymu2 * dS / 2)

    total_result = alpha0 + alpha1 * weak_contri + alpha2 * strong_contri
    return total_result