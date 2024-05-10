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
    return (1-fd) + fd * (1-np.exp(-sbymu * dS)) / (sbymu * dS)