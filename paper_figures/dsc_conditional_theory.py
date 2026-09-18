"""Empirical ``<dSc|dS>`` accumulation + conditional-expectation dN/dS theory.

Staging module for the revision's alternative to the ``f_r(dSc)`` Hill-based
theory curve (``fr_dSc_with_typical.py`` / ``plot_dNdS_separation.py``).

Instead of fitting ``f_r(dSc)`` with a Hill/logistic function and propagating it
through the recombination mixture model, we fit the **conditional expectation of
the clonal synonymous divergence given the total (full-genome) synonymous
divergence**, ``<dSc | dS>``, and plug it into the closed-form conditional dN/dS
average (main.tex, ``eq:phenomenological-dNdS-model-ratio-average``):

    <dN(T)/dS(T) | dS(T)=dS>  ~=  dNbar/dSbar
                                  + (1 - dNbar/dSbar) * <dSc|dS> / dS .

Derivation (from ``eq:phenomenological-dNdS-model`` with clonal dN/dS = 1, i.e.
``dNc ~= dSc``, and ``f_r << 1``):

    dS = (1 - f_r) dSc + f_r dSbar
    dN = (1 - f_r) dSc + f_r dNbar
       = (1 - f_r) dSc [1 - dNbar/dSbar] + (dNbar/dSbar) dS
    => dN/dS = dNbar/dSbar + (1 - dNbar/dSbar) * (1 - f_r) dSc / dS
             ~= dNbar/dSbar + (1 - dNbar/dSbar) * dSc / dS     (f_r << 1)

so the only empirical ingredient is the accumulation curve ``<dSc|dS>`` and the
independently-measured recombined ratio ``dNbar/dSbar``.

Everything here is written so it can be lifted into
``dnds_dynamics.figures.{dynamics,theory}`` when we merge.
"""
from __future__ import annotations

import functools

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.interpolate import UnivariateSpline

from dnds_dynamics.figures import dynamics as _dyn


# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #
def load_pair_table(full_dnds_df=None, transfer_df=None):
    """Per-pair table of full-genome dS/dN, recombined fraction, and dSc.

    Returns a DataFrame indexed by ``(species_name, sample 1, sample 2)`` with:

    - ``dS``            -- full-genome 4D synonymous divergence (core_diff/len).
    - ``dN``            -- full-genome 1D nonsynonymous divergence.
    - ``fr``            -- recombined fraction ``1 - clonal_fraction`` (0 where no
                           transfer was detected, i.e. purely-clonal pairs).
    - ``dSc_region``    -- clonal-*region* synonymous divergence ``d_S^{(c)}``
                           (the ``Clonal divergence`` column); for purely-clonal
                           pairs the clonal region is the whole genome so this
                           falls back to ``dS``.
    - ``clonal_contrib``-- clonal *contribution* to the total, ``(1 - f_r) dSc``.
                           This is the exact coefficient in the model; it equals
                           ``dSc_region`` in the ``f_r << 1`` regime.
    """
    if full_dnds_df is None:
        full_dnds_df = _dyn.load_dNdS_data()
    if transfer_df is None:
        transfer_df = _dyn.load_detected_transfers()
    transfer_df = transfer_df.rename(columns={"Species name": "species_name",
                                              "Sample 1": "sample 1",
                                              "Sample 2": "sample 2"})
    transfer_df = transfer_df.set_index(["species_name", "sample 1", "sample 2"])
    means = (transfer_df.groupby(level=["species_name", "sample 1", "sample 2"])
             [["Clonal divergence", "Clonal fraction"]].mean())

    df = pd.DataFrame(index=full_dnds_df.index)
    df["dS"] = pd.to_numeric(full_dnds_df["core_diff_4D"] / full_dnds_df["core_len_4D"],
                             errors="coerce")
    df["dN"] = pd.to_numeric(full_dnds_df["core_diff_1D"] / full_dnds_df["core_len_1D"],
                             errors="coerce")
    df["cf"] = pd.to_numeric(means["Clonal fraction"].reindex(full_dnds_df.index),
                             errors="coerce")
    df["cdiv"] = pd.to_numeric(means["Clonal divergence"].reindex(full_dnds_df.index),
                               errors="coerce")
    df["fr"] = (1 - df["cf"]).fillna(0.0)
    df["dSc_region"] = df["cdiv"].where(df["cdiv"].notna(), df["dS"]).astype(float)
    df["clonal_contrib"] = (1 - df["fr"]) * df["dSc_region"]
    return df


def estimate_recombined_ratio(typical_df=None):
    """Measure ``dSbar``, ``dNbar`` and ``dNbar/dSbar`` from typical (fully
    recombined) pairs, as median-over-species of the per-species median rate.

    Returns a dict with ``dSbar`` (a.k.a. ``theta``), ``dNbar``,
    ``ratio`` (= dNbar/dSbar, ratio of medians) and ``ratio_med_of_ratios``.
    """
    if typical_df is None:
        typical_df = _dyn.load_typical_pair_dNdS_data()
    typical_df = typical_df.copy()
    sp = typical_df.groupby(level="species_name").agg(
        d4=("core_diff_4D", "median"), l4=("core_len_4D", "median"),
        d1=("core_diff_1D", "median"), l1=("core_len_1D", "median"))
    dS_sp = sp["d4"] / sp["l4"]
    dN_sp = sp["d1"] / sp["l1"]
    dSbar = float(np.median(dS_sp))
    dNbar = float(np.median(dN_sp))
    return {"dSbar": dSbar, "theta": dSbar, "dNbar": dNbar,
            "ratio": dNbar / dSbar,
            "ratio_med_of_ratios": float(np.median(dN_sp / dS_sp)),
            "n_species": int(len(dS_sp))}


# --------------------------------------------------------------------------- #
# Accumulation functions for <dSc|dS>
#
# All are constrained so that <dSc|dS> -> dS as dS -> 0 (fully-clonal limit,
# where the whole genome is clonal so the clonal divergence equals the total),
# and saturate at a plateau ``ds_star`` as dS grows (recombination overwrites the
# clonal backbone once dSc reaches ~ds_star).
# --------------------------------------------------------------------------- #
def hyperbolic_accumulation(dS, ds_star):
    """``<dSc|dS> = dS * ds_star / (dS + ds_star)`` (harmonic soft-min).

    Equivalent to ``1/<dSc> = 1/dS + 1/ds_star``. One parameter; slope 1 at the
    origin, plateau ``ds_star``. This is the best-fit / recommended form.
    """
    dS = np.asarray(dS, float)
    return dS * ds_star / (dS + ds_star)


def softmin_accumulation(dS, ds_star, p):
    """p-norm soft-min of ``dS`` and ``ds_star``:
    ``(dS**-p + ds_star**-p)**(-1/p)``.

    Generalizes :func:`hyperbolic_accumulation` (``p=1``); larger ``p`` sharpens
    the elbow between the diagonal and the plateau. Empirically ``p ~= 1.1`` so
    the extra parameter buys almost nothing (kept for diagnostics).
    """
    dS = np.asarray(dS, float)
    return (dS ** (-p) + ds_star ** (-p)) ** (-1.0 / p)


def exp_accumulation(dS, ds_star):
    """``<dSc|dS> = ds_star * (1 - exp(-dS/ds_star))``.

    One parameter; slope 1 at the origin, plateau ``ds_star``. Alternative to the
    hyperbolic form (slightly worse empirical fit).
    """
    dS = np.asarray(dS, float)
    return ds_star * (1.0 - np.exp(-dS / ds_star))


def log_accumulation(dS, ds_star):
    """``<dSc|dS> = ds_star * ln(1 + dS/ds_star)``.

    One parameter; slope 1 at the origin, then grows only logarithmically (never
    plateaus, never diverges fast). A minimal 'no hard plateau' option, but it
    fits the data worse than the hyperbolic/mean-field forms (R^2~0.73).
    """
    dS = np.asarray(dS, float)
    return ds_star * np.log1p(dS / ds_star)


def arcsinh_accumulation(dS, ds_star):
    """``<dSc|dS> = ds_star * asinh(dS/ds_star)``.

    Like :func:`log_accumulation` (slope 1 at origin, ~log growth at large dS)
    but with a smoother crossover; better empirical fit (R^2~0.87).
    """
    dS = np.asarray(dS, float)
    return ds_star * np.arcsinh(dS / ds_star)


# Recombined-region synonymous divergence used by the mean-field forms below.
# Measured value ~2.5e-2 (median typical-pair dS); pass the exact value from
# estimate_recombined_ratio()["dSbar"] when you can.
DEFAULT_DSBAR = 2.514e-2


def _meanfield_invert(dS_query, h_of_x, dsbar):
    """Invert the deterministic mean relation dS(x) = x(1-h(x)) + h(x)*dsbar to
    recover x = <dSc> as a function of the observed total dS. dS(x) is strictly
    increasing in x, so a monotone interpolation suffices."""
    xg = np.logspace(-7, np.log10(dsbar * 0.99999), 6000)
    h = h_of_x(xg)
    dS_of_x = xg * (1.0 - h) + h * dsbar
    return np.interp(np.asarray(dS_query, float), dS_of_x, xg)


def meanfield_hill_accumulation(dS, xstar, k, dsbar=DEFAULT_DSBAR):
    """Model-derived ``<dSc|dS>``: invert the recombination mixture

        dS = dSc*(1 - f_r) + f_r*dsbar,   f_r = dSc^k / (dSc^k + xstar^k),

    i.e. the SAME steep-Hill accumulation used for ``f_r(dSc)`` in the published
    figure, re-expressed as the clonal divergence vs total divergence relation.

    This is the recommended form: best empirical fit (R^2~0.94), physically
    grounded, self-consistent with the ``f_r(dSc)`` fit (xstar~1.4e-4, k~8 here
    vs dS*~1.2e-4, k~6 there), and it does NOT impose a hard plateau -- it yields
    a gentle continued rise at high dS (tail slope ~0.16), matching the data,
    and only truly saturates by rejoining the diagonal at dS=dsbar (off-scale).
    """
    return _meanfield_invert(dS, lambda xg: xg ** k / (xg ** k + xstar ** k), dsbar)


def generalized_accumulation(dS, ds_star, gamma):
    """``<dSc|dS> = dS / (1 + (dS/ds_star)**gamma)``.

    Generalizes :func:`hyperbolic_accumulation` (``gamma=1``) so the large-dS
    behaviour is NOT forced to a constant plateau:

    - ``gamma = 1``  -> ``G -> ds_star`` (hard plateau; hyperbolic).
    - ``gamma < 1``  -> ``G ~ dS**(1-gamma)`` keeps rising (no plateau).
    - ``gamma > 1``  -> ``G`` turns over and decreases.

    Slope 1 at the origin for any ``gamma``. Because the *ratio* ``G/dS =
    1/(1+(dS/ds_star)**gamma)`` still -> 0 as dS grows (for gamma>0), the dN/dS
    theory curve is only weakly sensitive to ``gamma``; it mainly controls how
    faithfully the ``<dSc|dS>`` panel is described (see notes in the figure
    script). This is the form to use when a hard plateau is not desired.
    """
    dS = np.asarray(dS, float)
    return dS / (1.0 + (dS / ds_star) ** gamma)


_ACCUMULATION = {
    # empirical, saturating (plateau)
    "hyperbolic":  (hyperbolic_accumulation,  [1e-4], [(1e-5, 1e-1)]),
    "softmin":     (softmin_accumulation,     [1e-4, 1.3], [(1e-5, 1e-1), (0.3, 8.0)]),
    "exp":         (exp_accumulation,         [1e-4], [(1e-5, 1e-1)]),
    # empirical, non-saturating (keep rising)
    "generalized": (generalized_accumulation, [1e-4, 1.0], [(1e-5, 1e-1), (0.3, 2.0)]),
    "log":         (log_accumulation,         [1e-4], [(1e-6, 1e-1)]),
    "arcsinh":     (arcsinh_accumulation,     [1e-4], [(1e-6, 1e-1)]),
    # model-derived (recommended)
    "meanfield":   (meanfield_hill_accumulation, [1e-4, 8.0], [(1e-5, 1e-2), (1.0, 25.0)]),
}


# --------------------------------------------------------------------------- #
# Binning + fitting <dSc|dS>
# --------------------------------------------------------------------------- #
def bin_conditional(dS, dSc, bins=24, min_count=30, floor=1e-6, estimator="mean"):
    """Estimate ``<dSc | dS>`` in log-spaced bins of ``dS``.

    Returns ``(x, y, n)`` where ``x`` is the (geo-mean) bin dS, ``y`` summarizes
    dSc in the bin and ``n`` is the per-bin pair count. Bins with fewer than
    ``min_count`` pairs are dropped.

    ``estimator``: ``"mean"`` (arithmetic mean = the conditional expectation, the
    theory-correct quantity for the dN/dS formula) or ``"median"`` (a robustness
    check; lower than the mean because the per-bin dSc distribution is
    right-skewed, so it is NOT the exact conditional expectation).
    """
    dS = np.asarray(dS, float)
    dSc = np.asarray(dSc, float)
    m = np.isfinite(dS) & np.isfinite(dSc) & (dS > 0)
    dS, dSc = dS[m], dSc[m]
    summarize = np.median if estimator == "median" else np.mean
    edges = np.logspace(np.log10(max(dS.min(), floor)), np.log10(dS.max()), bins + 1)
    idx = np.digitize(dS, edges) - 1
    xs, ys, ns = [], [], []
    for i in range(bins):
        sel = idx == i
        if sel.sum() >= min_count:
            xs.append(10 ** np.mean(np.log10(dS[sel])))
            ys.append(summarize(dSc[sel]))
            ns.append(int(sel.sum()))
    return np.array(xs), np.array(ys), np.array(ns)


def fit_conditional_accumulation(dS, dSc, model="hyperbolic",
                                 bins=24, min_count=30, floor=1e-6, dsbar=None):
    """Fit an accumulation function to the binned ``<dSc|dS>``.

    Fit is done in log space, weighting each bin by its pair count. Returns a
    dict with ``params`` (tuple), ``r2`` (count-weighted log-space R^2), the
    binned data ``(x, y, n)``, the chosen ``func`` and ``model`` name.

    ``dsbar`` is only used by the ``"meanfield"`` model (the recombined-region
    ceiling); pass ``estimate_recombined_ratio()["dSbar"]`` so the inversion uses
    the measured value rather than the module default. The returned ``func`` has
    ``dsbar`` bound, so downstream ``conditional_dNdS`` stays consistent.
    """
    func, p0, bounds = _ACCUMULATION[model]
    if model == "meanfield" and dsbar is not None:
        func = functools.partial(func, dsbar=dsbar)
    x, y, n = bin_conditional(dS, dSc, bins=bins, min_count=min_count, floor=floor)

    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    def objective(params):
        pred = np.clip(func(x, *params), 1e-12, None)
        return np.sum(n * (np.log(y) - np.log(pred)) ** 2)

    # multistart to avoid local minima
    starts = [np.array(p0, float)]
    if len(p0) == 2:
        starts += [np.array([p0[0], pp]) for pp in (0.5, 1.0, 2.0, 4.0)]
    best = None
    for s in starts:
        s = np.clip(s, lo, hi)
        res = minimize(objective, s, bounds=bounds, method="L-BFGS-B")
        if best is None or res.fun < best.fun:
            best = res

    pred = np.clip(func(x, *best.x), 1e-12, None)
    ss_res = np.sum(n * (np.log(y) - np.log(pred)) ** 2)
    ss_tot = np.sum(n * (np.log(y) - np.average(np.log(y), weights=n)) ** 2)
    return {"model": model, "func": func, "params": tuple(best.x),
            "r2": float(1 - ss_res / ss_tot), "x": x, "y": y, "n": n}


def fit_spline_accumulation(dS, dSc, smooth=0.6, k=3, diag_below=None,
                            bins=24, min_count=30, floor=1e-6, estimator="mean"):
    """Non-parametric ``G(dS)`` via a count-weighted smoothing spline of
    ``log10<dSc>`` vs ``log10 dS``.

    No functional-form assumption (no plateau, no mechanism). Two physical
    conditions are imposed on evaluation:

    - ``G = min(spline, dS)`` enforces the bound ``G <= dS`` (and holds the ratio
      flat above the fitted support -- bounded extrapolation).
    - for ``dS <= diag_below`` the fit is replaced by the clonal diagonal
      ``G = dS`` (so ``dN/dS -> 1``). This removes the low-dS spline-boundary
      wiggle, which otherwise produces a spurious ``dN/dS < 1`` dip at the
      smallest divergences. ``diag_below`` is a *dS threshold*, not merely the
      lowest data point.

    If ``diag_below is None`` it is set automatically to the largest ``dS`` within
    the fitted support at which the raw spline still sits on/above the diagonal
    (the last clonal crossing), which makes the diagonal->spline handoff
    continuous. Pass a float to override.

    ``smooth`` scales the spline smoothing factor ``s = smooth * n_bins``
    (larger = smoother; ~0.6 keeps the shoulder without chasing bin noise).
    Returns the usual fit dict; ``func`` is the evaluator (call as ``func(dS)``),
    ``params`` is empty, ``support`` is the trusted ``(dS_min, dS_max)`` and
    ``diag_below`` is the diagonal-replacement threshold used.
    """
    x, y, n = bin_conditional(dS, dSc, bins=bins, min_count=min_count, floor=floor,
                              estimator=estimator)
    lx, ly = np.log10(x), np.log10(y)
    lo, hi = float(lx.min()), float(lx.max())
    spl = UnivariateSpline(lx, ly, w=np.sqrt(n), k=k, s=smooth * len(x))

    if diag_below is None:
        grid = np.logspace(lo, hi, 500)
        on_diagonal = grid[10 ** spl(np.log10(grid)) >= grid]   # raw spline >= dS
        diag_below = float(on_diagonal.max()) if on_diagonal.size else 10 ** lo

    def func(dS_query, *_ignored):
        dS_query = np.asarray(dS_query, float)
        lq = np.clip(np.log10(np.maximum(dS_query, 1e-300)), lo, hi)
        g = np.minimum(10 ** spl(lq), dS_query)
        return np.where(dS_query <= diag_below, dS_query, g)     # diagonal below thr

    pred = np.clip(func(x), 1e-12, None)
    ss_res = np.sum(n * (np.log(y) - np.log(pred)) ** 2)
    ss_tot = np.sum(n * (np.log(y) - np.average(np.log(y), weights=n)) ** 2)
    return {"model": "spline", "func": func, "params": (), "spline": spl,
            "support": (10 ** lo, 10 ** hi), "diag_below": diag_below,
            "r2": float(1 - ss_res / ss_tot), "x": x, "y": y, "n": n}


# --------------------------------------------------------------------------- #
# Conditional-expectation dN/dS theory curve
# --------------------------------------------------------------------------- #
def conditional_dNdS(dS, fit, ratio):
    """New theory curve ``<dN/dS | dS>``.

    ``fit``   -- output of :func:`fit_conditional_accumulation` (uses its
                 ``func`` + ``params`` for the accumulation ``<dSc|dS>``).
    ``ratio`` -- the measured recombined ratio ``dNbar/dSbar``.

        <dN/dS|dS> = ratio + (1 - ratio) * <dSc|dS> / dS
    """
    dS = np.asarray(dS, float)
    dsc = fit["func"](dS, *fit["params"])
    return ratio + (1.0 - ratio) * dsc / dS


def conditional_dNdS_from_accumulation(dS, accumulation, ratio):
    """Same as :func:`conditional_dNdS` but from a precomputed ``<dSc|dS>``
    array (``accumulation``), so callers can supply any accumulation model."""
    dS = np.asarray(dS, float)
    accumulation = np.asarray(accumulation, float)
    return ratio + (1.0 - ratio) * accumulation / dS


__all__ = [
    "load_pair_table", "estimate_recombined_ratio",
    "hyperbolic_accumulation", "softmin_accumulation", "exp_accumulation",
    "generalized_accumulation", "log_accumulation", "arcsinh_accumulation",
    "meanfield_hill_accumulation", "DEFAULT_DSBAR",
    "bin_conditional", "fit_conditional_accumulation", "fit_spline_accumulation",
    "conditional_dNdS", "conditional_dNdS_from_accumulation",
]
