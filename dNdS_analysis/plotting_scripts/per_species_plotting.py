
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

def _nice_axes(ax):
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

def _compute_purify_curve(purify, x_default):
    """
    Accepts a dict 'purify' with any of the following:
      - 'x': precomputed x grid (optional; defaults to x_default)
      - 'y': precomputed y values (optional if 'curve_fn' given)
      - 'curve_fn': callable like y = curve_fn(x, **kwargs)
      - 'kwargs': dict passed to curve_fn
      - 'ci': either
            {'y_lo': array, 'y_hi': array}
         or {'lo_kwargs': {...}, 'hi_kwargs': {...}} to be passed to curve_fn
      - 'label': legend label (optional)
      - styling (optional): 'color', 'ci_color', 'ls', 'lw', 'alpha', 'zorder'
    Returns a dict with keys:
      {'x','y','y_lo','y_hi','label','color','ci_color','ls','lw','alpha','zorder'}
    """
    if purify is None:
        return None

    # grid
    x = purify.get('x', None)
    if x is None:
        x = x_default
    else:
        x = np.asarray(x, dtype=float)

    # central curve
    y = purify.get('y', None)
    if y is None:
        fn = purify.get('curve_fn', None)
        kw = purify.get('kwargs', {}) or {}
        if fn is None:
            return None
        y = np.asarray(fn(x, **kw), dtype=float)
    else:
        y = np.asarray(y, dtype=float)

    # CI
    y_lo = y_hi = None
    ci = purify.get('ci', None)
    if ci is not None:
        if ('y_lo' in ci) and ('y_hi' in ci):
            y_lo = np.asarray(ci['y_lo'], dtype=float)
            y_hi = np.asarray(ci['y_hi'], dtype=float)
        else:
            fn = purify.get('curve_fn', None)
            if fn is not None:
                lo_kw = ci.get('lo_kwargs', None)
                hi_kw = ci.get('hi_kwargs', None)
                if lo_kw is not None and hi_kw is not None:
                    y_lo = np.asarray(fn(x, **lo_kw), dtype=float)
                    y_hi = np.asarray(fn(x, **hi_kw), dtype=float)

    return {
        'x': x,
        'y': y,
        'y_lo': y_lo,
        'y_hi': y_hi,
        'label': purify.get('label', None),
        'color': purify.get('color', None),
        'ci_color': purify.get('ci_color', None),
        'ls': purify.get('ls', '-'),
        'lw': purify.get('lw', 1.8),
        'alpha': purify.get('alpha', 1.0),
        'zorder': purify.get('zorder', 6),
    }

def _normalize_purify(purify, x_default):
    """
    Normalize 'purify' into a list of overlay specs.
    Supported inputs:
      - None: returns []
      - dict: a single spec or a dict with {'curves': [specs...]}
      - list[dict]: multiple specs
    Each spec is processed by _compute_purify_curve.
    """
    if purify is None:
        return []

    specs_in = None
    if isinstance(purify, dict):
        if 'curves' in purify and isinstance(purify['curves'], (list, tuple)):
            specs_in = purify['curves']
        else:
            specs_in = [purify]
    elif isinstance(purify, (list, tuple)):
        specs_in = list(purify)
    else:
        raise ValueError("purify must be None, a dict, or a list of dicts.")

    out = []
    for spec in specs_in:
        s = _compute_purify_curve(spec, x_default)
        if s is not None:
            out.append(s)
    return out

def plot_species_grid_styled(
    results,
    which_class="all",
    COLORS=None,
    species_dnds_tidy=None,     # tidy df with per-species unrelated-pair averages
    species_col="species",
    x_col="dS_x",
    y_col="R_hat",
    ncols=4,
    sharex=True,
    sharey=True,
    xlim=(1e-6, 1e-1),
    ylim=(0.03, 20.0),
    shade_xmin=1e-3,
    shade_xmax=1.0,
    legend_mode="first",         # 'first' | 'none' | 'each'
    purify=None,                 # single dict or list of dicts (multi-class overlays)
    sort_by="pairs",             # 'pairs' | 'name' | callable(sp, SpeciesResults) -> key
    sort_desc=True,              # descending order when sorting
    top_n=None                   # optionally show only top-N species after sorting
):
    """
    Styled small-multiples grid for per-species dN/dS dynamics.

    Purifying overlays:
      - Pass a single dict (as before) for one curve, e.g. aggregate fit.
      - Pass a list of dicts or {'curves': [ ... ]} to draw multiple curves
        (e.g., 2-class or 3-class model components), each with its own
        label/style/color and optional CI.
      - Each dict may include keys:
          'curve_fn', 'kwargs', 'label', 'color', 'ci_color', 'ls', 'lw', 'alpha', 'zorder', 'ci'
        where 'ci' is either {'y_lo','y_hi'} or {'lo_kwargs','hi_kwargs'}.

    Sorting:
      - sort_by='pairs' sorts by SpeciesResults.n_pairs
      - sort_by='name'  sorts by species name
      - sort_by=callable(sp, res) lets you provide a custom key
    """

    if COLORS is None:
        raise ValueError("Please pass the COLORS dict you used for single-panel plots.")

    if not results:
        raise ValueError("Empty results dict.")

    species = list(results.keys())

    # Apply sorting
    if callable(sort_by):
        species.sort(key=lambda sp: sort_by(sp, results[sp]), reverse=sort_desc)
    elif sort_by == "name":
        species.sort(reverse=sort_desc)
    else:  # default: 'pairs'
        species.sort(key=lambda sp: results[sp].n_pairs, reverse=sort_desc)

    # Optionally trim to top-N
    if (top_n is not None) and (top_n > 0):
        species = species[:int(top_n)]

    n = len(species)
    ncols = max(1, int(ncols))
    nrows = (n + ncols - 1) // ncols

    # Compact, readable figure size that scales with rows
    figsize = (16, max(3.0, 2.0 + 2.0*nrows))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize,
                             sharex=sharex, sharey=sharey)
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    legend_done = False

    # Prepare a default x-grid for the purify model (log spacing within current xlim)
    x_default = np.geomspace(max(xlim[0], 1e-12), xlim[1], 400)
    purify_specs = _normalize_purify(purify, x_default)

    # color fallbacks for multiple overlays if not specified
    fit_colors = [
        COLORS.get("fit", "#D55E00"),
        COLORS.get("fit2", "#E69F00"),
        COLORS.get("fit3", "#56B4E9"),
        COLORS.get("fit4", "#009E73"),
    ]

    for idx, sp in enumerate(species):
        r = results[sp]
        ax = axes[idx // ncols, idx % ncols]
        _nice_axes(ax)

        # scatter cloud (thin + subtle)
        x_sc, y_sc = r.scatter_by_class[which_class]
        sns.scatterplot(
            x=x_sc, y=y_sc,
            s=6, linewidth=0, alpha=0.20, rasterized=True,
            legend=False, ax=ax, color=COLORS["cloud"], zorder=1
        )

        # binned medians with 95% CI as error bars (from bootstrap quantiles)
        agg0, boots = r.binned_by_class[which_class]
        if (len(agg0) > 0) and (len(boots) > 0):
            x = agg0["dS_x"].to_numpy(float)
            y_med = boots["med"].to_numpy(float)
            y_lo = boots["lo"].to_numpy(float)
            y_hi = boots["hi"].to_numpy(float)

            # convert ribbon to asymmetric error bars relative to the median
            yerr_lo = np.clip(y_med - y_lo, 0, np.inf)
            yerr_hi = np.clip(y_hi - y_med, 0, np.inf)

            valid = np.isfinite(x) & np.isfinite(y_med) & np.isfinite(yerr_lo) & np.isfinite(yerr_hi)
            if valid.any():
                ax.errorbar(
                    x[valid], y_med[valid],
                    yerr=[yerr_lo[valid], yerr_hi[valid]],
                    fmt="-o", lw=1.5, ms=4.5, mew=0.0,
                    color=COLORS["trend"], ecolor=COLORS["trend"],
                    elinewidth=1.0, capsize=2.5, capthick=1.0,
                    zorder=4, label="Bin median ± 95% CI"
                )

            # single-sample ratio-of-totals (faint reference) at binned x
            ax.plot(
                agg0["dS_x"], agg0["R_hat"],
                ".", ms=3, alpha=0.7, color=COLORS["trend"], zorder=3
            )

        # Optional: overlay one or more purifying curves (aggregate params / class components)
        for i, spec in enumerate(purify_specs):
            color = spec['color'] if spec['color'] is not None else fit_colors[i % len(fit_colors)]
            ci_color = spec['ci_color'] if spec['ci_color'] is not None else color
            label = spec['label'] if spec['label'] is not None else "Purifying fit"
            ax.plot(spec['x'], spec['y'], lw=spec['lw'], ls=spec['ls'],
                    color=color, alpha=spec['alpha'], zorder=spec['zorder'], label=label)
            if spec['y_lo'] is not None and spec['y_hi'] is not None:
                ax.fill_between(spec['x'], spec['y_lo'], spec['y_hi'],
                                color=ci_color, alpha=0.15, zorder=max(0, spec['zorder']-1))

        # species-wise unrelated-pair mean as a cross
        if species_dnds_tidy is not None:
            row = species_dnds_tidy.loc[species_dnds_tidy[species_col] == sp]
            if len(row) >= 1:
                xv = float(row.iloc[0][x_col])
                yv = float(row.iloc[0][y_col])
                ax.scatter(
                    xv, yv, marker="X", s=50, linewidths=0.8,
                    edgecolors="white", color=COLORS["species"],
                    label="Unrelated pairs (species mean)",
                    zorder=7
                )

        # shaded region + neutral line
        ax.axvspan(shade_xmin, shade_xmax, color=COLORS["shade"], alpha=0.12, zorder=0)
        ax.axhline(1.0, color=COLORS["neutral"], linestyle="-", lw=1.0, zorder=2)

        # scales, limits, titles
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_title(f"{sp} (n={r.n_pairs})", fontsize=10)

        # legend handling
        if legend_mode == "each":
            shade_handle = Patch(facecolor=COLORS["shade"], alpha=0.12, label="Approx. detection limit")
            handles, labels = ax.get_legend_handles_labels()
            handles.append(shade_handle); labels.append("Approx. detection limit")
            # Deduplicate
            uniq = []
            seen = set()
            for h, lab in zip(handles, labels):
                if lab not in seen:
                    uniq.append((h, lab)); seen.add(lab)
            handles, labels = zip(*uniq)
            ax.legend(handles, labels, frameon=True, loc="lower left", ncol=1, handlelength=2.2, fontsize=7)

        elif (legend_mode == "first") and (not legend_done):
            shade_handle = Patch(facecolor=COLORS["shade"], alpha=0.12, label="Approx. detection limit")
            handles, labels = ax.get_legend_handles_labels()
            handles.append(shade_handle); labels.append("Approx. detection limit")
            # Deduplicate
            uniq = []
            seen = set()
            for h, lab in zip(handles, labels):
                if lab not in seen:
                    uniq.append((h, lab)); seen.add(lab)
            handles, labels = zip(*uniq)
            ax.legend(handles, labels, frameon=True, loc="lower left", ncol=1, handlelength=2.2, fontsize=7)
            legend_done = True

    # Turn off any empty axes
    for j in range(len(species), nrows*ncols):
        axes[j // ncols, j % ncols].axis('off')

    # shared labels
    fig.text(0.5, 0.02, r"$dS$ (Clonal region)", ha="center")
    fig.text(0.01, 0.5, r"$dN/dS$ (Clonal region)", va="center", rotation="vertical")

    fig.tight_layout(rect=(0.03, 0.03, 1, 1))
    return fig, axes
