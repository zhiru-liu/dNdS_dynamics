"""Demo / validation for cphmm 1D tract extension on the two worked examples.

Loads the two A. putredinis pairs from ``response_missed_recomb_examples`` (the QP
metagenome pair and the cultured-isolate pair), feeds each pair's detected tracts
plus its 1D SNV differences to ``cphmm.tract_extension.extend_tracts_by_1d_density``,
and checks that the known "missed" 1D-rich flank (yellow in the example figure) is
swallowed by an extended tract. Re-makes the zoom figure with the *extended*
boundaries shaded grey -- the yellow flank should now sit inside grey.

Run: python paper_figures/tract_extension_demo.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "paper_figures"))
MPL_CACHE_DIR = REPO_ROOT / ".cache" / "matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

from cphmm.tract_extension import (  # noqa: E402
    ExtensionParams,
    extend_tracts_by_1d_density,
    extend_tracts_by_density,
)
import response_missed_recomb_examples as ex  # noqa: E402

WIN = ex.WIN
OUT = REPO_ROOT / "figures" / "response_missed_recomb_extended"
OUT_ALL = REPO_ROOT / "figures" / "response_missed_recomb_extended_allsnv"
RED, BLUE, TRACT, DET = ex.RED, ex.BLUE, ex.TRACT, ex.DET
EXT = "#3a8f3a"  # extended-boundary shading


def _transfer_df_from_intervals(intervals, pair):
    """Detected tracts (grey) -> the transfer_df schema the extender consumes."""
    rows = []
    for contig, start, end in intervals:
        rows.append({
            "genome1": pair[0], "genome2": pair[1], "contig": str(contig),
            "types": 0, "start_site": int(start), "end_site": int(end),
        })
    return pd.DataFrame(rows)


def _snp_info(ex_dict, site_mask):
    """(snp_vec, contigs, locs) over one site class, from the example arrays.

    ``site_mask`` is the 1D or 4D boolean mask over the SNV index; ``diff`` marks
    where the pair differs. We expose every site of that class as a covered site
    and flag the differences -- exactly the get_pair_snp_info(site_class=...) form.
    """
    sc, sp, diff = ex_dict["sc"], ex_dict["sp"], ex_dict["diff"]
    sel = site_mask
    return diff[sel], sc[sel].astype(str), sp[sel].astype(int)


def run_example(ex_dict, pair, params, driver="1D"):
    """Extend the pair's detected tracts; ``driver`` is '1D' or 'all'.

    '1D' uses the narrow nonsynonymous-driven path; 'all' drives on every SNV
    (so 2D/3D-rich flanks are also absorbed), via the generic entry point.
    """
    snp_4d = _snp_info(ex_dict, ex_dict["s4m"])
    detected = _transfer_df_from_intervals(ex_dict["recomb_intervals"], pair)

    if driver == "1D":
        snp_1d = _snp_info(ex_dict, ex_dict["s1m"])
        extended = extend_tracts_by_1d_density(
            detected, snp_1d, params=params, snp_info_4d=snp_4d
        )
        n_col = "extension_1d_snvs"
    else:  # 'all' SNVs
        snp_all = _snp_info(ex_dict, np.ones(len(ex_dict["sc"]), dtype=bool))
        extended = extend_tracts_by_density(
            detected, snp_all, params=params, count_infos={"4d": snp_4d}
        )
        n_col = "extension_snvs"
    extended = extended.rename(columns={n_col: "extension_n"})

    # The known missed flank: the densest clonal 1D window the example located.
    contig, ws = ex_dict["contig"], ex_dict["ws"]
    flank_lo, flank_hi = ws, ws + WIN
    on_contig = extended[extended["contig"] == contig]
    covered = (
        (on_contig["start_site"] <= flank_lo) & (on_contig["end_site"] >= flank_hi)
    ).any()
    # also report the specific tract that grew over the flank
    grew = on_contig[(on_contig["start_site"] <= flank_hi)
                     & (on_contig["end_site"] >= flank_lo)]
    return extended, covered, grew


GREY = "#7f8c8d"  # 2D/3D ("other degeneracy") SNV ticks


def _zoom_panel(ax, ex_dict, extended, halfspan=6000, driver_tag="", mode="split"):
    """Zoom view of one pair around its missed flank.

    ``mode='split'`` plots 1D ticks up / 4D ticks down (the 1D-driver view).
    ``mode='all'`` plots every SNV up (coloured by class: red 1D, grey 2D/3D,
    blue 4D) -- what the all-SNV driver sees -- while keeping the 4D-only row down,
    since 4D density is what the CP-HMM originally detects on.
    """
    c, ws = ex_dict["contig"], ex_dict["ws"]
    lo, hi = ws - halfspan, ws + WIN + halfspan
    reg = (ex_dict["sc"] == c) & (ex_dict["sp"] >= lo) & (ex_dict["sp"] <= hi)
    diff = ex_dict["diff"]
    sp, s1m, s4m = ex_dict["sp"], ex_dict["s1m"], ex_dict["s4m"]
    # original detected tracts (grey)
    for cc, st, en in ex_dict["recomb_intervals"]:
        if cc == c and en >= lo and st <= hi:
            ax.axvspan(max(st, lo), min(en, hi), color=DET, zorder=0)
    # newly absorbed extension flanks (yellow) = extended tract minus original core,
    # and the full extended span (green outline). grey + yellow == green box.
    for _, t in extended[extended["contig"] == c].iterrows():
        if t["end_site"] < lo or t["start_site"] > hi:
            continue
        os_, oe = int(t["orig_start_site"]), int(t["orig_end_site"])
        s, e = int(t["start_site"]), int(t["end_site"])
        if os_ > s:  # left flank
            ax.axvspan(max(s, lo), min(os_, hi), color=TRACT, alpha=0.5, zorder=1)
        if e > oe:   # right flank
            ax.axvspan(max(oe, lo), min(e, hi), color=TRACT, alpha=0.5, zorder=1)
        ax.axvspan(max(s, lo), min(e, hi),
                   facecolor="none", edgecolor=EXT, lw=1.6, zorder=3)

    if mode == "all":
        # Top row: every SNV (driver), coloured by class (1D / 2D-3D / 4D).
        # Bottom row: 4D only -- what the CP-HMM originally detects on.
        other = ~s1m & ~s4m
        for mask, col in ((s4m, BLUE), (other, GREY), (s1m, RED)):
            p = sp[reg & diff & mask]
            ax.plot(p, np.ones(len(p)), "|", color=col, ms=20, mew=1.4, zorder=4)
        p4 = sp[reg & diff & s4m]
        ax.plot(p4, np.zeros(len(p4)), "|", color=BLUE, ms=20, mew=1.4, zorder=4)
        ax.set_ylim(-0.6, 1.6)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["4D\n(synon.)", "all\nSNVs"], fontsize=8)
    else:
        p1 = sp[reg & diff & s1m]
        p4 = sp[reg & diff & s4m]
        ax.plot(p1, np.ones(len(p1)), "|", color=RED, ms=20, mew=1.4, zorder=4)
        ax.plot(p4, np.zeros(len(p4)), "|", color=BLUE, ms=20, mew=1.4, zorder=4)
        ax.set_ylim(-0.6, 1.6)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["4D\n(synon.)", "1D\n(nonsyn.)"], fontsize=8)
    ax.set_xlim(lo, hi)
    ax.set_xlabel(f"reference position on {c} (bp)")
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    # Bold the data-type phrase (e.g. "QP metagenome") so the two panels' data
    # sources read as distinct; keep the pair names as plain text (underscores);
    # append the (italic) species name.
    prefix, _, rest = ex_dict["label"].partition(" pair ")
    bold = prefix.replace(" ", r"\ ")
    genus, sp = ex.SPECIES.split("_")[:2]
    species = rf"$\it{{{genus}\ {sp}}}$"
    ax.set_title(rf"$\mathbf{{{bold}}}$ pair {rest}   ({species}){driver_tag}",
                 fontsize=8.5, loc="left")


def _report(name, exd, ext, grew, ok):
    print(f"=== {name}: {exd['label']} ===")
    print(f"  missed flank: {exd['contig']}:{exd['ws']}-{exd['ws'] + WIN} "
          f"({exd['n1']} 1D / {exd['n4']} 4D)")
    if len(grew):
        g = grew.iloc[0]
        print(f"  extended tract: {g['contig']}:{int(g['start_site'])}-{int(g['end_site'])} "
              f"(orig {int(g['orig_start_site'])}-{int(g['orig_end_site'])}, "
              f"+{int(g['extension_bp'])} bp, {int(g['extension_n'])} driver-SNVs / "
              f"{int(g['extension_4d_snvs'])} 4D absorbed)")
    print(f"  flank absorbed: {'YES' if ok else 'NO'}")
    # Diagnostic: any pairwise SNVs in the zoom window but OUTSIDE the green box?
    c, ws = exd["contig"], exd["ws"]
    lo, hi = ws - 6000, ws + WIN + 6000
    reg = (exd["sc"] == c) & (exd["sp"] >= lo) & (exd["sp"] <= hi) & exd["diff"]
    inbox = np.zeros(reg.sum(), dtype=bool)
    rp = exd["sp"][reg]
    for _, t in ext[ext["contig"] == c].iterrows():
        inbox |= (rp >= t["start_site"]) & (rp <= t["end_site"])
    out1 = int(np.sum(reg & exd["s1m"] & ~np.isin(exd["sp"], rp[inbox])))
    out4 = int(np.sum(reg & exd["s4m"] & ~np.isin(exd["sp"], rp[inbox])))
    print(f"  SNVs in view window [{lo}-{hi}] outside green box: "
          f"{out1} 1D, {out4} 4D (total diffs in window: {int(reg.sum())})\n")


def build_figure(qp, iso, iso_pair, params, driver, out_path, driver_tag, mode="split"):
    qp_ext, qp_ok, qp_grew = run_example(qp, ("700023919", "ERR911954"), params, driver)
    iso_ext, iso_ok, iso_grew = run_example(iso, iso_pair, params, driver)

    print(f"---- driver = {driver} ----")
    _report("QP", qp, qp_ext, qp_grew, qp_ok)
    _report("isolate", iso, iso_ext, iso_grew, iso_ok)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(2, 1, figsize=(9, 5.0), dpi=300)
    _zoom_panel(axes[0], qp, qp_ext, driver_tag=driver_tag, mode=mode)
    _zoom_panel(axes[1], iso, iso_ext, driver_tag=driver_tag, mode=mode)
    tick_handles = [
        Line2D([0], [0], color=RED, lw=0, marker="|", ms=12, mew=2, label="1D (nonsynonymous) SNV"),
        Line2D([0], [0], color=BLUE, lw=0, marker="|", ms=12, mew=2, label="4D (synonymous) SNV"),
    ]
    if mode == "all":
        tick_handles.append(
            Line2D([0], [0], color=GREY, lw=0, marker="|", ms=12, mew=2, label="2D/3D SNV"))
    handles = tick_handles + [
        Patch(facecolor=DET, label="CP-HMM-detected core (4D)"),
        Patch(facecolor=TRACT, alpha=0.5, label="previously-missed flank"),
        Patch(facecolor="none", edgecolor=EXT, label="extended tract (grey+yellow)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=7.5,
               bbox_to_anchor=(0.5, 1.04), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext_ in ("png", "pdf"):
        fig.savefig(out_path.with_suffix(f".{ext_}"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path.with_suffix('.png')}\n")

    assert qp_ok and iso_ok, f"a missed flank was NOT absorbed (driver={driver})"


def main():
    params = ExtensionParams()
    print(f"ExtensionParams: {params}\n")

    qp = ex.qp_example()
    iso = ex.isolate_example()
    iso_pair = tuple(iso["label"].split("pair ")[1].split(" / "))

    # Narrow 1D-driven extension (the validated, paper path).
    build_figure(qp, iso, iso_pair, params, driver="1D", out_path=OUT,
                 driver_tag="  — driver: 1D only")
    # General all-SNV-driven extension (also absorbs 2D/3D-rich flanks). The
    # ticks show every SNV on one row, coloured by class, so the figure matches
    # what the driver sees (including 2D/3D sites the 1D-only view omits).
    build_figure(qp, iso, iso_pair, params, driver="all", out_path=OUT_ALL,
                 driver_tag="", mode="all")

    print("Both drivers: all missed flanks absorbed.")


if __name__ == "__main__":
    main()
