"""Example missed-recombination events for the reviewer response.

Two A. putredinis pairs where a recombination tract sits in the "clonal" region,
invisible to a 4D-density-based CP-HMM because it is nonsynonymous(1D)-rich but
synonymous(4D)-poor:

  (A) QP metagenome close pair (LiuGood2024 cohort)
  (B) cultured NCBI-isolate diverged pair (rules out a QP mapping artifact)

Figure 1 (zoom): ALL pairwise SNVs as ticks (red=1D nonsyn, blue=4D syn).
  CP-HMM-detected tracts shaded grey (note: dense in BOTH 1D and 4D -> detected);
  the missed tract shaded yellow (dense in 1D, ~no 4D -> undetected).
Figure 2 (genome-wide): per-pair local SNV density along the whole genome, 1D vs
  4D, showing the sparse clonal background, the SNV-dense detected tracts (both
  classes), and the 1D-only missed tract.
"""
from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
MPL_CACHE_DIR = REPO_ROOT / ".cache" / "matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

from dnds_dynamics.snv_helpers.isolate import IsolateSNVHelper  # noqa: E402

LIUGOOD_PKG = Path("/Users/Device6/Documents/Research/bgoodlab/LiuGood2024_data")
SNV_FEATHER = Path("/Volumes/Botein/GarudGood2019_snvs/snvs_feather")
DN = Path("/Users/Device6/Documents/Research/bgoodlab/dNdS/dNdS_dynamics/data")
SPECIES = "Alistipes_putredinis_61533"
ISO_ROOT = Path("/Volumes/Botein/ncbi_isolates/Alistipes_putredinis")
FIG_ZOOM = REPO_ROOT / "figures" / "response_missed_recomb_examples"
FIG_GW = REPO_ROOT / "figures" / "response_missed_recomb_genomewide"
WIN = 1000
RED = "#c0392b"; BLUE = "#2b6cb0"; TRACT = "#f6c026"; DET = "0.82"


def load_qp():
    cwd = os.getcwd()
    try:
        os.chdir(LIUGOOD_PKG)
        if str(LIUGOOD_PKG) not in sys.path:
            sys.path.insert(0, str(LIUGOOD_PKG))
        from snv_utils import SNVHelper
    finally:
        os.chdir(cwd)
    return SNVHelper(SPECIES, snv_path=SNV_FEATHER, snv_format="feather",
                     compute_bi_snvs=False, annotate=True, mask_multi_sites=True)


def _intervals_from_mask(sc, sp, mask, gap=2000):
    out = []
    for c in np.unique(sc[mask]):
        p = np.sort(sp[mask & (sc == c)])
        if len(p) == 0:
            continue
        start = prev = p[0]
        for x in p[1:]:
            if x - prev > gap:
                out.append((c, int(start), int(prev))); start = x
            prev = x
        out.append((c, int(start), int(prev)))
    return out


def best_clonal_window(sc, sp, clonal, s1m, s4m, min1d=12, max4d=1):
    win1 = Counter(); win4 = Counter()
    idx = np.flatnonzero(clonal)
    for i in idx:
        key = (sc[i], sp[i] // WIN)
        if s1m[i]:
            win1[key] += 1
        if s4m[i]:
            win4[key] += 1
    best = None
    for key, n1 in win1.items():
        n4 = win4.get(key, 0)
        if n1 >= min1d and n4 <= max4d and (best is None or n1 > best[2]):
            best = (key[0], key[1] * WIN, n1, n4)
    return best


def qp_example():
    qp = load_qp()
    snv = qp.snvs.index
    sc = snv.get_level_values("Contig").to_numpy().astype(str)
    sp = snv.get_level_values("Location").to_numpy().astype(int)
    s1m = qp.snv_1D.to_numpy(); s4m = qp.snv_4D.to_numpy()
    per = {}
    for c in np.unique(sc):
        gi = np.flatnonzero(sc == c); o = np.argsort(sp[gi]); per[c] = (sp[gi][o], gi[o])
    tr = pd.read_csv(DN / "gut_microbiome_transfers.csv", low_memory=False)
    tr = tr[tr["Species name"] == SPECIES].dropna(
        subset=["Reference contig", "Reference genome start loc", "Reference genome end loc"])
    trp = {}
    for (a, b), g in tr.groupby(["Sample 1", "Sample 2"]):
        trp[(str(a), str(b))] = g[["Reference contig", "Reference genome start loc",
                                   "Reference genome end loc"]].to_numpy()

    def mask_for(s1n, s2n):
        diff = qp.compute_pairwise_snvs(s1n, s2n).to_numpy()
        recomb = np.zeros(len(snv), bool)
        for c, st, en in trp.get((s1n, s2n), trp.get((s2n, s1n), [])):
            arr = per.get(str(c))
            if arr is None:
                continue
            ps, gi = arr
            lo = np.searchsorted(ps, st, "left"); hi = np.searchsorted(ps, en, "right")
            if hi > lo:
                recomb[gi[lo:hi]] = True
        return diff, recomb

    # use the strong known example
    s1n, s2n = "700023919", "ERR911954"
    diff, recomb = mask_for(s1n, s2n)
    bw = best_clonal_window(sc, sp, diff & ~recomb, s1m, s4m, min1d=12, max4d=1)
    c, ws, n1, n4 = bw
    print(f"QP example: {s1n}/{s2n}  {c}:{ws}-{ws+WIN}  {n1} 1D, {n4} 4D (clonal tract)")
    return dict(label=f"QP metagenome pair {s1n} / {s2n}", contig=c, ws=ws, n1=n1, n4=n4,
                sc=sc, sp=sp, diff=diff, recomb=recomb, s1m=s1m, s4m=s4m,
                recomb_intervals=_intervals_from_mask(sc, sp, recomb))


def isolate_example():
    iso = IsolateSNVHelper("snv_table", table_root=ISO_ROOT, source="tables",
                           annotate=True, compute_bi_snvs=False, mask_multi_sites=True)
    snv = iso.snv_index
    sc = snv.get_level_values("Contig").to_numpy().astype(str)
    sp = snv.get_level_values("Location").to_numpy().astype(int)
    s1m = iso.snv_1D.to_numpy(); s4m = iso.snv_4D.to_numpy()
    close = pd.read_csv(ISO_ROOT / "snv_table" / "dnds" / "close_pairs_with_recombination.csv")
    close["full_dS"] = close["core_diff_4D"] / close["core_len_4D"]
    div = close[(close.full_dS >= 3e-3) & (close.full_dS <= 1e-2)]

    best = None
    for s1n, s2n in div[["sample 1", "sample 2"]].astype(str).itertuples(index=False):
        diff = iso.compute_pairwise_snvs(s1n, s2n).to_numpy()
        recomb = iso.compute_pair_recombination_mask(s1n, s2n).reindex(snv, fill_value=False).to_numpy()
        bw = best_clonal_window(sc, sp, diff & ~recomb, s1m, s4m, min1d=8, max4d=1)
        if bw is not None and (best is None or bw[2] > best[1][2]):
            best = ((s1n, s2n), bw, diff, recomb)
            if bw[2] >= 15:
                break
    (s1n, s2n), (c, ws, n1, n4), diff, recomb = best
    print(f"isolate example: {s1n}/{s2n}  {c}:{ws}-{ws+WIN}  {n1} 1D, {n4} 4D (clonal tract)")
    return dict(label=f"Cultured isolate pair {s1n} / {s2n}", contig=c, ws=ws, n1=n1, n4=n4,
                sc=sc, sp=sp, diff=diff, recomb=recomb, s1m=s1m, s4m=s4m,
                recomb_intervals=_intervals_from_mask(sc, sp, recomb))


# --------------------------------------------------------------------------- #
def _zoom_panel(ax, ex, halfspan=6000):
    c = ex["contig"]; ws = ex["ws"]
    lo = ws - halfspan; hi = ws + WIN + halfspan
    reg = (ex["sc"] == c) & (ex["sp"] >= lo) & (ex["sp"] <= hi)
    diff = ex["diff"]
    p1 = ex["sp"][reg & diff & ex["s1m"]]
    p4 = ex["sp"][reg & diff & ex["s4m"]]
    for cc, st, en in ex["recomb_intervals"]:
        if cc == c and en >= lo and st <= hi:
            ax.axvspan(max(st, lo), min(en, hi), color=DET, zorder=0)
    ax.axvspan(ws, ws + WIN, color=TRACT, alpha=0.5, zorder=1)
    ax.plot(p1, np.ones(len(p1)), "|", color=RED, ms=20, mew=1.4, zorder=4)
    ax.plot(p4, np.zeros(len(p4)), "|", color=BLUE, ms=20, mew=1.4, zorder=4)
    ax.set_ylim(-0.6, 1.6); ax.set_yticks([0, 1])
    ax.set_yticklabels(["4D\n(synon.)", "1D\n(nonsyn.)"], fontsize=8)
    ax.set_xlim(lo, hi); ax.set_xlabel(f"reference position on {c} (bp)")
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title(f"{ex['label']} — missed tract (yellow): {ex['n1']} nonsyn vs {ex['n4']} syn "
                 f"in 1 kb; detected tracts (grey) are dense in BOTH classes",
                 fontsize=8.5, loc="left")


def _genome_panel(ax, ex, win=2000):
    sc = ex["sc"]; sp = ex["sp"]; diff = ex["diff"]
    contigs = list(dict.fromkeys(sc))
    off = {}; cum = 0
    for c in contigs:
        off[c] = cum; cum += sp[sc == c].max() + 20000
    gx_all = sp + np.array([off[c] for c in sc])
    edges = np.arange(0, cum + win, win)
    h1, _ = np.histogram(gx_all[diff & ex["s1m"]], bins=edges)
    h4, _ = np.histogram(gx_all[diff & ex["s4m"]], bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2
    # shade detected recombination
    for c, st, en in ex["recomb_intervals"]:
        ax.axvspan(off[c] + st, off[c] + en, color=DET, zorder=0)
    ax.fill_between(centers, 0, h1, step="mid", color=RED, alpha=0.85, lw=0, label="1D (nonsyn) SNVs / 2 kb")
    ax.fill_between(centers, 0, -h4, step="mid", color=BLUE, alpha=0.85, lw=0, label="4D (syn) SNVs / 2 kb")
    # mark the missed tract
    mx = off[ex["contig"]] + ex["ws"] + WIN / 2
    ymax = max(h1.max(), h4.max(), 1)
    ax.annotate("missed\ntract", xy=(mx, h1.max() if h1.size else 1), xytext=(mx, ymax * 1.15),
                ha="center", fontsize=7, color="#9a6a00",
                arrowprops=dict(arrowstyle="->", color="#9a6a00", lw=1.0))
    ax.axhline(0, color="0.5", lw=0.6)
    ax.set_xlim(0, cum); ax.set_ylim(-ymax * 1.25, ymax * 1.3)
    ax.set_ylabel("SNVs per 2 kb\n(1D up / 4D down)", fontsize=8)
    ax.set_xlabel("position along concatenated core genome (bp)")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.set_title(ex["label"], fontsize=9, loc="left")
    ax.legend(fontsize=7, loc="upper right", ncol=2, framealpha=0.9)


def main():
    qp = qp_example()
    iso = isolate_example()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    # ---- Figure 1: zoom (all SNVs) ----
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.0), dpi=300)
    _zoom_panel(axes[0], qp)
    _zoom_panel(axes[1], iso)
    handles = [
        Line2D([0], [0], color=RED, lw=0, marker="|", ms=12, mew=2, label="1D (nonsynonymous) SNV"),
        Line2D([0], [0], color=BLUE, lw=0, marker="|", ms=12, mew=2, label="4D (synonymous) SNV"),
        Patch(facecolor=TRACT, alpha=0.5, label="missed recombination tract (clonal region)"),
        Patch(facecolor=DET, label="CP-HMM-detected recombination"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=7.5,
               bbox_to_anchor=(0.5, 1.03), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(FIG_ZOOM.with_suffix(f".{ext}"), bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 2: genome-wide density ----
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.0), dpi=300)
    _genome_panel(axes[0], qp)
    _genome_panel(axes[1], iso)
    fig.suptitle("Genome-wide SNV density: clonal background is sparse; detected tracts are dense in "
                 "1D+4D; missed tracts are 1D-only", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for ext in ("png", "pdf"):
        fig.savefig(FIG_GW.with_suffix(f".{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {FIG_ZOOM.with_suffix('.png')}")
    print(f"wrote {FIG_GW.with_suffix('.png')}")


if __name__ == "__main__":
    main()
