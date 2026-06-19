"""Redo the missed-recombination mask test using the CP-HMM tract-EXTENSION feature
(cphmm.tract_extension) instead of the ad-hoc 1D-dense-window mask.

For every QP close pair: take the published recombination tracts, extend their
boundaries into adjacent 1D-dense flanks via
`extend_tracts_by_1d_density`, and count the clonal 1D/4D differences that move
from "clonal" into the extended tracts (the `extension_*_snvs` provenance). Clonal
dN/dS "after" = published clonal counts minus the moved differences (denominator
held at published clonal_len; flank length change is negligible — same convention
as the ad-hoc test, so the two are directly comparable).

Outputs:
  outputs/extended_recomb/per_pair_extended_counts.csv   (per pair; reused by the test script)
  outputs/extended_recomb/across_species_extended.csv     (per-species before/after)
  figures/qp_missed_recomb_across_species.{png,pdf}        (regenerated: after = extended)
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO / ".cache" / "matplotlib"))
sys.path.insert(0, str(REPO))
from cphmm.tract_extension import extend_tracts_by_1d_density  # noqa: E402
from dnds_dynamics.snv_helpers.qp import load_qp_snv_helper  # noqa: E402

FEATHER = Path("/Volumes/Botein/GarudGood2019_snvs/snvs_feather")
DN = REPO / "data"
CLOSE_DIR = DN / "gut_microbiome_close_pair_dNdS"
TRANSFERS = DN / "gut_microbiome_transfers.csv"
BLACKLIST = {"Lachnospiraceae_bacterium_51870"}
OUT = REPO / "outputs" / "extended_recomb"
FIG = REPO / "figures" / "qp_missed_recomb_across_species"
# Only count a tract's extension if it absorbs at least this many 1D SNVs.
# (Lone-singleton extensions are dropped when this is >1; default 1 = keep all.)
MIN_EXT_1D = 1


def eligible_species():
    out = []
    for f in sorted(CLOSE_DIR.glob("*.csv")):
        sp = f.stem
        if sp in BLACKLIST or not (FEATHER / sp).exists():
            continue
        if sum(1 for _ in open(f)) - 1 > 0:
            out.append(sp)
    return out


def load_helper(sp):
    return load_qp_snv_helper(sp, compute_bi_snvs=False, annotate=True)


def transfers_by_pair(sp):
    tr = pd.read_csv(TRANSFERS, low_memory=False)
    tr = tr[tr["Species name"] == sp].dropna(
        subset=["Reference contig", "Reference genome start loc", "Reference genome end loc"])
    out = {}
    for (a, b), g in tr.groupby(["Sample 1", "Sample 2"]):
        out[(str(a), str(b))] = g[["Reference contig", "Reference genome start loc",
                                   "Reference genome end loc"]].to_numpy()
    return out


def species_rows(sp):
    qp = load_helper(sp)
    idx = qp.coverage.index
    ca = idx.get_level_values("Contig").to_numpy().astype(str)
    la = idx.get_level_values("Location").to_numpy().astype(int)
    c1 = np.asarray(qp.core_1D); c4 = np.asarray(qp.core_4D)
    cts = np.asarray(qp.core_to_snvs); n = len(idx)
    ca1, la1, ca4, la4 = ca[c1], la[c1], ca[c4], la[c4]
    trp = transfers_by_pair(sp)
    close = pd.read_csv(CLOSE_DIR / f"{sp}.csv")
    rows = []
    for s1, s2, cd1, cl1, cd4, cl4 in zip(
            close["sample 1"].astype(str), close["sample 2"].astype(str),
            close["clonal_diff_1D"], close["clonal_len_1D"],
            close["clonal_diff_4D"], close["clonal_len_4D"]):
        moved1 = moved4 = 0; ext_bp = 0
        iv = trp.get((s1, s2), trp.get((s2, s1), []))
        if len(iv):
            try:
                diff = qp.compute_pairwise_snvs(s1, s2).to_numpy()
            except KeyError:
                diff = None
            if diff is not None:
                df = np.zeros(n, bool); df[cts] = diff
                si1 = (df[c1], ca1, la1); si4 = (df[c4], ca4, la4)
                tdf = pd.DataFrame({"contig": [str(x[0]) for x in iv],
                                    "start_site": [int(x[1]) for x in iv],
                                    "end_site": [int(x[2]) for x in iv], "types": 0})
                ext = extend_tracts_by_1d_density(tdf, si1, snp_info_4d=si4)
                keep = ext["extension_1d_snvs"] >= MIN_EXT_1D  # drop sub-threshold extensions
                moved1 = int(ext.loc[keep, "extension_1d_snvs"].sum())
                moved4 = int(ext.loc[keep, "extension_4d_snvs"].sum())
                ext_bp = int(ext.loc[keep, "extension_bp"].sum())
        rows.append({"species": sp, "s1": s1, "s2": s2,
                     "clonal_diff_1D": cd1, "clonal_len_1D": cl1,
                     "clonal_diff_4D": cd4, "clonal_len_4D": cl4,
                     "moved_1d": moved1, "moved_4d": moved4, "ext_bp": ext_bp})
    del qp
    return rows


def aggregate(df):
    g = df.groupby("species")
    out = []
    for sp, s in g:
        l1, l4 = s.clonal_len_1D.sum(), s.clonal_len_4D.sum()
        b1, b4 = s.clonal_diff_1D.sum(), s.clonal_diff_4D.sum()
        a1, a4 = b1 - s.moved_1d.sum(), b4 - s.moved_4d.sum()
        before = (b1 / l1) / (b4 / l4) if b4 and l4 else np.nan
        after = (a1 / l1) / (a4 / l4) if a4 and l4 else np.nan
        out.append({"species": sp, "n_pairs": len(s),
                    "clonal_dnds_before": before, "clonal_dnds_after_extended": after,
                    "frac_1D_moved": s.moved_1d.sum() / b1 if b1 else np.nan,
                    "moved_1d": int(s.moved_1d.sum()), "moved_4d": int(s.moved_4d.sum())})
    return pd.DataFrame(out)


def plot(agg):
    import matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    d = agg.dropna(subset=["clonal_dnds_before"]).sort_values("clonal_dnds_before")
    y = np.arange(len(d))
    fig, axes = plt.subplots(1, 2, figsize=(12, max(5, 0.32 * len(d))), dpi=300,
                             gridspec_kw={"width_ratios": [2, 1]})
    ax = axes[0]
    ax.hlines(y, d["clonal_dnds_after_extended"], d["clonal_dnds_before"], color="0.7", zorder=1)
    ax.scatter(d["clonal_dnds_before"], y, s=30, color="#2b6cb0", label="before (detected tracts)", zorder=3)
    ax.scatter(d["clonal_dnds_after_extended"], y, s=30, color="#c0392b", label="after (extended tracts)", zorder=3)
    ax.axvline(1, ls="--", lw=0.8, color="0.5")
    ax.set_yticks(y); ax.set_yticklabels([s.replace("_", " ").rsplit(" ", 1)[0] for s in d["species"]], fontsize=7)
    ax.set_xlabel("aggregate clonal $dN/dS$")
    ax.set_title("Clonal dN/dS before vs after 1D-density tract extension")
    ax.legend(fontsize=8, loc="lower right")
    ax = axes[1]
    ax.scatter(d["frac_1D_moved"], y, s=30, color="#7b3294")
    ax.set_yticks(y); ax.set_yticklabels([])
    ax.set_xlabel("frac clonal 1D moved\ninto extended tracts")
    ax.set_title("Extension burden"); ax.set_xlim(0, 1)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG.with_suffix(f".{ext}"), bbox_inches="tight")
    plt.close(fig)


def main():
    import argparse
    global MIN_EXT_1D, FIG
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-ext-1d", type=int, default=MIN_EXT_1D,
                    help="min 1D SNVs an extension must absorb to count (default 1=keep all)")
    ap.add_argument("--suffix", default=None,
                    help="output suffix; default '_min{N}' when threshold>1, else ''")
    args = ap.parse_args()
    MIN_EXT_1D = args.min_ext_1d
    suffix = args.suffix if args.suffix is not None else (f"_min{MIN_EXT_1D}" if MIN_EXT_1D > 1 else "")
    FIG = FIG.with_name(FIG.name + suffix)
    print(f"MIN_EXT_1D={MIN_EXT_1D}  suffix='{suffix}'")

    OUT.mkdir(parents=True, exist_ok=True)
    pp = OUT / f"per_pair_extended_counts{suffix}.csv"
    species = eligible_species()
    # Resume: keep any species already cached in pp, recompute only the rest.
    rows = []
    done = set()
    if pp.exists():
        prev = pd.read_csv(pp)
        done = set(prev["species"].unique())
        rows = prev.to_dict("records")
        print(f"resume: {len(done)} species already cached ({len(prev)} pairs)")
    todo = [sp for sp in species if sp not in done]
    print(f"{len(species)} species total, {len(todo)} to compute")
    for i, sp in enumerate(todo, 1):
        rows.extend(species_rows(sp))
        pd.DataFrame(rows).to_csv(pp, index=False)
        sub = aggregate(pd.DataFrame(rows))
        r = sub[sub.species == sp].iloc[0]
        print(f"[{i}/{len(todo)}] {sp:35s} dN/dS {r.clonal_dnds_before:.2f} -> "
              f"{r.clonal_dnds_after_extended:.2f}  (frac 1D moved {r.frac_1D_moved:.2f})", flush=True)
    df = pd.DataFrame(rows)
    agg = aggregate(df)
    asp = OUT / f"across_species_extended{suffix}.csv"
    agg.to_csv(asp, index=False)
    plot(agg)
    print(f"\nwrote {pp}\nwrote {asp}\nwrote {FIG.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
