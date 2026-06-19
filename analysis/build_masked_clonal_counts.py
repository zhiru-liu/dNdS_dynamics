"""Per-pair clonal dN/dS counts + dense-window (missed-recombination) removals.

For every close pair (with recombination) and clonal pair (no recombination) in
the QP publication cohort, record the published clonal-region counts plus the
number of clonal 1D/4D SNVs that fall in that pair's within-pair 1D-dense windows
(>=3 clonal 1D SNVs in a 1 kb window) -- the putative missed-recombination tracts
invisible to the 4D-only CP-HMM.

Output (one row per pair):
  outputs/masked_clonal_fit/per_pair_masked_clonal_counts.csv
  columns: species, s1, s2, pair_class,
           clonal_diff_1D, clonal_len_1D, clonal_diff_4D, clonal_len_4D  (published),
           rm1, rm4                                  (1D/4D SNVs in 1D-dense windows),
           recomputed_diff_1D, recomputed_diff_4D    (from transfer masking; validation)

Masked counts are then clonal_diff_* - rm* (computed downstream).
"""
from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
LIUGOOD_PKG = Path("/Users/Device6/Documents/Research/bgoodlab/LiuGood2024_data")
SNV_FEATHER = Path("/Volumes/Botein/GarudGood2019_snvs/snvs_feather")
DN = Path("/Users/Device6/Documents/Research/bgoodlab/dNdS/dNdS_dynamics/data")
CLOSE_DIR = DN / "gut_microbiome_close_pair_dNdS"
CLONAL_DIR = DN / "gut_microbiome_clonal_pair_dNdS"
TRANSFERS = DN / "gut_microbiome_transfers.csv"
BLACKLIST = {"Lachnospiraceae_bacterium_51870"}
OUT_DIR = REPO_ROOT / "outputs" / "masked_clonal_fit"
WIN = 1000
DENSE_1D = 3


def eligible_species() -> list[str]:
    out = []
    for f in sorted(CLOSE_DIR.glob("*.csv")):
        sp = f.stem
        if sp in BLACKLIST or not (SNV_FEATHER / sp).exists():
            continue
        if sum(1 for _ in open(f)) - 1 > 0:
            out.append(sp)
    return out


def load_helper(species: str):
    cwd = os.getcwd()
    try:
        os.chdir(LIUGOOD_PKG)
        if str(LIUGOOD_PKG) not in sys.path:
            sys.path.insert(0, str(LIUGOOD_PKG))
        from snv_utils import SNVHelper
    finally:
        os.chdir(cwd)
    return SNVHelper(species, snv_path=SNV_FEATHER, snv_format="feather",
                     compute_bi_snvs=False, annotate=True, mask_multi_sites=True)


def transfers_by_pair(species: str) -> dict:
    tr = pd.read_csv(TRANSFERS, low_memory=False)
    tr = tr[tr["Species name"] == species].dropna(
        subset=["Reference contig", "Reference genome start loc", "Reference genome end loc"])
    out = {}
    for (a, b), g in tr.groupby(["Sample 1", "Sample 2"]):
        out[(str(a), str(b))] = g[["Reference contig", "Reference genome start loc",
                                   "Reference genome end loc"]].to_numpy()
    return out


def rm_for_pair(helper, sc, sp, s1mask, s4mask, per, recomb_intervals, s1n, s2n):
    """Return (rm1, rm4, recomputed_diff1, recomputed_diff4) for one pair."""
    try:
        diff = helper.compute_pairwise_snvs(s1n, s2n).to_numpy()
    except KeyError:
        return None
    n_snv = len(diff)
    recomb = np.zeros(n_snv, bool)
    for c, st, en in recomb_intervals:
        arr = per.get(str(c))
        if arr is None:
            continue
        ps, gi = arr
        lo = np.searchsorted(ps, st, "left"); hi = np.searchsorted(ps, en, "right")
        if hi > lo:
            recomb[gi[lo:hi]] = True
    clonal = diff & ~recomb
    c1 = clonal & s1mask; c4 = clonal & s4mask
    rd1 = int(c1.sum()); rd4 = int(c4.sum())
    win1 = Counter(); win4 = Counter()
    for c in np.unique(sc[clonal]):
        for p in sp[c1 & (sc == c)]:
            win1[(c, p // WIN)] += 1
        for p in sp[c4 & (sc == c)]:
            win4[(c, p // WIN)] += 1
    dense = {w for w, n in win1.items() if n >= DENSE_1D}
    rm1 = sum(win1[w] for w in dense)
    rm4 = sum(win4.get(w, 0) for w in dense)
    return rm1, rm4, rd1, rd4


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / "per_pair_masked_clonal_counts.csv"
    species = eligible_species()
    print(f"{len(species)} species")
    rows = []
    for i, sp in enumerate(species, 1):
        helper = load_helper(sp)
        snv_idx = helper.snvs.index
        sc = snv_idx.get_level_values("Contig").to_numpy().astype(str)
        spos = snv_idx.get_level_values("Location").to_numpy().astype(int)
        s1mask = helper.snv_1D.to_numpy(); s4mask = helper.snv_4D.to_numpy()
        per = {}
        for c in np.unique(sc):
            gi = np.flatnonzero(sc == c); o = np.argsort(spos[gi]); per[c] = (spos[gi][o], gi[o])
        trp = transfers_by_pair(sp)

        # close pairs (with recombination)
        close = pd.read_csv(CLOSE_DIR / f"{sp}.csv")
        for s1n, s2n, cd1, cl1, cd4, cl4 in zip(
                close["sample 1"].astype(str), close["sample 2"].astype(str),
                close["clonal_diff_1D"], close["clonal_len_1D"],
                close["clonal_diff_4D"], close["clonal_len_4D"]):
            iv = trp.get((s1n, s2n), trp.get((s2n, s1n), []))
            res = rm_for_pair(helper, sc, spos, s1mask, s4mask, per, iv, s1n, s2n)
            if res is None:
                continue
            rm1, rm4, rd1, rd4 = res
            rows.append({"species": sp, "s1": s1n, "s2": s2n, "pair_class": "close",
                         "clonal_diff_1D": cd1, "clonal_len_1D": cl1,
                         "clonal_diff_4D": cd4, "clonal_len_4D": cl4,
                         "rm1": rm1, "rm4": rm4, "recomputed_diff_1D": rd1, "recomputed_diff_4D": rd4})

        # clonal pairs (no recombination); core_* == clonal region
        cf = CLONAL_DIR / f"{sp}.csv"
        if cf.exists():
            clon = pd.read_csv(cf)
            for s1n, s2n, cd1, cl1, cd4, cl4 in zip(
                    clon["sample 1"].astype(str), clon["sample 2"].astype(str),
                    clon["core_diff_1D"], clon["core_len_1D"],
                    clon["core_diff_4D"], clon["core_len_4D"]):
                res = rm_for_pair(helper, sc, spos, s1mask, s4mask, per, [], s1n, s2n)
                if res is None:
                    continue
                rm1, rm4, rd1, rd4 = res
                rows.append({"species": sp, "s1": s1n, "s2": s2n, "pair_class": "clonal",
                             "clonal_diff_1D": cd1, "clonal_len_1D": cl1,
                             "clonal_diff_4D": cd4, "clonal_len_4D": cl4,
                             "rm1": rm1, "rm4": rm4, "recomputed_diff_1D": rd1, "recomputed_diff_4D": rd4})

        pd.DataFrame(rows).to_csv(out_csv, index=False)
        n_sp = sum(1 for x in rows if x["species"] == sp)
        print(f"[{i}/{len(species)}] {sp:35s} pairs={n_sp}", flush=True)
        del helper

    print(f"\nwrote {out_csv} ({len(rows)} pairs)")


if __name__ == "__main__":
    main()
