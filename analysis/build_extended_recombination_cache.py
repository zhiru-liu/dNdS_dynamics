"""Apply the CP-HMM 1D-density tract-extension to a cached recombination set.

Reads an existing isolate recombination cache (``recombination_events.parquet`` +
``recombination_pairs.parquet`` under ``<recombination-root>/<accession>/``),
extends every detected tract's boundaries into adjacent 1D-dense flanks via
``cphmm.tract_extension.extend_tracts_by_1d_density``, and writes a NEW cache with
the extended tract intervals. The pairs table is copied unchanged (extension only
widens existing tracts, so the close/clonal classification by ``event_count`` is
preserved), so the downstream dN/dS pipeline
(``compute_isolate_dnds`` -> ``compute_stratified_isolate_dnds``) consumes the new
cache with no code change and simply sees wider "recombined" regions.

Only ``reference_contig``/``reference_start``/``reference_end`` (plus the pair keys
and ``schema_version``/``dedup_representative``) are needed by
``recombination_mask_from_events``; provenance columns (``extension_bp``,
``extension_1d_snvs``, ``extension_4d_snvs``, original boundaries) are carried for
auditing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from cphmm.tract_extension import ExtensionParams, extend_tracts_by_1d_density  # noqa: E402

from dnds_dynamics.recombination.cache import CACHE_SCHEMA_VERSION  # noqa: E402
from dnds_dynamics.snv_helpers.isolate import IsolateSNVHelper  # noqa: E402


EVENT_OUT_COLUMNS = [
    "schema_version",
    "accession",
    "sample_1",
    "sample_2",
    "reference_contig",
    "reference_start",
    "reference_end",
    "dedup_representative",
    "orig_reference_start",
    "orig_reference_end",
    "extension_bp",
    "extension_1d_snvs",
    "extension_4d_snvs",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("accession", help='Cache accession, e.g. "snv_table".')
    parser.add_argument("--table-root", type=Path, required=True,
                        help="SNV table root (parent of <accession>).")
    parser.add_argument("--recombination-root", type=Path, required=True,
                        help="Existing cache root (parent of <accession>) to extend.")
    parser.add_argument("--output-root", type=Path, required=True,
                        help="Output cache root; writes <output-root>/<accession>/recombination_*.parquet.")
    parser.add_argument("--gap-bp", type=int, default=ExtensionParams.gap_bp)
    parser.add_argument("--min-abs-per-kb", type=float, default=ExtensionParams.min_abs_per_kb)
    parser.add_argument("--rate-multiple", type=float, default=ExtensionParams.rate_multiple)
    parser.add_argument("--max-extension-bp", type=int, default=ExtensionParams.max_extension_bp)
    return parser.parse_args()


def pair_snp_infos(helper: IsolateSNVHelper, s1: str, s2: str):
    """Return (snp_info_1d, snp_info_4d) for a pair in one coverage/diff scan."""
    diffs = helper.compute_pairwise_snvs(s1, s2).to_numpy()
    covered = helper.compute_pairwise_coverage(s1, s2).to_numpy()
    diff_core = np.zeros(helper.num_sites, dtype=bool)
    diff_core[helper.core_to_snvs] = diffs
    chrom = helper.chromosomes.astype(str)
    loc = helper.locations.astype(int)
    m1 = covered & helper.core_1D.to_numpy()
    m4 = covered & helper.core_4D.to_numpy()
    si1 = (diff_core[m1], chrom[m1], loc[m1])
    si4 = (diff_core[m4], chrom[m4], loc[m4])
    return si1, si4


def main() -> None:
    args = parse_args()
    params = ExtensionParams(
        gap_bp=args.gap_bp,
        min_abs_per_kb=args.min_abs_per_kb,
        rate_multiple=args.rate_multiple,
        max_extension_bp=args.max_extension_bp,
    )

    helper = IsolateSNVHelper(
        args.accession,
        table_root=args.table_root,
        recombination_root=args.recombination_root,
        source="tables",
        compute_bi_snvs=False,
        annotate=True,
    )

    events = helper.recombination_events
    pairs = helper.recombination_pairs
    event_pairs = pairs[pairs["event_count"] > 0]

    out_rows: list[pd.DataFrame] = []
    n_pairs = len(event_pairs)
    tot_ext_bp = tot_moved_1d = tot_moved_4d = 0
    for i, pr in enumerate(event_pairs.itertuples(index=False), 1):
        s1, s2 = str(pr.sample_1), str(pr.sample_2)
        pair_events = events[(events["sample_1"].astype(str) == s1)
                             & (events["sample_2"].astype(str) == s2)]
        if pair_events.empty:
            continue
        transfer_df = pd.DataFrame({
            "genome1": s1,
            "genome2": s2,
            "contig": pair_events["reference_contig"].astype(str).to_numpy(),
            "start_site": pair_events["reference_start"].astype(int).to_numpy(),
            "end_site": pair_events["reference_end"].astype(int).to_numpy(),
            "types": 0,
        })
        si1, si4 = pair_snp_infos(helper, s1, s2)
        ext = extend_tracts_by_1d_density(transfer_df, si1, params=params, snp_info_4d=si4)
        if ext.empty:
            continue
        out = pd.DataFrame({
            "schema_version": CACHE_SCHEMA_VERSION,
            "accession": args.accession,
            "sample_1": s1,
            "sample_2": s2,
            "reference_contig": ext["contig"].astype(str).to_numpy(),
            "reference_start": ext["start_site"].astype(int).to_numpy(),
            "reference_end": ext["end_site"].astype(int).to_numpy(),
            "dedup_representative": True,
            "orig_reference_start": ext["orig_start_site"].astype(int).to_numpy(),
            "orig_reference_end": ext["orig_end_site"].astype(int).to_numpy(),
            "extension_bp": ext["extension_bp"].astype(int).to_numpy(),
            "extension_1d_snvs": ext["extension_1d_snvs"].astype(int).to_numpy(),
            "extension_4d_snvs": ext["extension_4d_snvs"].astype(int).to_numpy(),
        }, columns=EVENT_OUT_COLUMNS)
        out_rows.append(out)
        tot_ext_bp += int(out["extension_bp"].sum())
        tot_moved_1d += int(out["extension_1d_snvs"].sum())
        tot_moved_4d += int(out["extension_4d_snvs"].sum())
        if i % 50 == 0 or i == n_pairs:
            print(f"[{i}/{n_pairs}] pairs extended; "
                  f"cum ext_bp={tot_ext_bp} moved_1D={tot_moved_1d} moved_4D={tot_moved_4d}",
                  flush=True)

    extended_events = (
        pd.concat(out_rows, ignore_index=True)
        if out_rows else pd.DataFrame(columns=EVENT_OUT_COLUMNS)
    )

    out_dir = args.output_root / args.accession
    out_dir.mkdir(parents=True, exist_ok=True)
    events_path = out_dir / "recombination_events.parquet"
    pairs_path = out_dir / "recombination_pairs.parquet"
    meta_path = out_dir / "recombination_cache_metadata.json"
    extended_events.to_parquet(events_path, index=False)
    pairs.to_parquet(pairs_path, index=False)

    metadata = {
        "accession": args.accession,
        "source": "tract_extension_1d",
        "extends_cache": str(args.recombination_root / args.accession),
        "table_root": str(args.table_root),
        "extension_params": {
            "gap_bp": params.gap_bp,
            "min_abs_per_kb": params.min_abs_per_kb,
            "rate_multiple": params.rate_multiple,
            "max_extension_bp": params.max_extension_bp,
        },
        "event_bearing_pairs": int(n_pairs),
        "event_rows": int(len(extended_events)),
        "total_extension_bp": int(tot_ext_bp),
        "total_moved_1d_snvs": int(tot_moved_1d),
        "total_moved_4d_snvs": int(tot_moved_4d),
        "events_path": str(events_path),
        "pairs_path": str(pairs_path),
    }
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
