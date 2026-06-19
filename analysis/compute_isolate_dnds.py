"""Compute isolate dN/dS summaries for close, clonal, and typical pair classes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dnds_dynamics.dnds.isolate import (  # noqa: E402
    DEFAULT_IDENTICAL_FRACTION_ROOT,
    DEFAULT_MIN_CONTIG_4D_SITES,
    compute_isolate_dnds,
)
from dnds_dynamics.snv_helpers.isolate import DEFAULT_SNV_TABLE_ROOT  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("accessions", nargs="+")
    parser.add_argument("--table-root", type=Path, default=DEFAULT_SNV_TABLE_ROOT)
    parser.add_argument(
        "--recombination-root",
        type=Path,
        help="Defaults to --table-root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Defaults to <table-root>/<accession>/dnds.",
    )
    parser.add_argument("--identical-fraction-root", type=Path, default=DEFAULT_IDENTICAL_FRACTION_ROOT)
    parser.add_argument("--typical-threshold", type=float, default=0.05)
    parser.add_argument("--typical-pairs", type=int, default=20)
    parser.add_argument("--exclude-samples", nargs="*", default=None,
                        help="Sample names to drop from typical-pair sampling "
                             "(e.g. a misclassified genome).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--min-contig-4D-sites",
        type=int,
        default=None,
        help=(
            "If set, skip per-pair contigs with fewer than this many covered 4D "
            "sites, matching the CP-HMM contig filter "
            f"(HMM_BLOCK_SIZE * HMM_MIN_SEQ_LEN = {DEFAULT_MIN_CONTIG_4D_SITES}). "
            "Default: no filter (legacy behavior)."
        ),
    )
    parser.add_argument(
        "--match-cphmm-contig-filter",
        action="store_true",
        help=(
            f"Shortcut for --min-contig-4D-sites={DEFAULT_MIN_CONTIG_4D_SITES}."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    min_contig_4D_sites = args.min_contig_4D_sites
    if args.match_cphmm_contig_filter and min_contig_4D_sites is None:
        min_contig_4D_sites = DEFAULT_MIN_CONTIG_4D_SITES
    summaries = []
    for accession in args.accessions:
        result = compute_isolate_dnds(
            accession,
            table_root=args.table_root,
            recombination_root=args.recombination_root,
            output_root=args.output_root,
            identical_fraction_root=args.identical_fraction_root,
            typical_threshold=args.typical_threshold,
            typical_pairs=args.typical_pairs,
            seed=args.seed,
            min_contig_4D_sites=min_contig_4D_sites,
            exclude_samples=set(args.exclude_samples) if args.exclude_samples else None,
        )
        summaries.append(result.summary)
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
