"""Stratify isolate dN/dS outputs into core / recombined / clonal tables and plots.

Run this after ``compute_isolate_dnds.py`` for the same accession. Reads
``close_pairs_with_recombination.csv`` and ``clonal_pairs_no_recombination.csv``
under ``<table-root>/<accession>/dnds/`` and writes ``stratified_dnds_points.csv``,
``stratified_dnds_summary.csv``, ``stratified_dnds_metadata.json``, and
``stratified_dnds_separation.{png,pdf}`` (by default alongside the inputs).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
MPL_CACHE_DIR = PROJECT_ROOT / ".cache" / "matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

from dnds_dynamics.snv_helpers.isolate import DEFAULT_SNV_TABLE_ROOT  # noqa: E402
from dnds_dynamics.dnds.stratified import compute_stratified_isolate_dnds  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("accessions", nargs="+")
    parser.add_argument("--table-root", type=Path, default=DEFAULT_SNV_TABLE_ROOT)
    parser.add_argument(
        "--dnds-dir",
        type=Path,
        help="Input dN/dS directory. Only valid for one accession; defaults to <table-root>/<accession>/dnds.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Output root. Defaults to the input dN/dS directory; otherwise writes to <output-root>/<accession>.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--thinning-p", type=float, default=0.5)
    parser.add_argument(
        "--plot-formats",
        nargs="*",
        default=["png", "pdf"],
        help="Plot formats to write. Pass no values after the flag to skip plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dnds_dir is not None and len(args.accessions) != 1:
        raise ValueError("--dnds-dir can only be used with one accession")

    metadata = []
    for accession in args.accessions:
        output_dir = None
        if args.output_root is not None:
            output_dir = args.output_root / accession
        result = compute_stratified_isolate_dnds(
            accession,
            table_root=args.table_root,
            dnds_dir=args.dnds_dir,
            output_dir=output_dir,
            seed=args.seed,
            thinning_p=args.thinning_p,
            plot_formats=tuple(args.plot_formats),
        )
        metadata.append(result.metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
