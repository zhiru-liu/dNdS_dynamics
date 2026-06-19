"""Compute per-pair identical-fraction artifact for UHGG isolate accessions.

The artifact mirrors the legacy "clonal fraction" matrix that the close-pair
HMM stage 2 used as its pre-HMM pair filter, but stored under the more
accurate name ``identical_fraction`` and as a long-form parquet rather than a
square matrix. See ``notes/identical_fraction_artifact_handoff.md``.

For each ordered pair ``(i, j)`` with sample names ``s_i < s_j`` (lexicographic):
    1. Build the pair's covered ``site_class`` SNP vector
       (default site_class='4D', concatenated across contigs).
    2. Truncate to a whole-block multiple of ``block_size`` (default 1000).
    3. Reshape to ``(num_blocks, block_size)`` and sum per row.
    4. ``identical_fraction = (block_sums == 0).sum() / num_blocks``.

Pairs with fewer than one full block of covered sites are omitted.
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dnds_dynamics.snv_helpers.isolate import (  # noqa: E402
    DEFAULT_SNV_TABLE_ROOT,
    IsolateSNVHelper,
    IsolateTablePaths,
)
from dnds_dynamics.snv_helpers.table_contract import save_snv_dataframe  # noqa: E402


SCHEMA_VERSION = "isolate-identical-fraction-v1"


def _site_class_mask(helper: IsolateSNVHelper, site_class: str) -> np.ndarray:
    if site_class == "4D":
        return helper.core_4D.to_numpy(dtype=bool, copy=False)
    if site_class == "1D":
        return helper.core_1D.to_numpy(dtype=bool, copy=False)
    if site_class in {"all", "covered"}:
        return np.ones(helper.num_sites, dtype=bool)
    raise ValueError(f"Unsupported site_class={site_class!r}")


def _compute_identical_fraction_table(
    helper: IsolateSNVHelper,
    *,
    block_size: int,
    site_class: str,
    progress: bool = True,
) -> pd.DataFrame:
    """Return long-form DataFrame of (sample_1, sample_2, identical_fraction, ...)."""

    if helper.source != "tables":
        raise RuntimeError("compute_identical_fraction requires source='tables'")

    cov_arr = helper.coverage.loc[:, helper.samples].to_numpy(dtype=bool, copy=False)
    snvs_arr = helper.snvs.loc[:, helper.samples].to_numpy(dtype=np.uint8, copy=False)
    site_mask = _site_class_mask(helper, site_class)
    core_to_snvs = np.asarray(helper.core_to_snvs, dtype=bool)

    # Positions of each SNV row inside the core/coverage index.
    snv_positions_in_core = np.flatnonzero(core_to_snvs)
    if snv_positions_in_core.shape[0] != snvs_arr.shape[0]:
        # Defensive: handle the case where snv_index does not align with
        # flatnonzero(core_to_snvs) (e.g. when biallelic SNVs were filtered).
        snv_positions_in_core = helper.coverage.index.get_indexer(helper.snvs.index)
        if (snv_positions_in_core < 0).any():
            raise RuntimeError(
                f"{helper.species}: some SNV rows are absent from the coverage index"
            )

    # Restrict SNV rows to the chosen site class.
    snv_is_in_class = site_mask[snv_positions_in_core]
    snv_indices_in_class = np.flatnonzero(snv_is_in_class)
    snv_positions_in_class_sites = snv_positions_in_core[snv_indices_in_class]

    # Compress to "class-only" coordinates so we can build a per-pair bool
    # vector of length n_class_sites without ever touching the full core.
    class_positions = np.flatnonzero(site_mask)
    n_class_sites = class_positions.shape[0]
    snv_index_within_class = np.searchsorted(class_positions, snv_positions_in_class_sites)

    cov_at_class = cov_arr[class_positions, :]  # (n_class_sites, n_samples) bool

    samples = helper.samples.astype(str).to_numpy()
    n_samples = samples.size
    order = np.argsort(samples, kind="stable")

    snp_vec_buffer = np.zeros(n_class_sites, dtype=bool)

    iterator: object
    pair_indices: list[tuple[int, int]] = [
        (order[i], order[j])
        for i in range(n_samples)
        for j in range(i + 1, n_samples)
    ]
    if progress:
        try:
            from tqdm import tqdm  # type: ignore

            iterator = tqdm(pair_indices, desc=f"pairs[{helper.species}]")
        except ModuleNotFoundError:
            iterator = pair_indices
    else:
        iterator = pair_indices

    sample_1: list[str] = []
    sample_2: list[str] = []
    fractions: list[float] = []
    num_blocks_list: list[int] = []
    num_identical_blocks_list: list[int] = []

    for i_pos, j_pos in iterator:
        s_i = samples[i_pos]
        s_j = samples[j_pos]
        if s_i > s_j:
            s_i, s_j = s_j, s_i

        covered_class = cov_at_class[:, i_pos] & cov_at_class[:, j_pos]
        n_covered = int(covered_class.sum())
        n_blocks = n_covered // block_size
        if n_blocks == 0:
            continue

        # Differences at SNV rows whose site is in the chosen class.
        snv_i = snvs_arr[snv_indices_in_class, i_pos]
        snv_j = snvs_arr[snv_indices_in_class, j_pos]
        diff_at_snv = (snv_i != snv_j) & (snv_i != 255) & (snv_j != 255)

        # Build snp_vec at class-site resolution, then restrict to covered.
        snp_vec_buffer[:] = False
        if diff_at_snv.any():
            snp_vec_buffer[snv_index_within_class[diff_at_snv]] = True
        snp_vec = snp_vec_buffer[covered_class]
        snp_vec_trunc = snp_vec[: n_blocks * block_size]
        block_sums = snp_vec_trunc.reshape(n_blocks, block_size).sum(axis=1)
        n_identical = int((block_sums == 0).sum())

        sample_1.append(s_i)
        sample_2.append(s_j)
        fractions.append(float(n_identical) / float(n_blocks))
        num_blocks_list.append(int(n_blocks))
        num_identical_blocks_list.append(n_identical)

    return pd.DataFrame(
        {
            "sample_1": pd.array(sample_1, dtype="string"),
            "sample_2": pd.array(sample_2, dtype="string"),
            "identical_fraction": np.asarray(fractions, dtype=np.float64),
            "num_blocks": np.asarray(num_blocks_list, dtype=np.int64),
            "num_identical_blocks": np.asarray(num_identical_blocks_list, dtype=np.int64),
        }
    )


def compute_identical_fraction(
    accession: str,
    *,
    table_root: Path = DEFAULT_SNV_TABLE_ROOT,
    block_size: int = 1000,
    site_class: str = "4D",
    overwrite: bool = False,
    progress: bool = True,
) -> dict:
    paths = IsolateTablePaths.from_root(accession, table_root)
    out_path = paths.identical_fraction
    meta_path = paths.identical_fraction_metadata

    if out_path.exists() and meta_path.exists() and not overwrite:
        meta = json.loads(meta_path.read_text())
        return {**meta, "skipped": True}

    helper = IsolateSNVHelper(
        accession,
        source="tables",
        table_root=table_root,
        compute_bi_snvs=False,
        mask_multi_sites=True,
        annotate=True,
    )
    if not helper.has_site_annotations:
        raise RuntimeError(
            f"{accession}: site_annotations.parquet missing; cannot identify {site_class} sites."
        )

    df = _compute_identical_fraction_table(
        helper,
        block_size=block_size,
        site_class=site_class,
        progress=progress,
    )

    save_snv_dataframe(df, out_path, helper.format)

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "accession": accession,
        "block_size": int(block_size),
        "site_class": site_class,
        "num_samples": int(len(helper.samples)),
        "num_pairs": int(df.shape[0]),
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "helper_kwargs": {
            "source": "tables",
            "annotate": True,
            "compute_bi_snvs": False,
            "mask_multi_sites": True,
        },
        "table_root": str(table_root),
    }
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("accessions", nargs="+")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_SNV_TABLE_ROOT)
    parser.add_argument("--block-size", type=int, default=1000)
    parser.add_argument("--site-class", default="4D")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--cutoff", type=float, default=0.5,
                        help="Only used for the per-accession summary printed at the end.")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = []
    for accession in args.accessions:
        meta = compute_identical_fraction(
            accession,
            table_root=args.output_root,
            block_size=args.block_size,
            site_class=args.site_class,
            overwrite=args.overwrite,
            progress=not args.no_progress,
        )
        # Load to print summary.
        paths = IsolateTablePaths.from_root(accession, args.output_root)
        df = pd.read_parquet(paths.identical_fraction) if paths.identical_fraction.suffix == ".parquet" else pd.read_feather(paths.identical_fraction)
        n_close = int((df["identical_fraction"] > args.cutoff).sum())
        print(
            f"[{accession}] n_pairs={df.shape[0]:6d}  n_close(@{args.cutoff})={n_close:5d}  "
            f"mean={df['identical_fraction'].mean():.4f}  max={df['identical_fraction'].max():.4f}"
        )
        summaries.append({**meta, "n_close_pairs": n_close})
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
