"""CP-HMM recombination detection for isolate SNV tables — single clean entry point.

The single CP-HMM driver for isolate SNV tables, parameterized by per-species
presets (sequential by default; ``--workers N`` selects the shared-memory parallel
path). Mirrors the structure of the ``close_pair_hmm`` workflows: a data-helper
adapter (``IsolateCPHMMDataHelper``) feeds ``cphmm.infer_pipelines``.

Pipeline (all from the on-disk SNV table, which now carries a QP/MIDAS ``core_genes.json``):
  1. Build the CP-HMM data helper over the table (uses the prevalence-filtered core).
  2. Generate / reuse the per-species transfer-divergence prior (from diverged pairs).
  3. Run CP-HMM inference over the close pairs (checkpointed by default).
  4. Write inference / transfer CSVs and the recombination-event cache.

Checkpointing (``--batch-size N``, default 50):
  Pairs are processed in batches of N.  After each batch, pair + transfer results
  are appended as parquet files to ``<out-base>/checkpoints/``.  On restart the
  completed pairs are loaded and skipped so the run continues from where it
  stopped.  Set ``--batch-size 0`` to disable checkpointing (all-or-nothing).

By default this is **sequential (single worker)** — lowest memory, no shared-memory
bundle. ``--workers N`` (N>1) uses the parallel path instead.

Outputs default to ``outputs/<species>_isolate_cphmm_qpcore/`` (results, priors, and a
self-contained ``cache/`` recombination cache) so previous runs are never overwritten.

  python analysis/run_isolate_cphmm.py aputredinis
  python analysis/run_isolate_cphmm.py pvulgatus --max-pairs 2     # smoke test
  python analysis/run_isolate_cphmm.py pvulgatus                   # resumes from checkpoints
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dnds_dynamics import config  # noqa: E402
from dnds_dynamics.recombination.cphmm_isolate import (  # noqa: E402
    IsolateCPHMMDataHelper,
    build_and_save_cache,
    generate_prior,
    write_inference_csvs,
)

ISOLATE_ROOT = config.NCBI_ISOLATES_ROOT

# Per-species presets. ``midas_species`` selects the QP/MIDAS reference (and the
# core_genes.json reused from the matching QP catalog). Both species are single
# clade (no within/between split); Pv drops the misclassified P. dorei genome.
PRESETS = {
    "aputredinis": dict(
        table_root=ISOLATE_ROOT / "Alistipes_putredinis",
        midas_species="Alistipes_putredinis_61533",
        exclude_samples=[],
        separate_clades=False,
        iterative=True,
    ),
    "pvulgatus": dict(
        table_root=ISOLATE_ROOT / "Phocaeicola_vulgatus",
        midas_species="Bacteroides_vulgatus_57955",
        exclude_samples=["GCF_048453085.1"],
        separate_clades=False,
        iterative=True,
    ),
}


def _load_checkpoints(checkpoint_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, set]:
    """Return (pair_dfs_concat, transfer_dfs_concat, done_pair_set) from saved checkpoints."""
    pair_files = sorted(checkpoint_dir.glob("pairs_batch_*.parquet"))
    transfer_files = sorted(checkpoint_dir.glob("transfers_batch_*.parquet"))
    if not pair_files:
        return pd.DataFrame(), pd.DataFrame(), set()
    pair_dfs = [pd.read_parquet(f) for f in pair_files]
    transfer_dfs = [pd.read_parquet(f) for f in transfer_files]
    pair_df = pd.concat(pair_dfs, ignore_index=True)
    transfer_df = pd.concat(transfer_dfs, ignore_index=True) if transfer_dfs else pd.DataFrame()
    done = set(zip(pair_df["genome1"], pair_df["genome2"]))
    return pair_df, transfer_df, done


def _infer_batch(dh, batch, *, clade_cutoff_bin, iterative, n_iter):
    """Call infer_pipelines.infer_pairs on a subset of pairs."""
    from dnds_dynamics import config as _config
    _config.ensure_cphmm_importable()
    from cphmm import infer_pipelines
    return infer_pipelines.infer_pairs(
        dh, batch,
        clade_cutoff_bin=clade_cutoff_bin,
        iterative=iterative, n_iter=n_iter,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("species", choices=sorted(PRESETS), help="per-species preset")
    ap.add_argument("--accession", default="snv_table")
    ap.add_argument("--out-base", type=Path, default=None,
                    help="Output base dir (default outputs/<species>_isolate_cphmm_qpcore).")
    ap.add_argument("--max-pairs", type=int, default=None, help="limit pairs (smoke test)")
    ap.add_argument("--workers", type=int, default=1,
                    help="1 = sequential (default, low memory); >1 = shared-memory parallel.")
    ap.add_argument("--batch-size", type=int, default=50,
                    help="Checkpoint every N pairs (default 50). 0 = no checkpointing.")
    ap.add_argument("--clonal-frac-cutoff", type=float, default=0.5)
    ap.add_argument("--diverged-max-if", type=float, default=0.05)
    ap.add_argument("--prior-samples", type=int, default=5000)
    ap.add_argument("--n-iter", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite-prior", action="store_true", default=True,
                    help="Regenerate the prior with the current (filtered) core (default on).")
    ap.add_argument("--keep-prior", dest="overwrite_prior", action="store_false")
    args = ap.parse_args()

    cfg = PRESETS[args.species]
    out_base = args.out_base or (REPO_ROOT / "outputs" / f"{args.species}_isolate_cphmm_qpcore")
    results_dir = out_base
    prior_dir = out_base / "priors"
    cache_root = out_base / "cache"  # self-contained; never touches the on-disk table cache
    checkpoint_dir = out_base / "checkpoints"
    out_base.mkdir(parents=True, exist_ok=True)

    tag = f"[{args.species}]"
    print(f"{tag} loading SNV table {cfg['table_root']} at {time.ctime()}")
    dh = IsolateCPHMMDataHelper(
        args.accession, cfg["table_root"], prior_dir=prior_dir,
        clonal_frac_cutoff=args.clonal_frac_cutoff,
        diverged_pair_max_identical_fraction=args.diverged_max_if,
        max_pairs=args.max_pairs, exclude_samples=cfg["exclude_samples"],
    )
    if cfg["exclude_samples"]:
        print(f"{tag} excluded: {sorted(dh.exclude_samples)}")
    close_pairs = dh.get_close_pairs()
    print(f"{tag} {len(dh.sample_names)} isolates, genome_len={dh.genome_len} core-{dh.site_class} "
          f"sites, {len(close_pairs)} close pairs, {len(dh.get_diverged_pairs())} diverged (prior)")

    print(f"{tag} generating prior at {time.ctime()}")
    prior_path = generate_prior(
        dh, prior_dir=prior_dir, num_samples=args.prior_samples,
        separate_clades=cfg["separate_clades"], seed=args.seed, overwrite=args.overwrite_prior,
    )
    print(f"{tag} prior: {prior_path}")

    # single-clade => no within/between split => clade_cutoff_bin must be None
    clade_cutoff_bin = 40 if cfg["separate_clades"] else None
    t0 = time.time()
    print(f"{tag} CP-HMM inference at {time.ctime()} "
          f"(workers={args.workers}, separate_clades={cfg['separate_clades']}, "
          f"iterative={cfg['iterative']}, batch_size={args.batch_size})")

    # --- parallel path (no checkpointing) ---
    if args.workers > 1:
        from dnds_dynamics.recombination.cphmm_parallel import (
            build_inference_bundle, run_inference_parallel,
        )
        bundle = build_inference_bundle(dh.helper, prior_path=str(prior_dir))
        pair_dat, transfer_dat = run_inference_parallel(
            bundle, close_pairs, n_workers=args.workers,
            clade_cutoff_bin=clade_cutoff_bin, iterative=cfg["iterative"], n_iter=args.n_iter,
        )

    # --- sequential, no checkpointing ---
    elif args.batch_size <= 0:
        pair_dat, transfer_dat = _infer_batch(
            dh, close_pairs,
            clade_cutoff_bin=clade_cutoff_bin,
            iterative=cfg["iterative"], n_iter=args.n_iter,
        )

    # --- sequential, checkpointed ---
    else:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        prev_pair_df, prev_transfer_df, done_pairs = _load_checkpoints(checkpoint_dir)
        n_done_prev = len(done_pairs)
        if n_done_prev:
            print(f"{tag} resuming: {n_done_prev} pairs already done from checkpoints")

        remaining = [(a, b) for a, b in close_pairs if (a, b) not in done_pairs]
        print(f"{tag} {len(remaining)} pairs to process")

        pair_dfs = [prev_pair_df] if not prev_pair_df.empty else []
        transfer_dfs = [prev_transfer_df] if not prev_transfer_df.empty else []
        batch_idx_start = len(list(checkpoint_dir.glob("pairs_batch_*.parquet")))

        for batch_start in range(0, len(remaining), args.batch_size):
            batch = remaining[batch_start:batch_start + args.batch_size]
            b_pair, b_transfer = _infer_batch(
                dh, batch,
                clade_cutoff_bin=clade_cutoff_bin,
                iterative=cfg["iterative"], n_iter=args.n_iter,
            )
            batch_idx = batch_idx_start + batch_start // args.batch_size
            b_pair.to_parquet(checkpoint_dir / f"pairs_batch_{batch_idx:04d}.parquet", index=False)
            b_transfer.to_parquet(checkpoint_dir / f"transfers_batch_{batch_idx:04d}.parquet", index=False)
            pair_dfs.append(b_pair)
            transfer_dfs.append(b_transfer)
            n_total_done = n_done_prev + batch_start + len(batch)
            print(f"{tag} checkpoint {batch_idx}: {n_total_done}/{len(close_pairs)} pairs done "
                  f"at {time.ctime()}")

        pair_dat = pd.concat(pair_dfs, ignore_index=True) if pair_dfs else pd.DataFrame(
            columns=["genome1", "genome2", "naive_div", "est_div", "genome_len", "clonal_len"])
        transfer_dat = pd.concat(transfer_dfs, ignore_index=True) if transfer_dfs else pd.DataFrame(
            columns=["genome1", "genome2", "snp_vec_start", "snp_vec_end"])

    dt = time.time() - t0
    print(f"{tag} inference done in {dt:.1f}s ({dt/max(len(pair_dat),1):.3f}s/pair); "
          f"{len(pair_dat)} pairs, {len(transfer_dat)} transfers")

    suffix = "all_pairs" if args.max_pairs is None else f"first_{args.max_pairs}_pairs"
    if cfg["iterative"]:
        suffix += "__iterative"
    pair_csv, transfer_csv = write_inference_csvs(
        args.accession, pair_dat, transfer_dat, results_dir, suffix=suffix
    )
    print(f"{tag} wrote {pair_csv.name}, {transfer_csv.name}")
    paths = build_and_save_cache(
        args.accession, results_dir=results_dir, cache_root=cache_root, suffix=suffix,
    )
    print(f"{tag} wrote recombination cache: {paths.events}")
    print(f"{tag} DONE at {time.ctime()}")


if __name__ == "__main__":
    main()
