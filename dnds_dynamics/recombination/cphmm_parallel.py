"""Lean, parallel CP-HMM inference for isolate SNV tables.

The profiler showed CP-HMM per-pair time is ~98% HMM inference and ~1% data
prep, and that the *only* data inference touches is 4D sites. This module
exploits both facts:

1. ``build_inference_bundle`` reduces a fully-loaded :class:`IsolateSNVHelper`
   to a compact set of NumPy arrays covering **only 4D sites**:
     - ``cov`` : (n_samples, n_4D) bool   — per-sample coverage at 4D sites
     - ``geno``: (n_samples, n_4D_snv) uint8 (0/1/255) at 4D *polymorphic* sites
     - ``snv_pos`` : index of each 4D-SNV site within the ordered 4D-site axis
     - ``locs`` / ``contig_codes`` (+ ``contig_names``) on the 4D axis
   Footprint ~285 MB vs the ~4 GB helper, so it fits in shared memory and many
   worker processes can attach to a single copy.

2. :class:`LeanPairProvider` reimplements ``get_pair_snp_info`` in pure NumPy
   over the bundle, byte-identical to ``IsolateSNVHelper.get_pair_snp_info``.

3. :func:`run_inference_parallel` shards the close pairs across processes that
   attach to the shared-memory bundle and run ``infer_pipelines.infer_pairs``.

Correctness is asserted by :func:`verify_against_helper` before any parallel
run is trusted.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import config


def _ensure_cphmm_on_path() -> None:
    config.ensure_cphmm_importable()


@dataclass
class InferenceBundle:
    """Compact 4D-only representation for CP-HMM inference.

    Arrays are sample-major (row per sample) so a pair's coverage/genotype rows
    are contiguous, making the per-pair boolean ops cache-friendly.
    """

    species: str
    genome_len: int
    sample_names: np.ndarray          # (n_samples,) object
    cov: np.ndarray                   # (n_samples, n_4D) bool
    geno: np.ndarray                  # (n_samples, n_4D_snv) uint8 (0/1/255)
    snv_pos: np.ndarray               # (n_4D_snv,) int64 -> column in 4D axis
    locs: np.ndarray                  # (n_4D,) int64
    contig_codes: np.ndarray          # (n_4D,) int32
    contig_names: np.ndarray          # (n_codes,) object; name = contig_names[code]
    hmm_prior_path: str | None = None

    @property
    def sample_to_index(self) -> dict:
        return {str(s): i for i, s in enumerate(self.sample_names)}


def build_inference_bundle(helper, *, prior_path: str | None = None) -> InferenceBundle:
    """Reduce a fully-loaded ``IsolateSNVHelper`` to the 4D-only bundle."""
    if not helper.has_site_annotations:
        raise RuntimeError("bundle requires site annotations (core_4D).")

    core_4d = helper.core_4D.to_numpy(dtype=bool, copy=False)
    idx_4d = np.flatnonzero(core_4d)                     # full-index positions of 4D sites
    n_4d = idx_4d.size

    # Restrict to sample columns in the canonical sample order; the coverage and
    # biallelic tables also carry non-sample columns (e.g. Ref/Major/Alt).
    samples = helper.samples

    # Coverage at 4D sites, sample-major. _coverage_df is (sites x samples).
    cov_full = helper.coverage.loc[:, samples].to_numpy(dtype=bool)  # (n_sites, n_samples)
    cov = np.ascontiguousarray(cov_full[idx_4d].T)       # (n_samples, n_4D)
    del cov_full

    # Genotypes at 4D *polymorphic* sites, sample-major.
    snv_4d_mask = helper.snv_4D.to_numpy(dtype=bool, copy=False)   # over snv_index
    geno_full = helper.snvs.loc[:, samples].to_numpy()   # (n_snv, n_samples) 0/1/255
    geno = np.ascontiguousarray(geno_full[snv_4d_mask].T.astype(np.uint8))  # (n_samples, n_4D_snv)
    del geno_full

    # Map each 4D-SNV site to its column index on the ordered 4D axis.
    snv_full_positions = np.flatnonzero(helper.core_to_snvs)        # full positions of all SNVs
    snv_4d_full_positions = snv_full_positions[snv_4d_mask]         # full positions of 4D SNVs
    snv_pos = np.searchsorted(idx_4d, snv_4d_full_positions).astype(np.int64)
    # Sanity: searchsorted must land exactly on a 4D site.
    if not np.array_equal(idx_4d[snv_pos], snv_4d_full_positions):
        raise AssertionError("4D-SNV sites not all present on the 4D axis.")

    locs = helper.locations[idx_4d].astype(np.int64)
    contigs_4d = helper.chromosomes[idx_4d].astype(str)
    contig_names, contig_codes = np.unique(contigs_4d, return_inverse=True)

    return InferenceBundle(
        species=str(helper.species),
        genome_len=int(core_4d.sum()),
        sample_names=np.asarray([str(s) for s in helper.samples], dtype=object),
        cov=cov,
        geno=geno,
        snv_pos=snv_pos,
        locs=locs,
        contig_codes=contig_codes.astype(np.int32),
        contig_names=np.asarray(contig_names, dtype=object),
        hmm_prior_path=prior_path,
    )


class LeanPairProvider:
    """Datahelper consumed by ``infer_pipelines.infer_pairs`` (pure NumPy)."""

    def __init__(self, bundle: InferenceBundle):
        self.bundle = bundle
        self.species = bundle.species
        self.species_name = bundle.species
        self.genome_len = bundle.genome_len
        self.hmm_prior_path = bundle.hmm_prior_path
        self.sample_names = bundle.sample_names
        self._idx = bundle.sample_to_index

    def get_pair_snp_info(self, pair):
        b = self.bundle
        i = self._idx[str(pair[0])]
        j = self._idx[str(pair[1])]
        covered = b.cov[i] & b.cov[j]                    # (n_4D,)
        gi = b.geno[i]
        gj = b.geno[j]
        d = (gi != gj) & (gi != 255) & (gj != 255)       # (n_4D_snv,)
        diff_4d = np.zeros(b.cov.shape[1], dtype=bool)
        diff_4d[b.snv_pos] = d
        keep = covered
        snp_vec = diff_4d[keep]
        # contig_names already hold Python str; b.locs is int64 — no recasting.
        contigs = b.contig_names[b.contig_codes[keep]]
        locs = b.locs[keep]
        return snp_vec, contigs, locs


# ---------------------------------------------------------------------------
# Shared-memory parallel inference
# ---------------------------------------------------------------------------

# Per-worker globals (populated by _worker_init in each spawned process).
_W: dict = {}


def _worker_init(cov_name, cov_shape, geno_name, geno_shape, small, cfg):
    """Attach to the shared-memory bundle once per worker process."""
    from multiprocessing import shared_memory

    _ensure_cphmm_on_path()
    cov_shm = shared_memory.SharedMemory(name=cov_name)
    geno_shm = shared_memory.SharedMemory(name=geno_name)
    cov = np.ndarray(cov_shape, dtype=bool, buffer=cov_shm.buf)
    geno = np.ndarray(geno_shape, dtype=np.uint8, buffer=geno_shm.buf)
    bundle = InferenceBundle(
        species=small["species"], genome_len=small["genome_len"],
        sample_names=small["sample_names"], cov=cov, geno=geno,
        snv_pos=small["snv_pos"], locs=small["locs"],
        contig_codes=small["contig_codes"], contig_names=small["contig_names"],
        hmm_prior_path=small["prior_path"],
    )
    _W["provider"] = LeanPairProvider(bundle)
    _W["shm"] = (cov_shm, geno_shm)  # keep refs alive
    _W["cfg"] = cfg


def _worker_run(pairs_chunk):
    from cphmm import infer_pipelines

    prov = _W["provider"]
    ccb, iterative, n_iter = _W["cfg"]
    pair_dat, transfer_dat = infer_pipelines.infer_pairs(
        prov, pairs_chunk, clade_cutoff_bin=ccb, iterative=iterative, n_iter=n_iter,
    )
    return pair_dat, transfer_dat


def run_inference_parallel(
    bundle: InferenceBundle,
    pairs,
    *,
    n_workers: int,
    clade_cutoff_bin=None,
    iterative: bool = True,
    n_iter: int = 3,
    chunks_per_worker: int = 4,
):
    """Run CP-HMM over ``pairs`` across ``n_workers`` processes.

    The large ``cov``/``geno`` arrays are placed in shared memory so every
    worker attaches to a single copy (no per-worker multi-GB load). Inference is
    deterministic per pair, so results are independent of sharding/order.
    """
    from multiprocessing import shared_memory
    import concurrent.futures as cf

    pairs = [(str(a), str(b)) for a, b in pairs]
    if n_workers <= 1 or len(pairs) <= 1:
        _ensure_cphmm_on_path()
        from cphmm import infer_pipelines
        return infer_pipelines.infer_pairs(
            LeanPairProvider(bundle), pairs,
            clade_cutoff_bin=clade_cutoff_bin, iterative=iterative, n_iter=n_iter,
        )

    cov_shm = shared_memory.SharedMemory(create=True, size=bundle.cov.nbytes)
    geno_shm = shared_memory.SharedMemory(create=True, size=bundle.geno.nbytes)
    try:
        np.ndarray(bundle.cov.shape, dtype=bool, buffer=cov_shm.buf)[:] = bundle.cov
        np.ndarray(bundle.geno.shape, dtype=np.uint8, buffer=geno_shm.buf)[:] = bundle.geno

        small = {
            "species": bundle.species, "genome_len": bundle.genome_len,
            "sample_names": bundle.sample_names, "snv_pos": bundle.snv_pos,
            "locs": bundle.locs, "contig_codes": bundle.contig_codes,
            "contig_names": bundle.contig_names, "prior_path": bundle.hmm_prior_path,
        }
        cfg = (clade_cutoff_bin, iterative, n_iter)

        # Interleaved shards keep each worker's load balanced across the run.
        n_chunks = max(1, n_workers * chunks_per_worker)
        chunks = [pairs[k::n_chunks] for k in range(n_chunks)]
        chunks = [c for c in chunks if c]

        with cf.ProcessPoolExecutor(
            max_workers=n_workers, initializer=_worker_init,
            initargs=(cov_shm.name, bundle.cov.shape, geno_shm.name, bundle.geno.shape, small, cfg),
        ) as ex:
            results = list(ex.map(_worker_run, chunks))

        pair_dat = pd.concat([r[0] for r in results], ignore_index=True)
        transfer_dat = pd.concat([r[1] for r in results], ignore_index=True)
        return pair_dat, transfer_dat
    finally:
        cov_shm.close(); cov_shm.unlink()
        geno_shm.close(); geno_shm.unlink()


def verify_against_helper(helper, bundle: InferenceBundle, pairs) -> None:
    """Assert the lean provider matches the helper byte-for-byte on ``pairs``."""
    prov = LeanPairProvider(bundle)
    for pair in pairs:
        sv_h, ct_h, lc_h = helper.get_pair_snp_info(
            (str(pair[0]), str(pair[1])), site_class="4D"
        )
        sv_l, ct_l, lc_l = prov.get_pair_snp_info(pair)
        if not (np.array_equal(np.asarray(sv_h, bool), sv_l)
                and np.array_equal(np.asarray(ct_h, str), ct_l)
                and np.array_equal(np.asarray(lc_h, int), lc_l)):
            raise AssertionError(f"lean provider mismatch on pair {pair}")
