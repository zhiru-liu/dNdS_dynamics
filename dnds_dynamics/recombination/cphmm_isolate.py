"""CP-HMM recombination detection for in-house isolate SNV tables.

Self-contained driver that imports the *core* CP-HMM functions from the
``close_pair_hmm`` package and runs recombination inference directly on the
isolate SNV tables built in this repo (e.g. the A. putredinis NCBI-isolate
table at ``/Volumes/Botein/ncbi_isolates/Alistipes_putredinis/snv_table``),
without depending on the UHGG-specific ``Isolate_test`` helpers.

Pipeline (mirrors ``close_pair_hmm/Isolate_test`` but parameterized by an
arbitrary ``table_root``/``accession`` and wired to this repo's
recombination-cache writer):

1. ``IsolateCPHMMDataHelper`` wraps :class:`IsolateSNVHelper`, exposing the
   attributes/methods CP-HMM expects: ``species``, ``genome_len``,
   ``get_close_pairs``, ``get_pair_snp_info``, and the prior-sampling hooks
   (``get_random_pair``/``get_snp_vector``/``sample_prior_blocks``).
2. :func:`generate_prior` builds a per-species transfer-divergence prior from
   "fully recombined" (diverged) pairs via ``cphmm.prior``.
3. :func:`run_inference` runs ``infer_pipelines.infer_pairs`` over close pairs.
4. :func:`build_and_save_cache` converts the inference/transfer CSVs into this
   repo's recombination-event cache (``recombination_events.parquet`` etc.).

Prereqs on the SNV table directory:
- ``site_annotations.parquet`` (for 4D masks; built with annotate=True).
- ``identical_fraction.parquet`` + metadata (close/diverged pair selection);
  generate with ``analysis/compute_identical_fraction.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .. import config
from ..snv_helpers.isolate import IsolateSNVHelper
from .cache import (
    RecombinationCachePaths,
    build_new_cphmm_recombination_cache,
    save_recombination_cache,
)

CLONAL_FRAC_CUTOFF = 0.5
IDENTICAL_FRACTION_BLOCK_SIZE = 1000
DIVERGED_PAIR_MAX_IDENTICAL_FRACTION = 0.05
SITE_CLASS = "4D"


def _ensure_cphmm_on_path() -> None:
    """Make close_pair_hmm (cphmm, infer_pipelines) importable."""
    config.ensure_cphmm_importable()


# ---------------------------------------------------------------------------
# Data helper
# ---------------------------------------------------------------------------


class IsolateCPHMMDataHelper:
    """Adapter consumed by ``infer_pipelines.infer_pairs`` and ``cphmm.prior``.

    Parameters
    ----------
    accession
        Sub-directory name under ``table_root`` (e.g. ``"snv_table"``).
    table_root
        Root containing ``<accession>/`` with the SNV-table parquets.
    prior_dir
        Directory holding ``<species>.csv`` priors (read/written by cphmm.prior).
    """

    def __init__(
        self,
        accession: str,
        table_root: Path | str,
        *,
        prior_dir: Path | str,
        clonal_frac_cutoff: float = CLONAL_FRAC_CUTOFF,
        identical_fraction_block_size: int = IDENTICAL_FRACTION_BLOCK_SIZE,
        diverged_pair_max_identical_fraction: float = DIVERGED_PAIR_MAX_IDENTICAL_FRACTION,
        site_class: str = SITE_CLASS,
        max_pairs: int | None = None,
        exclude_samples: Iterable[str] | None = None,
    ) -> None:
        self.accession = accession
        self.species = accession  # CP-HMM prior filename key
        self.species_name = accession
        self.table_root = Path(table_root)
        self.hmm_prior_path = str(prior_dir)
        self.clonal_frac_cutoff = float(clonal_frac_cutoff)
        self.identical_fraction_block_size = int(identical_fraction_block_size)
        self.diverged_pair_max_identical_fraction = float(diverged_pair_max_identical_fraction)
        self.site_class = site_class

        self.helper = IsolateSNVHelper(
            accession,
            table_root=self.table_root,
            source="tables",
            annotate=True,
            compute_bi_snvs=False,
            save_bi_snvs=False,
            mask_multi_sites=True,
        )
        if not self.helper.has_site_annotations:
            raise RuntimeError(
                f"{accession}: site_annotations.parquet missing; CP-HMM needs core_{site_class}."
            )

        # Excluded samples are removed from sample_names so they never enter the
        # close-pair or diverged-pair selection (both filter through
        # sample_to_index). Used here to drop the misclassified P. dorei genome
        # GCF_048453085.1 and keep P. vulgatus a single clade.
        self.exclude_samples = {str(s) for s in (exclude_samples or [])}
        all_names = [str(s) for s in self.helper.samples]
        self.sample_names = np.asarray(
            [s for s in all_names if s not in self.exclude_samples], dtype=object
        )
        self.sample_to_index = {s: i for i, s in enumerate(self.sample_names)}
        self.genome_len = int(self.helper.genome_len)

        self.missing_pairs: list[tuple[str, str]] = []
        self.close_pairs = self._load_close_pairs(max_pairs=max_pairs)
        self._diverged_pairs: list[tuple[str, str]] | None = None

    # ---- close / diverged pair selection -----------------------------------

    def _load_close_pairs(self, *, max_pairs: int | None) -> list[tuple[str, str]]:
        pairs = self.helper.get_close_pairs(
            cutoff=self.clonal_frac_cutoff,
            block_size=self.identical_fraction_block_size,
            site_class=self.site_class,
        )
        kept = []
        for a, b in pairs:
            a, b = str(a), str(b)
            if a in self.sample_to_index and b in self.sample_to_index:
                kept.append((a, b))
            else:
                self.missing_pairs.append((a, b))
        if max_pairs is not None:
            kept = kept[:max_pairs]
        return kept

    def get_close_pairs(self) -> list[tuple[str, str]]:
        return self.close_pairs

    def _load_diverged_pairs(self) -> list[tuple[str, str]]:
        cutoff = self.diverged_pair_max_identical_fraction
        df = self.helper.identical_fraction
        mask = df["identical_fraction"] <= cutoff
        pairs = [
            (str(a), str(b))
            for a, b in df.loc[mask, ["sample_1", "sample_2"]].itertuples(index=False)
            if str(a) in self.sample_to_index and str(b) in self.sample_to_index
        ]
        if not pairs:
            raise RuntimeError(
                f"{self.accession}: no pairs with identical_fraction <= {cutoff} "
                "for prior block sampling. Loosen the cutoff or check the cache."
            )
        return pairs

    def get_diverged_pairs(self) -> list[tuple[str, str]]:
        if self._diverged_pairs is None:
            self._diverged_pairs = self._load_diverged_pairs()
        return self._diverged_pairs

    # ---- CP-HMM inference inputs --------------------------------------------

    def get_pair_snp_info(self, pair):
        snp_vec, contigs, locs = self.helper.get_pair_snp_info(
            (str(pair[0]), str(pair[1])), site_class=self.site_class
        )
        return (
            np.asarray(snp_vec, dtype=bool),
            np.asarray(contigs, dtype=str),
            np.asarray(locs, dtype=int),
        )

    # ---- prior-sampling hooks (cphmm.prior.sample_blocks) -------------------

    def get_snp_vector(self, pair):
        return self.get_pair_snp_info(pair)[0]

    def get_random_pair(self):
        diverged = self.get_diverged_pairs()
        return diverged[np.random.randint(0, len(diverged))]

    def sample_prior_blocks(self, num_samples=5000, block_size=1000, random_state=None):
        """Sample local/genome divergences from diverged pairs only.

        Grouped so each pair's SNP vector is built at most once. Mirrors the
        UHGG ``DataHelper_Isolate.sample_prior_blocks``.
        """
        rng = np.random.default_rng(random_state)
        local_divs = np.empty(num_samples)
        genome_divs = np.empty(num_samples)
        diverged_pairs = self.get_diverged_pairs()
        n_pairs = len(diverged_pairs)

        pair_choices = rng.integers(0, n_pairs, size=num_samples)
        grouped: dict[int, list[int]] = {}
        for out_idx, pair_idx in enumerate(pair_choices):
            grouped.setdefault(int(pair_idx), []).append(out_idx)

        pending: list[int] = []
        for pair_idx, out_idxs in grouped.items():
            snp_vec = self.get_snp_vector(diverged_pairs[pair_idx])
            if len(snp_vec) < block_size:
                pending.extend(out_idxs)
                continue
            genome_div = float(np.mean(snp_vec))
            starts = rng.integers(0, len(snp_vec) - block_size + 1, size=len(out_idxs))
            for out_idx, start in zip(out_idxs, starts):
                local_divs[out_idx] = float(np.mean(snp_vec[start:start + block_size]))
                genome_divs[out_idx] = genome_div

        attempts, max_attempts = 0, max(len(pending) * 100, 1000)
        while pending:
            attempts += 1
            if attempts > max_attempts:
                raise ValueError(
                    f"{self.accession}: cannot sample {block_size}-site blocks from "
                    f"{n_pairs} diverged pairs; loosen diverged-pair cutoff."
                )
            out_idx = pending.pop()
            snp_vec = self.get_snp_vector(diverged_pairs[int(rng.integers(0, n_pairs))])
            if len(snp_vec) < block_size:
                pending.append(out_idx)
                continue
            start = rng.integers(0, len(snp_vec) - block_size + 1)
            local_divs[out_idx] = float(np.mean(snp_vec[start:start + block_size]))
            genome_divs[out_idx] = float(np.mean(snp_vec))

        return local_divs, genome_divs


# ---------------------------------------------------------------------------
# Prior + inference + cache
# ---------------------------------------------------------------------------


def generate_prior(
    dh: IsolateCPHMMDataHelper,
    *,
    prior_dir: Path | str,
    num_samples: int = 5000,
    block_size: int = 1000,
    num_bins: int | None = None,
    separate_clades: bool = True,
    clade_cutoff: float = 0.03,
    seed: int = 0,
    overwrite: bool = False,
) -> Path:
    """Build and save a per-species transfer-divergence prior via cphmm.prior.

    With ``separate_clades=False`` the histogram is a single ``num_bins`` block
    (no within/between-clade split) — appropriate for single-clade species such
    as the P. vulgatus isolate set after dropping the P. dorei outlier.
    """
    _ensure_cphmm_on_path()
    import cphmm.config
    import cphmm.prior

    if num_bins is None:
        num_bins = cphmm.config.HMM_PRIOR_BINS
    prior_dir = Path(prior_dir)
    prior_dir.mkdir(parents=True, exist_ok=True)
    prior_path = Path(cphmm.prior.get_prior_filename(dh.species, prior_path=str(prior_dir)))
    if prior_path.exists() and not overwrite:
        return prior_path

    local_divs, genome_divs = cphmm.prior.sample_blocks(
        dh, num_samples=num_samples, block_size=block_size, random_state=seed
    )
    divs, counts = cphmm.prior.compute_div_histogram(
        local_divs, genome_divs, num_bins=num_bins,
        separate_clades=separate_clades, clade_cutoff=clade_cutoff,
    )
    cphmm.prior.save_prior(divs, counts, dh.species, prior_path=str(prior_dir))
    return prior_path


def run_inference(
    dh: IsolateCPHMMDataHelper,
    *,
    clade_cutoff_bin: int | None = 40,
    iterative: bool = False,
    n_iter: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run CP-HMM over the helper's close pairs; returns (pair_dat, transfer_dat)."""
    _ensure_cphmm_on_path()
    from cphmm import infer_pipelines

    return infer_pipelines.infer_pairs(
        dh, dh.get_close_pairs(),
        clade_cutoff_bin=clade_cutoff_bin, iterative=iterative, n_iter=n_iter,
    )


def write_inference_csvs(
    accession: str,
    pair_dat: pd.DataFrame,
    transfer_dat: pd.DataFrame,
    results_dir: Path | str,
    *,
    suffix: str = "all_pairs",
) -> tuple[Path, Path]:
    """Write the two CSVs in the schema build_new_cphmm_recombination_cache reads."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    pair_path = results_dir / f"{accession}__{suffix}__inference_summary.csv"
    transfer_path = results_dir / f"{accession}__{suffix}__transfer_summary.csv"
    pair_dat.to_csv(pair_path, index=False)
    transfer_dat.to_csv(transfer_path, index=False)
    return pair_path, transfer_path


def build_and_save_cache(
    accession: str,
    *,
    results_dir: Path | str,
    cache_root: Path | str,
    suffix: str = "all_pairs",
) -> RecombinationCachePaths:
    """Convert inference CSVs to the recombination cache and save under cache_root/accession."""
    events, pairs, metadata = build_new_cphmm_recombination_cache(
        accession, results_dir=results_dir, suffix=suffix
    )
    paths = RecombinationCachePaths.from_root(accession, cache_root)
    save_recombination_cache(events, pairs, metadata, paths)
    return paths
