"""QP (Garud & Good 2019 / Liu & Good 2024) metagenome SNV access.

Thin adapter over the QP ``SNVHelper`` bundled in the editable ``cphmm`` package
(``cphmm.io.liugood2024_qp.snv_utils``), wired to this repo's :mod:`config`
paths. This is the single entry point for QP SNV access and replaces the old
``os.chdir`` + ``config.yml`` import of the ``LiuGood2024_data`` sibling repo —
that repo is no longer a code dependency.

The bundled helper takes explicit ``data_dir`` / ``reference_dir`` (keyword-only)
where the old one used a defaulted ``snv_path`` and a cwd-relative reference;
:func:`load_qp_snv_helper` fills both from config so callers don't pass paths.

``compute_biallelic_snvs`` and ``polarize_reference_seq`` are re-exported for the
isolate-table builder, which previously imported them from the sibling repo.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from cphmm.io.liugood2024_qp.snv_utils import (  # noqa: F401  (re-exported)
    SNVHelper,
    compute_biallelic_snvs,
    polarize_reference_seq,
)

from .. import config


class QPSNVHelper(SNVHelper):
    """QP ``SNVHelper`` extended with identical-block (clonal-fraction) lookups.

    The cphmm-bundled QP ``SNVHelper`` covers recombination-side needs but does
    not expose the precomputed per-pair *identical-block fraction* (clonal
    fraction) used by the clonal-SNV-prevalence figure. These methods, ported
    verbatim from the original published-repo helper, read the per-species
    identical-block matrices at :data:`config.identical_fraction_path` and add
    the fully-recombined-pair sampling that depended on them. This consolidates
    the two parallel QP helpers onto the cphmm base class.
    """

    def load_identical_block(self) -> None:
        path = config.identical_fraction_path / f"{self.species_name}.csv"
        try:
            identical_block_frac = pd.read_csv(path, index_col=None, header=None)
        except FileNotFoundError:
            logging.error("Identical block fraction file not found: %s", path)
            return
        identical_block_frac.set_index(self.samples, inplace=True)
        identical_block_frac.columns = self.samples
        self.identical_block_frac = identical_block_frac

    def get_pair_identical_block(self, sample1, sample2):
        if not hasattr(self, "identical_block_frac"):
            self.load_identical_block()
        return self.identical_block_frac.at[sample1, sample2]

    def check_if_any_pair_fully_recombined(self, recomb_threshold=None) -> bool:
        if recomb_threshold is None:
            recomb_threshold = config.fully_recombined_threshold
        if not hasattr(self, "identical_block_frac"):
            self.load_identical_block()
        return bool((self.identical_block_frac <= recomb_threshold).any().any())

    def sample_random_fully_recombined_pair(self, recomb_threshold=None):
        if recomb_threshold is None:
            recomb_threshold = config.fully_recombined_threshold
        if not self.check_if_any_pair_fully_recombined(recomb_threshold):
            # do not want to get stuck in an infinite loop if no fully recombined pairs
            raise ValueError("No fully recombined pairs found")
        sample1, sample2 = self.sample_random_pair()
        while self.get_pair_identical_block(sample1, sample2) > recomb_threshold:
            sample1, sample2 = self.sample_random_pair()
        return sample1, sample2


def load_qp_snv_helper(
    species: str,
    *,
    annotate: bool = True,
    data_dir: Path | str | None = None,
    reference_dir: Path | str | None = None,
    snv_format: str = "feather",
    compute_bi_snvs: bool = False,
    save_bi_snvs: bool = False,
    mask_multi_sites: bool = True,
    **kwargs,
) -> "SNVHelper":
    """Construct a QP ``SNVHelper`` for ``species`` using config data roots.

    Defaults reproduce the previous LiuGood-loader behaviour (feather catalogs,
    no biallelic recompute, multi-allelic sites masked). Returns a
    :class:`QPSNVHelper` so identical-block (clonal-fraction) lookups are
    available alongside the base recombination-side API.
    """
    return QPSNVHelper(
        species,
        data_dir=Path(data_dir) if data_dir is not None else config.GG2019_SNV_FEATHER,
        reference_dir=(
            Path(reference_dir) if reference_dir is not None else config.GG2019_REP_GENOMES
        ),
        snv_format=snv_format,
        compute_bi_snvs=compute_bi_snvs,
        save_bi_snvs=save_bi_snvs,
        mask_multi_sites=mask_multi_sites,
        annotate=annotate,
        **kwargs,
    )
