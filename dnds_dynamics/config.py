"""Central configuration: data roots, sibling-repo locations, path helpers.

Every absolute path the analysis touches is resolved here, and every value can
be overridden with an environment variable so the repo runs unchanged on a
different machine. Defaults reproduce the original development layout
(``/Volumes/Botein`` data drive, sibling repos under ``~/.../bgoodlab``).

Resolution order for a root: ``$ENV_VAR`` if set, else the built-in default.

The only external *code* dependency is ``close_pair_hmm`` (the editable
``cphmm`` package), reached through :func:`ensure_cphmm_importable`. It tries a
normal import first (so the ``pip install -e`` just works) and only falls back
to inserting the repo root on ``sys.path``, so it is a no-op with the editable
install. The QP ``SNVHelper`` is sourced from ``cphmm.io.liugood2024_qp`` (see
``snv_helpers/qp.py``) and codon site-type annotation is vendored in
``snv_helpers/codon_annotation.py`` — neither is a separate external dependency.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path


def _root(env_var: str, default: str | Path) -> Path:
    """Return ``$env_var`` as a Path if set, else ``default``."""
    return Path(os.environ.get(env_var, str(default))).expanduser()


# ---------------------------------------------------------------------------
# Top-level roots (override these to relocate everything below)
# ---------------------------------------------------------------------------
#: External data drive holding SNV catalogs, isolate tables, analysis pickles.
BOTEIN_ROOT = _root("DNDS_BOTEIN_ROOT", "/Volumes/Botein")
#: Parent of the sibling research repos (close_pair_hmm, LiuGood2024_data, ...).
BGOODLAB_ROOT = _root(
    "DNDS_BGOODLAB_ROOT", "/Users/Device6/Documents/Research/bgoodlab"
)
#: Parent of external command-line tools (MUMmer, NCBI datasets).
TOOLS_ROOT = _root("DNDS_TOOLS_ROOT", "/Users/Device6/Documents/Research/tools")


# ---------------------------------------------------------------------------
# Sibling code repositories
# ---------------------------------------------------------------------------
CLOSE_PAIR_HMM_ROOT = _root("DNDS_CPHMM_ROOT", BGOODLAB_ROOT / "close_pair_hmm")
#: Original published dN/dS-dynamics repo (for its ``data/`` figure inputs).
DNDS_DYNAMICS_ROOT = _root(
    "DNDS_DYNAMICS_ROOT", BGOODLAB_ROOT / "dNdS" / "dNdS_dynamics"
)
DNDS_DYNAMICS_DATA = DNDS_DYNAMICS_ROOT / "data"

# CP-HMM per-experiment result dirs (under the cphmm repo).
CPHMM_BF_RESULTS = CLOSE_PAIR_HMM_ROOT / "Bf_test" / "results"
CPHMM_ISOLATE_RESULTS = CLOSE_PAIR_HMM_ROOT / "Isolate_test" / "results"


# ---------------------------------------------------------------------------
# Garud & Good 2019 QP SNV catalogs + MIDAS references
# ---------------------------------------------------------------------------
GG2019_ROOT = BOTEIN_ROOT / "GarudGood2019_snvs"
GG2019_SNV_FEATHER = GG2019_ROOT / "snvs_feather"
GG2019_REP_GENOMES = GG2019_ROOT / "midas_db_data" / "rep_genomes"


# ---------------------------------------------------------------------------
# UHGG isolate SNV pipeline data
# ---------------------------------------------------------------------------
UHGG_DATA_ROOT = BOTEIN_ROOT / "uhgg"
UHGG_DH_FORMAT = UHGG_DATA_ROOT / "dh_format"
UHGG_SNV_CATALOGUE = UHGG_DATA_ROOT / "snv_catalogue_compressed"
UHGG_REFERENCE_GENOMES = UHGG_DATA_ROOT / "reference_genomes"
UHGG_GENES = UHGG_DATA_ROOT / "genes"
UHGG_ISOLATE_SNVS = UHGG_DATA_ROOT / "isolate_snvs"
UHGG_METADATA = UHGG_DATA_ROOT / "genomes-nr_metadata.tsv"


# ---------------------------------------------------------------------------
# NCBI-isolate alignment SNV tables
# ---------------------------------------------------------------------------
NCBI_ISOLATES_ROOT = BOTEIN_ROOT / "ncbi_isolates"


# ---------------------------------------------------------------------------
# LiuGood 2024 precomputed analysis intermediates
# ---------------------------------------------------------------------------
LIUGOOD_ANALYSIS_ROOT = BOTEIN_ROOT / "LiuGood2024_files" / "zhiru_analysis"
LIUGOOD_PAIRWISE_CLONAL_FRACTION = LIUGOOD_ANALYSIS_ROOT / "pairwise_clonal_fraction"
LIUGOOD_CF_BETWEEN_HOSTS = LIUGOOD_PAIRWISE_CLONAL_FRACTION / "between_hosts"
LIUGOOD_CF_ISOLATES = LIUGOOD_PAIRWISE_CLONAL_FRACTION / "isolates"
LIUGOOD_CLOSELY_RELATED_ISOLATES = LIUGOOD_ANALYSIS_ROOT / "closely_related" / "isolates"


# ---------------------------------------------------------------------------
# Staphylococcus aureus isolate-alignment data (supplementary dN/dS figure)
# ---------------------------------------------------------------------------
#: Holds variants.npy, gene_ids.npy, Saureus.fasta (MSA, ~320 MB), div_mat.npy,
#: cf_mat.npy. Large + external; not vendored.
STAPH_DATA_DIR = _root(
    "DNDS_STAPH_DATA_DIR", BGOODLAB_ROOT / "bioinformatics" / "dNdS" / "staph"
)


# ---------------------------------------------------------------------------
# External command-line tools
# ---------------------------------------------------------------------------
MUMMER_ROOT = _root("DNDS_MUMMER_ROOT", TOOLS_ROOT / "mummer-4.0.0rc1")
NUCMER_BIN = MUMMER_ROOT / "nucmer"
DELTA_FILTER_BIN = MUMMER_ROOT / "delta-filter"
SHOW_SNPS_BIN = MUMMER_ROOT / "show-snps"
SHOW_COORDS_BIN = MUMMER_ROOT / "show-coords"
DATASETS_BIN = _root("DNDS_DATASETS_BIN", TOOLS_ROOT / "datasets")


# ---------------------------------------------------------------------------
# Figure reproduction (in-repo data + outputs)
#
# Names mirror the original dNdS_dynamics repo's ``dNdS_analysis/config.py`` so
# the ported figure code (paper_figures/, dnds_dynamics/figures/) runs with only
# its top-level import rewired. The cached per-pair dN/dS CSVs are vendored under
# ``data/``; ``gut_microbiome_transfers.csv`` (44 MB, the Liu & Good 2024
# supplementary table pbio.3002472.s003) is kept locally but git-ignored.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
data_path = _root("DNDS_DATA_DIR", REPO_ROOT / "data")
fig_dat_path = data_path / "figure_data"
fig_path = _root("DNDS_FIG_DIR", REPO_ROOT / "figures")
table_path = REPO_ROOT / "tables"

blacklist_species = ["Lachnospiraceae_bacterium_51870"]
fully_recombined_threshold = 0.05

# Aliases used by the ported figure utils (point at this repo's canonical roots).
snv_data_path = GG2019_SNV_FEATHER
ref_genome_path = GG2019_REP_GENOMES
LiuGood2024_path = LIUGOOD_ANALYSIS_ROOT
# Per-species between-host identical-fraction matrices (Liu & Good 2024); the
# species used by the clonal-SNV-prevalence figures are vendored under data/.
identical_fraction_path = _root("DNDS_IDENTICAL_FRACTION_DIR", REPO_ROOT / "data" / "identical_fraction")


# ---------------------------------------------------------------------------
# Import shims for sibling code repos (editable-install aware)
# ---------------------------------------------------------------------------
def _ensure_importable(module_name: str, repo_root: Path) -> None:
    """Make ``module_name`` importable.

    Prefer an already-installed/editable package; only if it is not found do we
    insert ``repo_root`` on ``sys.path``. This means ``pip install -e`` of the
    dependency makes the fallback a no-op automatically.
    """
    import sys

    if importlib.util.find_spec(module_name) is not None:
        return
    root = str(repo_root)
    if root not in sys.path:
        sys.path.insert(0, root)


def ensure_cphmm_importable() -> None:
    """Make the ``cphmm`` package (close_pair_hmm) importable.

    ``cphmm`` is normally a ``pip install -e`` editable package in the env, so
    this is a no-op. The sys.path fallback (repo root) is kept only for a dev
    checkout without the editable install. ``infer_pipelines`` now lives inside
    the package (``cphmm.infer_pipelines``), so ensuring ``cphmm`` is enough.
    """
    _ensure_importable("cphmm", CLOSE_PAIR_HMM_ROOT)
