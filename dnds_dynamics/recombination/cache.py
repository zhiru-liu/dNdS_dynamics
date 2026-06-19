"""Recombination event cache helpers for isolate dN/dS analyses.

The cache is intentionally table-first so both the legacy Python2 CP-HMM
outputs and future Python3 ``close_pair_hmm`` runs can write the same contract.

Per accession, the cache directory contains:

- ``recombination_events.parquet``: one row per pair-specific event.
- ``recombination_pairs.parquet``: one row per analyzed pair.
- ``recombination_cache_metadata.json``: source paths and conversion summary.

The event table uses reference contig coordinates as the primary downstream mask
coordinate because the new isolate SNV tables are indexed by ``(Contig,
Location)``. Legacy all-core coordinates are kept as provenance and for
compatibility with old DH-style code.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import pickle
import sys
import types
from typing import Iterable

import numpy as np
import pandas as pd


from .. import config

DEFAULT_DH_ROOT = config.UHGG_DH_FORMAT
DEFAULT_LEGACY_TRANSFER_ROOT = config.LIUGOOD_CLOSELY_RELATED_ISOLATES
DEFAULT_RECOMBINATION_CACHE_ROOT = config.UHGG_ISOLATE_SNVS
DEFAULT_NEW_CPHMM_RESULTS_ROOT = config.CPHMM_ISOLATE_RESULTS
NEW_CPHMM_SOURCE = "new_cphmm_isolate"

CACHE_SCHEMA_VERSION = "isolate-recombination-cache-v1"
PAIR_SEPARATOR = "|"

EVENT_COLUMNS = [
    "schema_version",
    "accession",
    "event_id",
    "source",
    "source_event_index",
    "pair_id",
    "pair_i",
    "pair_j",
    "sample_1",
    "sample_2",
    "event_order",
    "hmm_block_start",
    "hmm_block_end",
    "hmm_block_length",
    "hmm_event_type",
    "core_start",
    "core_end",
    "core_transfer_length",
    "reference_contig",
    "reference_start",
    "reference_end",
    "clonal_divergence",
    "clonal_fraction",
    "transfer_divergence_synonymous",
    "transfer_divergence",
    "dedup_representative",
    "dedup_start_bin",
    "dedup_end_bin",
]

PAIR_COLUMNS = [
    "schema_version",
    "accession",
    "pair_id",
    "pair_i",
    "pair_j",
    "sample_1",
    "sample_2",
    "genome_length",
    "clonal_length",
    "clonal_divergence",
    "naive_clonal_divergence",
    "expected_clonal_snps",
    "transfer_count",
    "normalized_transfer_count",
    "total_transfer_length",
    "clonal_fraction",
    "event_count",
    "dedup_event_count",
]

EVENT_KEY_COLUMNS = [
    "sample_1",
    "sample_2",
    "core_start",
    "core_end",
    "reference_contig",
    "reference_start",
    "reference_end",
]


@dataclass(frozen=True)
class RecombinationCachePaths:
    """Resolved cache paths for one accession."""

    accession: str
    cache_dir: Path
    events: Path
    pairs: Path
    metadata: Path

    @classmethod
    def from_root(
        cls,
        accession: str,
        root: Path | str = DEFAULT_RECOMBINATION_CACHE_ROOT,
        table_format: str = "parquet",
    ) -> "RecombinationCachePaths":
        cache_dir = Path(root) / accession
        return cls(
            accession=accession,
            cache_dir=cache_dir,
            events=cache_dir / f"recombination_events.{table_format}",
            pairs=cache_dir / f"recombination_pairs.{table_format}",
            metadata=cache_dir / "recombination_cache_metadata.json",
        )

    def exists(self) -> bool:
        return self.events.exists()


def decode_scalar(value):
    """Normalize old pickle scalar values to Python/string values."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def decode_array(values: np.ndarray) -> np.ndarray:
    """Decode byte-string arrays from legacy npy files."""
    if values.dtype.kind == "S":
        return values.astype(str)
    if values.dtype.kind == "O" and len(values) and isinstance(values[0], bytes):
        return np.asarray([x.decode("utf-8") for x in values])
    return values.astype(str) if values.dtype.kind in {"O", "U"} else values


def _install_pandas_pickle_compat() -> None:
    """Provide old pandas index modules needed by some Python2-era pickles."""
    module_name = "pandas.core.indexes.numeric"
    if module_name in sys.modules:
        return
    mod = types.ModuleType(module_name)
    mod.Int64Index = pd.Index
    mod.UInt64Index = pd.Index
    mod.Float64Index = pd.Index
    sys.modules[module_name] = mod


def load_legacy_pickle(path: Path | str):
    """Load a legacy pickle with Python2/pandas compatibility shims."""
    _install_pandas_pickle_compat()
    with Path(path).open("rb") as handle:
        return pickle.load(handle, encoding="latin1")


def load_good_genomes(accession: str, dh_root: Path | str = DEFAULT_DH_ROOT) -> np.ndarray:
    path = Path(dh_root) / accession / "good_genomes.npy"
    if not path.exists():
        raise FileNotFoundError(f"good_genomes.npy not found: {path}")
    return decode_array(np.load(path, allow_pickle=True))


def parse_pair_indices(pair_value) -> tuple[int, int]:
    """Return integer pair indices from tuple/list/array-like legacy values."""
    if isinstance(pair_value, str):
        text = pair_value.strip().strip("()")
        parts = [part.strip() for part in text.split(",") if part.strip()]
        if len(parts) != 2:
            raise ValueError(f"Could not parse pair value {pair_value!r}")
        return int(parts[0]), int(parts[1])
    pair = tuple(pair_value)
    if len(pair) != 2:
        raise ValueError(f"Expected two pair indices, got {pair_value!r}")
    return int(pair[0]), int(pair[1])


def make_pair_id(sample_1: str, sample_2: str) -> str:
    return f"{sample_1}{PAIR_SEPARATOR}{sample_2}"


def _to_nullable_int(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def _to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(float)


def _optional_series(df: pd.DataFrame, column: str, default=pd.NA) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series(default, index=df.index)


def _event_key_frame(events: pd.DataFrame) -> pd.DataFrame:
    keys = events.loc[:, EVENT_KEY_COLUMNS].copy()
    for col in ["sample_1", "sample_2", "reference_contig"]:
        keys[col] = keys[col].astype(str)
    for col in ["core_start", "core_end", "reference_start", "reference_end"]:
        keys[col] = _to_nullable_int(keys[col])
    return keys


def _dedup_key_frame(dedup_df: pd.DataFrame) -> pd.DataFrame:
    renamed = dedup_df.rename(
        columns={
            "Sample 1": "sample_1",
            "Sample 2": "sample_2",
            "Core genome start loc": "core_start",
            "Core genome end loc": "core_end",
            "Reference contig": "reference_contig",
            "Reference genome start loc": "reference_start",
            "Reference genome end loc": "reference_end",
            "Start bin": "dedup_start_bin",
            "End bin": "dedup_end_bin",
        }
    )
    keys = renamed.loc[:, EVENT_KEY_COLUMNS].copy()
    for col in ["sample_1", "sample_2", "reference_contig"]:
        keys[col] = keys[col].astype(str)
    for col in ["core_start", "core_end", "reference_start", "reference_end"]:
        keys[col] = _to_nullable_int(keys[col])
    if "dedup_start_bin" in renamed.columns:
        keys["dedup_start_bin"] = _to_nullable_int(renamed["dedup_start_bin"])
    else:
        keys["dedup_start_bin"] = pd.Series(pd.NA, index=keys.index, dtype="Int64")
    if "dedup_end_bin" in renamed.columns:
        keys["dedup_end_bin"] = _to_nullable_int(renamed["dedup_end_bin"])
    else:
        keys["dedup_end_bin"] = pd.Series(pd.NA, index=keys.index, dtype="Int64")
    return keys


def annotate_dedup_representatives(
    events: pd.DataFrame,
    dedup_csv_path: Path | str,
) -> tuple[pd.DataFrame, dict]:
    """Mark event rows that are present in the old deduplicated CSV."""
    dedup_df = pd.read_csv(dedup_csv_path)
    dedup_keys = _dedup_key_frame(dedup_df)
    dedup_keys = dedup_keys.drop_duplicates(EVENT_KEY_COLUMNS)
    dedup_keys["dedup_representative"] = True

    annotated = events.merge(dedup_keys, on=EVENT_KEY_COLUMNS, how="left", suffixes=("", "_dedup"))
    annotated["dedup_representative"] = annotated["dedup_representative_dedup"].notna()
    annotated["dedup_start_bin"] = _to_nullable_int(annotated["dedup_start_bin_dedup"])
    annotated["dedup_end_bin"] = _to_nullable_int(annotated["dedup_end_bin_dedup"])
    annotated.drop(
        columns=[
            "dedup_representative_dedup",
            "dedup_start_bin_dedup",
            "dedup_end_bin_dedup",
        ],
        inplace=True,
    )

    event_keys = set(map(tuple, _event_key_frame(events).to_numpy(dtype=object)))
    dedup_key_tuples = list(map(tuple, dedup_keys.loc[:, EVENT_KEY_COLUMNS].to_numpy(dtype=object)))
    matched = sum(key in event_keys for key in dedup_key_tuples)
    summary = {
        "dedup_csv_path": str(dedup_csv_path),
        "dedup_csv_rows": int(len(dedup_df)),
        "dedup_unique_keys": int(len(dedup_keys)),
        "dedup_keys_matched_in_events": int(matched),
        "dedup_keys_unmatched": int(len(dedup_key_tuples) - matched),
        "dedup_representative_events": int(annotated["dedup_representative"].sum()),
    }
    return annotated, summary


def legacy_all_transfers_to_events(
    accession: str,
    all_transfers_path: Path | str,
    *,
    dh_root: Path | str = DEFAULT_DH_ROOT,
    dedup_csv_path: Path | str | None = None,
    source: str = "legacy_all_transfers_pickle",
) -> tuple[pd.DataFrame, dict]:
    """Convert a legacy ``_all_transfers.pickle`` into cache event rows."""
    raw = load_legacy_pickle(all_transfers_path)
    if not isinstance(raw, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame in {all_transfers_path}, got {type(raw)!r}")
    source_event_index = raw.index.to_numpy()
    raw = raw.reset_index(drop=True)

    good_genomes = load_good_genomes(accession, dh_root)
    pair_indices = raw["pairs"].map(parse_pair_indices)
    pair_i = pair_indices.map(lambda pair: pair[0])
    pair_j = pair_indices.map(lambda pair: pair[1])
    max_idx = max(pair_i.max(), pair_j.max()) if len(raw) else -1
    if max_idx >= len(good_genomes):
        raise IndexError(
            f"{accession}: legacy pair index {max_idx} exceeds good_genomes length {len(good_genomes)}"
        )

    sample_1 = pair_i.map(lambda idx: str(good_genomes[idx]))
    sample_2 = pair_j.map(lambda idx: str(good_genomes[idx]))
    pair_id = [make_pair_id(a, b) for a, b in zip(sample_1, sample_2)]

    events = pd.DataFrame(index=np.arange(len(raw)))
    events["schema_version"] = CACHE_SCHEMA_VERSION
    events["accession"] = accession
    events["source"] = source
    events["source_event_index"] = list(map(decode_scalar, source_event_index))
    events["pair_i"] = _to_nullable_int(pair_i)
    events["pair_j"] = _to_nullable_int(pair_j)
    events["sample_1"] = sample_1.astype(str)
    events["sample_2"] = sample_2.astype(str)
    events["pair_id"] = pair_id
    events["event_order"] = events.groupby("pair_id").cumcount().astype("Int64")
    events["event_id"] = [
        f"{accession}{PAIR_SEPARATOR}{int(i)}{PAIR_SEPARATOR}{int(j)}{PAIR_SEPARATOR}{int(order):06d}"
        for i, j, order in zip(events["pair_i"], events["pair_j"], events["event_order"])
    ]

    events["hmm_block_start"] = _to_nullable_int(_optional_series(raw, "starts"))
    events["hmm_block_end"] = _to_nullable_int(_optional_series(raw, "ends"))
    events["hmm_block_length"] = _to_nullable_int(_optional_series(raw, "lengths"))
    events["hmm_event_type"] = _to_nullable_int(_optional_series(raw, "types"))
    events["core_start"] = _to_nullable_int(_optional_series(raw, "core genome starts"))
    events["core_end"] = _to_nullable_int(_optional_series(raw, "core genome ends"))
    events["core_transfer_length"] = _to_nullable_int(
        _optional_series(raw, "transfer lengths (core genome)")
    )
    events["reference_contig"] = _optional_series(raw, "contigs").map(decode_scalar).astype(str)
    events["reference_start"] = _to_nullable_int(_optional_series(raw, "reference genome starts"))
    events["reference_end"] = _to_nullable_int(_optional_series(raw, "reference genome ends"))
    events["clonal_divergence"] = _to_float(_optional_series(raw, "clonal divergence"))
    events["clonal_fraction"] = _to_float(_optional_series(raw, "clonal fraction"))
    events["transfer_divergence_synonymous"] = _to_float(
        _optional_series(raw, "synonymous divergences")
    )
    events["transfer_divergence"] = _to_float(_optional_series(raw, "divergences"))
    events["dedup_representative"] = False
    events["dedup_start_bin"] = pd.Series(pd.NA, index=events.index, dtype="Int64")
    events["dedup_end_bin"] = pd.Series(pd.NA, index=events.index, dtype="Int64")

    events = events.loc[:, EVENT_COLUMNS]
    summary = {
        "accession": accession,
        "all_transfers_path": str(all_transfers_path),
        "event_rows": int(len(events)),
        "event_bearing_pairs": int(events["pair_id"].nunique()),
        "schema_version": CACHE_SCHEMA_VERSION,
    }

    if dedup_csv_path is not None and Path(dedup_csv_path).exists():
        events, dedup_summary = annotate_dedup_representatives(events, dedup_csv_path)
        summary.update(dedup_summary)
        events = events.loc[:, EVENT_COLUMNS]

    return events, summary


def legacy_thirdpass_to_pairs(
    accession: str,
    thirdpass_path: Path | str | None,
    events: pd.DataFrame,
    *,
    dh_root: Path | str = DEFAULT_DH_ROOT,
) -> pd.DataFrame:
    """Build the pair-level cache table from legacy third-pass output."""
    good_genomes = load_good_genomes(accession, dh_root)
    event_counts = events.groupby("pair_id").agg(
        event_count=("event_id", "count"),
        dedup_event_count=("dedup_representative", "sum"),
    )

    if thirdpass_path is not None and Path(thirdpass_path).exists():
        raw = load_legacy_pickle(thirdpass_path)
        if not isinstance(raw, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame in {thirdpass_path}, got {type(raw)!r}")
        raw = raw.reset_index(drop=True)
        pair_indices = raw["pairs"].map(parse_pair_indices)
        pairs = pd.DataFrame(index=np.arange(len(raw)))
        pairs["pair_i"] = _to_nullable_int(pair_indices.map(lambda pair: pair[0]))
        pairs["pair_j"] = _to_nullable_int(pair_indices.map(lambda pair: pair[1]))
        pairs["sample_1"] = pairs["pair_i"].map(lambda idx: str(good_genomes[int(idx)]))
        pairs["sample_2"] = pairs["pair_j"].map(lambda idx: str(good_genomes[int(idx)]))
        pairs["pair_id"] = [make_pair_id(a, b) for a, b in zip(pairs["sample_1"], pairs["sample_2"])]
        pairs["genome_length"] = _to_nullable_int(_optional_series(raw, "genome lengths"))
        pairs["clonal_length"] = _to_nullable_int(_optional_series(raw, "clonal lengths"))
        pairs["clonal_divergence"] = _to_float(_optional_series(raw, "clonal divs"))
        pairs["naive_clonal_divergence"] = _to_float(_optional_series(raw, "naive clonal divs"))
        pairs["expected_clonal_snps"] = _to_float(_optional_series(raw, "expected clonal snps"))
        pairs["transfer_count"] = _to_nullable_int(_optional_series(raw, "transfer counts"))
        pairs["normalized_transfer_count"] = _to_float(
            _optional_series(raw, "normalized transfer counts")
        )
        pairs["total_transfer_length"] = _to_float(_optional_series(raw, "total transfer lengths"))
        pairs["clonal_fraction"] = _to_float(_optional_series(raw, "clonal fractions"))
    else:
        grouped = events.groupby(["pair_id", "pair_i", "pair_j", "sample_1", "sample_2"], dropna=False)
        pairs = grouped.agg(
            clonal_divergence=("clonal_divergence", "first"),
            clonal_fraction=("clonal_fraction", "first"),
        ).reset_index()
        for col in [
            "genome_length",
            "clonal_length",
            "naive_clonal_divergence",
            "expected_clonal_snps",
            "transfer_count",
            "normalized_transfer_count",
            "total_transfer_length",
        ]:
            pairs[col] = pd.NA

    pairs = pairs.merge(event_counts, left_on="pair_id", right_index=True, how="left")
    pairs["event_count"] = _to_nullable_int(pairs["event_count"].fillna(0))
    pairs["dedup_event_count"] = _to_nullable_int(pairs["dedup_event_count"].fillna(0))
    pairs["schema_version"] = CACHE_SCHEMA_VERSION
    pairs["accession"] = accession
    return pairs.loc[:, PAIR_COLUMNS]


def build_legacy_recombination_cache(
    accession: str,
    *,
    legacy_root: Path | str = DEFAULT_LEGACY_TRANSFER_ROOT,
    dh_root: Path | str = DEFAULT_DH_ROOT,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Build event and pair cache tables from the old isolate output directory."""
    legacy_root = Path(legacy_root)
    all_transfers = legacy_root / f"{accession}_all_transfers.pickle"
    thirdpass = legacy_root / f"{accession}_thirdpass.pickle"
    dedup_csv = legacy_root / f"{accession}_all_transfers_dedup.csv"
    events, metadata = legacy_all_transfers_to_events(
        accession,
        all_transfers,
        dh_root=dh_root,
        dedup_csv_path=dedup_csv if dedup_csv.exists() else None,
    )
    pairs = legacy_thirdpass_to_pairs(
        accession,
        thirdpass if thirdpass.exists() else None,
        events,
        dh_root=dh_root,
    )
    metadata.update(
        {
            "legacy_root": str(legacy_root),
            "thirdpass_path": str(thirdpass) if thirdpass.exists() else None,
            "pair_rows": int(len(pairs)),
            "pair_rows_with_events": int((pairs["event_count"] > 0).sum()),
        }
    )
    return events, pairs, metadata


def _lex_sort_pair(genome1, genome2):
    a, b = str(genome1), str(genome2)
    if a <= b:
        return a, b, False
    return b, a, True


def new_cphmm_inference_summary_to_pairs(
    accession: str,
    inference_df: pd.DataFrame,
    transfer_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build the pair-level cache table from a new-CPHMM inference summary CSV."""
    if inference_df.empty:
        return pd.DataFrame(columns=PAIR_COLUMNS)

    inference_df = inference_df.copy()
    sorted_pairs = [_lex_sort_pair(g1, g2) for g1, g2 in zip(inference_df["genome1"], inference_df["genome2"])]
    inference_df["sample_1"] = [p[0] for p in sorted_pairs]
    inference_df["sample_2"] = [p[1] for p in sorted_pairs]
    inference_df["pair_id"] = [make_pair_id(a, b) for a, b in zip(inference_df["sample_1"], inference_df["sample_2"])]

    # Per-pair event counts, total transfer length (reference-coords; right-inclusive).
    if transfer_df is None or transfer_df.empty:
        event_counts = pd.Series(0, index=inference_df["pair_id"], name="event_count")
        total_lens = pd.Series(0, index=inference_df["pair_id"], name="total_transfer_length")
    else:
        tdf = transfer_df.copy()
        t_sorted = [_lex_sort_pair(g1, g2) for g1, g2 in zip(tdf["genome1"], tdf["genome2"])]
        tdf["sample_1"] = [p[0] for p in t_sorted]
        tdf["sample_2"] = [p[1] for p in t_sorted]
        tdf["pair_id"] = [make_pair_id(a, b) for a, b in zip(tdf["sample_1"], tdf["sample_2"])]
        grouped = tdf.groupby("pair_id")
        event_counts = grouped.size().rename("event_count")
        total_lens = (
            (tdf["end_site"].astype(np.int64) - tdf["start_site"].astype(np.int64) + 1)
            .groupby(tdf["pair_id"]).sum().rename("total_transfer_length")
        )

    pairs = pd.DataFrame(index=np.arange(len(inference_df)))
    pairs["schema_version"] = CACHE_SCHEMA_VERSION
    pairs["accession"] = accession
    pairs["pair_id"] = inference_df["pair_id"].to_numpy()
    pairs["pair_i"] = pd.Series(pd.NA, index=pairs.index, dtype="Int64")
    pairs["pair_j"] = pd.Series(pd.NA, index=pairs.index, dtype="Int64")
    pairs["sample_1"] = inference_df["sample_1"].to_numpy()
    pairs["sample_2"] = inference_df["sample_2"].to_numpy()
    pairs["genome_length"] = _to_nullable_int(inference_df["genome_len"])
    pairs["clonal_length"] = _to_nullable_int(inference_df["clonal_len"])
    pairs["clonal_divergence"] = _to_float(inference_df["est_div"])
    pairs["naive_clonal_divergence"] = _to_float(inference_df["naive_div"])
    pairs["expected_clonal_snps"] = pd.Series(pd.NA, index=pairs.index, dtype="Float64")
    pairs["transfer_count"] = _to_nullable_int(
        pairs["pair_id"].map(event_counts).fillna(0)
    )
    pairs["normalized_transfer_count"] = pd.Series(pd.NA, index=pairs.index, dtype="Float64")
    pairs["total_transfer_length"] = _to_float(
        pairs["pair_id"].map(total_lens).fillna(0)
    )
    pairs["clonal_fraction"] = pairs["clonal_length"].astype(float) / pairs["genome_length"].astype(float)
    pairs["event_count"] = pairs["transfer_count"].copy()
    # No dedup step in the new pipeline; mark every event as a representative.
    pairs["dedup_event_count"] = pairs["event_count"].copy()
    return pairs.loc[:, PAIR_COLUMNS]


def new_cphmm_transfer_summary_to_events(
    accession: str,
    transfer_df: pd.DataFrame,
    pairs: pd.DataFrame,
    *,
    source: str = NEW_CPHMM_SOURCE,
) -> pd.DataFrame:
    """Build the event-level cache table from a new-CPHMM transfer summary CSV."""
    if transfer_df.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)

    tdf = transfer_df.copy().reset_index(drop=False).rename(columns={"index": "source_event_index"})
    sorted_pairs = [_lex_sort_pair(g1, g2) for g1, g2 in zip(tdf["genome1"], tdf["genome2"])]
    tdf["sample_1"] = [p[0] for p in sorted_pairs]
    tdf["sample_2"] = [p[1] for p in sorted_pairs]
    tdf["pair_id"] = [make_pair_id(a, b) for a, b in zip(tdf["sample_1"], tdf["sample_2"])]
    tdf = tdf.sort_values(["pair_id", "contig", "start_site", "end_site"], kind="stable").reset_index(drop=True)

    events = pd.DataFrame(index=np.arange(len(tdf)))
    events["schema_version"] = CACHE_SCHEMA_VERSION
    events["accession"] = accession
    events["source"] = source
    events["source_event_index"] = tdf["source_event_index"].astype("Int64")
    events["pair_i"] = pd.Series(pd.NA, index=events.index, dtype="Int64")
    events["pair_j"] = pd.Series(pd.NA, index=events.index, dtype="Int64")
    events["sample_1"] = tdf["sample_1"].astype(str)
    events["sample_2"] = tdf["sample_2"].astype(str)
    events["pair_id"] = tdf["pair_id"].astype(str)
    events["event_order"] = events.groupby("pair_id").cumcount().astype("Int64")
    events["event_id"] = [
        f"{accession}{PAIR_SEPARATOR}{s1}{PAIR_SEPARATOR}{s2}{PAIR_SEPARATOR}{int(order):06d}"
        for s1, s2, order in zip(events["sample_1"], events["sample_2"], events["event_order"])
    ]

    events["hmm_block_start"] = _to_nullable_int(tdf["block_start"])
    events["hmm_block_end"] = _to_nullable_int(tdf["block_end"])
    # block_end is right-exclusive (per CPHMM rerun handoff), so length = end - start.
    events["hmm_block_length"] = (
        _to_nullable_int(tdf["block_end"]) - _to_nullable_int(tdf["block_start"])
    )
    events["hmm_event_type"] = _to_nullable_int(tdf["types"])
    # Refactor does not emit legacy all-core coordinates.
    events["core_start"] = pd.Series(pd.NA, index=events.index, dtype="Int64")
    events["core_end"] = pd.Series(pd.NA, index=events.index, dtype="Int64")
    events["core_transfer_length"] = pd.Series(pd.NA, index=events.index, dtype="Int64")
    events["reference_contig"] = tdf["contig"].astype(str)
    events["reference_start"] = _to_nullable_int(tdf["start_site"])
    events["reference_end"] = _to_nullable_int(tdf["end_site"])

    # Broadcast pair-level clonal divergence/fraction onto events for parity with the legacy schema.
    pair_lookup = pairs.set_index("pair_id")[["clonal_divergence", "clonal_fraction"]]
    events["clonal_divergence"] = events["pair_id"].map(pair_lookup["clonal_divergence"]).astype(float)
    events["clonal_fraction"] = events["pair_id"].map(pair_lookup["clonal_fraction"]).astype(float)
    events["transfer_divergence_synonymous"] = pd.Series(np.nan, index=events.index, dtype=float)
    events["transfer_divergence"] = pd.Series(np.nan, index=events.index, dtype=float)
    events["dedup_representative"] = True  # no dedup in new pipeline; every event is a "representative".
    events["dedup_start_bin"] = pd.Series(pd.NA, index=events.index, dtype="Int64")
    events["dedup_end_bin"] = pd.Series(pd.NA, index=events.index, dtype="Int64")

    return events.loc[:, EVENT_COLUMNS]


def build_new_cphmm_recombination_cache(
    accession: str,
    *,
    results_dir: Path | str = DEFAULT_NEW_CPHMM_RESULTS_ROOT,
    suffix: str = "all_pairs",
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Build event and pair cache tables from new-CPHMM CSV outputs."""
    results_dir = Path(results_dir)
    pair_path = results_dir / f"{accession}__{suffix}__inference_summary.csv"
    transfer_path = results_dir / f"{accession}__{suffix}__transfer_summary.csv"
    if not pair_path.exists():
        raise FileNotFoundError(f"Missing inference summary: {pair_path}")
    if not transfer_path.exists():
        raise FileNotFoundError(f"Missing transfer summary: {transfer_path}")

    inference_df = pd.read_csv(pair_path, dtype={"genome1": str, "genome2": str})
    transfer_df = pd.read_csv(transfer_path, dtype={"genome1": str, "genome2": str, "contig": str})

    pairs = new_cphmm_inference_summary_to_pairs(accession, inference_df, transfer_df)
    events = new_cphmm_transfer_summary_to_events(accession, transfer_df, pairs)

    metadata = {
        "accession": accession,
        "source": NEW_CPHMM_SOURCE,
        "results_dir": str(results_dir),
        "suffix": suffix,
        "inference_summary_path": str(pair_path),
        "transfer_summary_path": str(transfer_path),
        "schema_version": CACHE_SCHEMA_VERSION,
        "event_rows": int(len(events)),
        "pair_rows": int(len(pairs)),
        "pair_rows_with_events": int((pairs["event_count"] > 0).sum()),
        "event_bearing_pairs": int(events["pair_id"].nunique()) if len(events) else 0,
    }
    return events, pairs, metadata


def save_recombination_cache(
    events: pd.DataFrame,
    pairs: pd.DataFrame,
    metadata: dict,
    paths: RecombinationCachePaths,
) -> None:
    paths.cache_dir.mkdir(parents=True, exist_ok=True)
    events.to_parquet(paths.events, index=False)
    pairs.to_parquet(paths.pairs, index=False)
    metadata = dict(metadata)
    metadata["schema_version"] = CACHE_SCHEMA_VERSION
    metadata["events_path"] = str(paths.events)
    metadata["pairs_path"] = str(paths.pairs)
    paths.metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def load_recombination_events(path_or_dir: Path | str) -> pd.DataFrame:
    path = Path(path_or_dir)
    if path.is_dir():
        path = path / "recombination_events.parquet"
    events = pd.read_parquet(path)
    if "schema_version" not in events.columns:
        raise ValueError(f"{path} does not look like a recombination event cache")
    return events


def load_recombination_pairs(path_or_dir: Path | str) -> pd.DataFrame:
    path = Path(path_or_dir)
    if path.is_dir():
        path = path / "recombination_pairs.parquet"
    return pd.read_parquet(path)


def events_for_pair(
    events: pd.DataFrame,
    sample_1: str,
    sample_2: str,
    *,
    include_reverse: bool = True,
    dedup_only: bool = False,
) -> pd.DataFrame:
    sample_1 = str(sample_1)
    sample_2 = str(sample_2)
    mask = (events["sample_1"].astype(str) == sample_1) & (events["sample_2"].astype(str) == sample_2)
    if include_reverse:
        mask |= (events["sample_1"].astype(str) == sample_2) & (
            events["sample_2"].astype(str) == sample_1
        )
    sub = events.loc[mask].copy()
    if dedup_only:
        sub = sub[sub["dedup_representative"]]
    return sub.sort_values(["sample_1", "sample_2", "reference_contig", "reference_start", "reference_end"])


def recombination_mask_from_events(index: pd.MultiIndex, events: pd.DataFrame) -> pd.Series:
    """Build an inclusive reference-interval mask over a ``(Contig, Location)`` index."""
    if not isinstance(index, pd.MultiIndex) or "Contig" not in index.names or "Location" not in index.names:
        raise ValueError("index must be a MultiIndex with Contig and Location levels")
    mask = np.zeros(len(index), dtype=bool)
    if events.empty:
        return pd.Series(mask, index=index, name="recombination")

    contigs = index.get_level_values("Contig").astype(str).to_numpy()
    locations = index.get_level_values("Location").to_numpy(dtype=int)
    valid = events.dropna(subset=["reference_contig", "reference_start", "reference_end"])
    for contig, grouped in valid.groupby("reference_contig"):
        contig_mask = contigs == str(contig)
        if not contig_mask.any():
            continue
        contig_positions = np.flatnonzero(contig_mask)
        contig_locs = locations[contig_positions]
        for _, row in grouped.iterrows():
            start = int(row["reference_start"])
            end = int(row["reference_end"])
            if end < start:
                start, end = end, start
            in_event = (contig_locs >= start) & (contig_locs <= end)
            mask[contig_positions[in_event]] = True
    return pd.Series(mask, index=index, name="recombination")


def verify_events_against_dedup_csv(
    events: pd.DataFrame,
    dedup_csv_path: Path | str,
    *,
    max_examples: int = 5,
) -> dict:
    """Return a concise verification summary against the existing dedup CSV."""
    dedup_df = pd.read_csv(dedup_csv_path)
    dedup_keys = _dedup_key_frame(dedup_df)
    event_keys = _event_key_frame(events)
    event_key_set = set(map(tuple, event_keys.to_numpy(dtype=object)))

    examples = []
    matched = 0
    for idx, key_row in dedup_keys.iterrows():
        key = tuple(key_row.loc[EVENT_KEY_COLUMNS].to_numpy(dtype=object))
        if key not in event_key_set:
            continue
        matched += 1
        if len(examples) < max_examples:
            event_match = events.loc[
                (
                    (events["sample_1"].astype(str) == str(key_row["sample_1"]))
                    & (events["sample_2"].astype(str) == str(key_row["sample_2"]))
                    & (_to_nullable_int(events["core_start"]) == key_row["core_start"])
                    & (_to_nullable_int(events["core_end"]) == key_row["core_end"])
                    & (events["reference_contig"].astype(str) == str(key_row["reference_contig"]))
                    & (_to_nullable_int(events["reference_start"]) == key_row["reference_start"])
                    & (_to_nullable_int(events["reference_end"]) == key_row["reference_end"])
                )
            ].head(1)
            event_id = event_match["event_id"].iloc[0] if not event_match.empty else None
            examples.append(
                {
                    "dedup_csv_row": int(idx),
                    "event_id": event_id,
                    "sample_1": str(key_row["sample_1"]),
                    "sample_2": str(key_row["sample_2"]),
                    "reference_interval": (
                        f"{key_row['reference_contig']}:{int(key_row['reference_start'])}-"
                        f"{int(key_row['reference_end'])}"
                    ),
                    "core_interval": f"{int(key_row['core_start'])}-{int(key_row['core_end'])}",
                }
            )

    return {
        "dedup_csv_path": str(dedup_csv_path),
        "dedup_csv_rows": int(len(dedup_df)),
        "dedup_rows_matched": int(matched),
        "dedup_rows_unmatched": int(len(dedup_df) - matched),
        "all_dedup_rows_matched": bool(matched == len(dedup_df)),
        "examples": examples,
    }
