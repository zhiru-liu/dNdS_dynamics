"""Mutation-spectrum estimation from LiuGood2024 SNV catalogs.

Phase 1 of the mutation-spectrum-vs-dNdS analysis. Given a SNVHelper-style
object (annotated with site types and carrying a biallelic SNV table with
``Ref``, ``Major``, ``Alt`` columns), this module estimates the six-class
strand-folded mutation spectrum on 4D sites using *rare* polymorphisms across
hosts as a proxy for de novo mutations.

Design choices:

- The Major → Alt direction is taken as the derived mutation. For sites where
  ``Alt`` is rare this is essentially equivalent to using an outgroup; the rare
  filter is what justifies the assumption.
- 4D sites are identified on the major-polarized reference (already the case
  for the helper's site-annotation cache), so the codon context matches the
  inferred ancestral state.
- Opportunity is the count of 4D sites in the major-polarized reference,
  bucketed by source base and strand-folded to pyrimidine anchors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C"}
CANONICAL_CLASSES = ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G"]
TRANSITIONS = {"C>T", "T>C"}


def fold_to_canonical(major: str, alt: str) -> str:
    """Return the pyrimidine-anchored mutation class for a (Major, Alt) pair."""
    if major in {"A", "G"}:
        major = COMPLEMENT[major]
        alt = COMPLEMENT[alt]
    return f"{major}>{alt}"


def _alt_counts(snvs_at_sites: pd.DataFrame, sample_cols: list[str]) -> tuple[pd.Series, pd.Series]:
    """Return (alt_count, covered_count) per row across sample columns."""
    arr = snvs_at_sites[sample_cols].to_numpy()
    alt = (arr == 1).sum(axis=1)
    covered = (arr != 255).sum(axis=1)
    idx = snvs_at_sites.index
    return pd.Series(alt, index=idx, name="alt_count"), pd.Series(covered, index=idx, name="covered_count")


@dataclass(frozen=True)
class SpectrumResult:
    species: str
    spectrum: pd.DataFrame  # one row per canonical class
    summary: dict


def compute_4D_spectrum(
    biallelic_snvs: pd.DataFrame,
    mut_df: pd.DataFrame,
    sample_cols: list[str],
    *,
    species: str = "",
    alt_count_max: int = 1,
    require_min_covered: int = 4,
) -> SpectrumResult:
    """Estimate the strand-folded mutation spectrum on 4D sites.

    Parameters
    ----------
    biallelic_snvs
        Biallelic SNV table with index ``(Contig, Location)`` and columns
        ``Ref``, ``Major``, ``Alt``, plus one column per sample carrying 0/1/255.
    mut_df
        Site-annotation table with index ``(Contig, Location)`` and at least a
        ``Site Type`` column. The major-polarized reference base used to assign
        site types should be reflected by ``Ref Base`` (used as a sanity check).
    sample_cols
        Sample column names within ``biallelic_snvs``.
    species
        Optional label propagated into the result.
    alt_count_max
        Maximum Alt allele count (across covered samples) for a SNV to be
        treated as a rare polymorphism. Default 1 (singletons).
    require_min_covered
        Drop sites whose covered-sample count falls below this threshold. Guards
        against poorly-covered sites with noisy allele calls.
    """
    site_type = mut_df["Site Type"].reindex(biallelic_snvs.index, fill_value="NA")
    snv_4d = site_type == "4D"
    snvs_4d = biallelic_snvs.loc[snv_4d]
    if snvs_4d.empty:
        empty_spec = pd.DataFrame(
            {"class": CANONICAL_CLASSES, "count": 0, "opportunity": 0, "rate": np.nan, "proportion": np.nan}
        )
        return SpectrumResult(species=species, spectrum=empty_spec, summary={"total_snvs_4d_rare": 0})

    alt_count, covered_count = _alt_counts(snvs_4d, sample_cols)
    rare = (alt_count >= 1) & (alt_count <= alt_count_max) & (covered_count >= require_min_covered)
    rare_snvs = snvs_4d.loc[rare, ["Ref", "Major", "Alt"]].copy()

    majors = rare_snvs["Major"].astype(str)
    alts = rare_snvs["Alt"].astype(str)
    valid = majors.isin(COMPLEMENT) & alts.isin(COMPLEMENT) & (majors != alts)
    rare_snvs = rare_snvs.loc[valid]
    majors = majors.loc[valid]
    alts = alts.loc[valid]

    folded = [fold_to_canonical(m, a) for m, a in zip(majors, alts)]
    class_series = pd.Series(folded, index=rare_snvs.index, name="class")
    counts = class_series.value_counts().reindex(CANONICAL_CLASSES, fill_value=0)

    # opportunity = count of major-polarized 4D reference bases, strand-folded
    site_4d_mask = mut_df["Site Type"] == "4D"
    if "Ref Base" in mut_df.columns:
        bases = mut_df.loc[site_4d_mask, "Ref Base"].astype(str)
    else:
        # fallback: use the helper's reference bases via biallelic_snvs is not safe; require Ref Base
        raise KeyError("mut_df must include 'Ref Base' for opportunity computation")
    bases = bases[bases.isin(COMPLEMENT)]
    base_counts = bases.value_counts().reindex(["A", "C", "G", "T"], fill_value=0)
    opportunity_C = int(base_counts["C"] + base_counts["G"])
    opportunity_T = int(base_counts["T"] + base_counts["A"])
    opportunity = {
        "C>A": opportunity_C,
        "C>G": opportunity_C,
        "C>T": opportunity_C,
        "T>A": opportunity_T,
        "T>C": opportunity_T,
        "T>G": opportunity_T,
    }

    spectrum = pd.DataFrame(
        {
            "class": CANONICAL_CLASSES,
            "count": [int(counts[c]) for c in CANONICAL_CLASSES],
            "opportunity": [opportunity[c] for c in CANONICAL_CLASSES],
        }
    )
    spectrum["rate"] = spectrum["count"] / spectrum["opportunity"].replace(0, np.nan)
    total = spectrum["count"].sum()
    spectrum["proportion"] = spectrum["count"] / total if total > 0 else np.nan

    ts = int(spectrum.loc[spectrum["class"].isin(TRANSITIONS), "count"].sum())
    tv = int(spectrum.loc[~spectrum["class"].isin(TRANSITIONS), "count"].sum())
    ts_tv = (ts / tv) if tv > 0 else np.nan

    summary = {
        "species": species,
        "alt_count_max": int(alt_count_max),
        "n_4D_sites": int(site_4d_mask.sum()),
        "n_4D_snvs_total": int(snvs_4d.shape[0]),
        "n_4D_snvs_rare": int(rare_snvs.shape[0]),
        "opportunity_C_source": opportunity_C,
        "opportunity_T_source": opportunity_T,
        "transitions": ts,
        "transversions": tv,
        "ts_tv": ts_tv,
        "n_samples": len(sample_cols),
    }
    return SpectrumResult(species=species, spectrum=spectrum, summary=summary)
