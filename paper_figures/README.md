# paper_figures/ — figure-generating scripts

Every script that produces a manuscript, SI, or response figure lives here. Each
reads vendored inputs (`data/`), cached analysis outputs (`outputs/`), or the raw
QP/isolate catalogs, and writes to `figures/`. All **analysis / data-prep** that
feeds these scripts lives in `analysis/` — see "Upstream data prep" at the bottom.

To locate the code behind a figure, find its output stem in the table below.

## Main-text figures

| Figure | Output stem | Script |
|---|---|---|
| dN/dS separation (clonal/recombined/full) | `dNdS_separation*` | `plot_dNdS_separation.py` |
| Clonal dN/dS dynamics + missense/nonsense | `clonal_dNdS_*panels_with_fits`, `clonal_dNdS_purifying_fit` | `clonal_dNdS_dynamics.py` |
| Clonal SNV prevalence (Bv detail) | `Bv_pnps_by_freq` | `clonal_snv_prevalence.py` |
| f_r vs dS_c composite | `fr_dSc_*` | `fr_dSc_with_typical.py` |

## SI / response figures

| Figure | Output stem | Script |
|---|---|---|
| Isolate stratified dN/dS grid (Ap + Pv) | `isolate_cphmm_qpcore_stratified_dnds_grid` | `plot_isolate_cphmm_stratified_dnds_grid.py` |
| Per-species clonal grid (all / missense / nonsense) | `clonal_dNdS_species_grid_{all,missense,nonsense}` | `clonal_dNdS_dynamics.py` |
| Three-class purifying scan | `purify_three_class_scan` | `clonal_dNdS_dynamics.py` |
| Per-species clonal grid, clonal SNV prevalence | `clonal_snv_prevalence_grid` | `clonal_snv_prevalence.py` |
| Missed-recombination worked examples (extended) | `response_missed_recomb_extended` | `tract_extension_demo.py` ¹ |
| Masked-extended fit + species grid (with Ap) | `clonal_dNdS_purifying_fit_withAp_masked_extended_min2`, `clonal_dNdS_species_grid_masked_extended_min2` | `plot_masked_purifying_fit.py` |
| P. vulgatus isolate vs QP comparison | `pvulgatus_vs_qp_dnds_comparison` | `compare_pvulgatus_qp_clonal_dnds.py` |
| Clonal dN/dS spread vs Poisson null | `purify_spread_null_overlay`, `purify_spread_within_species` | `purify_spread_null_sim.py` |
| Typical dN/dS–dS + s/μ composite | `typical_dnds_smu_composite` | `combine_typical_dnds_smu.py` ² |
| A. putredinis 1D hotspots (genome view) | `aputredinis_1D_hotspot_genomeview` | `plot_1d_hotspot_genomeview.py` |
| S. aureus dN/dS separation | `Staph_dNdS` | `staph_dNdS_separation.py` |

¹ imports `response_missed_recomb_examples.py` (example-loading helper, not a
  standalone figure).
² imports `smu_robustness_typical_grid.py` (panels b–c, also standalone
  `smu_robustness_typical_grid`) and reads `outputs/typical_dnds_across_species/
  species_summary.csv` (from `analysis/typical_dnds_vs_ds_across_species.py`).

## Upstream data prep (not figure code) — all in `analysis/`

- **Isolate pipeline**: `download_ncbi_isolates` → `build_isolate_snv_table`
  (`--midas-species`, one builder for Ap + Pv) → `run_isolate_cphmm` (presets
  `aputredinis`/`pvulgatus`; `--workers` for parallel) → `build_extended_recombination_cache`
  → `compute_identical_fraction`, `compute_isolate_dnds`, `compute_stratified_isolate_dnds`.
- **QP cohort prep**: `qp_extended_recomb_across_species` (extended tract pair-table
  for the masked-extended figures), `build_masked_clonal_counts` (full pair-set
  scaffold), `typical_dnds_vs_ds_across_species` (typical-pair summary).
