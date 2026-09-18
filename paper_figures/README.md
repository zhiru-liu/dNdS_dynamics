# paper_figures/ — figure-generating scripts

Every script that produces a figure in the manuscript lives here. Each reads the
vendored input tables in `data/` (plus, for a few figures, the external public
data listed below) and writes to `figures/` (`config.fig_path`) using the same
file name as the published figure. The figures as they appear in the paper are
kept in [`figs/`](figs/). The tables below map each figure to its file, the
script that produces it, and the data it is drawn from.

## Input data

Per-pair tables vendored in `data/`:

| `data/` path | Contents | Produced by |
|---|---|---|
| `gut_microbiome_{close,clonal,typical}_pair_dNdS/<species>.csv` | Per-pair 1D/4D site and difference counts (clonal / recombined / core genome) for close, purely clonal, and unrelated QP strain pairs | `dnds_dynamics.dnds.qp` from the QP SNV catalogs and CP-HMM transfers |
| `qp_extended_recomb/per_pair_extended_for_grid_min2.csv` | Per-pair clonal counts after extending recombination tracts over nonsynonymous-rich flanks | `analysis/qp_extended_recomb_across_species.py --min-ext-1d 2`, merged onto `analysis/build_masked_clonal_counts.py` |
| `typical_dnds_across_species/species_summary.csv` | Per-species unrelated-pair dN/dS and dS summaries | `analysis/typical_dnds_vs_ds_across_species.py` |
| `isolate_dnds/{aputredinis,pvulgatus,pvulgatus_extended}/*.csv` | Per-pair isolate dN/dS tables (close, purely clonal, unrelated pairs; stratified points); `_extended` = extended recombination mask. `isolate_manifest.tsv` lists the analysed NCBI assemblies (accession, BioSample, BioProject, CheckM, quality tier) | Isolate pipeline in `analysis/` (below) |
| `pvulgatus_vs_qp/qp_sample_div_to_ref.csv` | Clade (vulgatus / dorei) of each QP *P. vulgatus* sample from its 4D divergence to the reference | `compare_pvulgatus_qp_clonal_dnds.py` |
| `identical_fraction/<species>.csv` | Between-host identical-fraction matrices (fraction of 1,000-site 4D blocks with no differences) for the species in Figs 5 and Q | Liu & Good 2024 |
| `Bv_clades.txt`, `QP_samples.csv` | *P. vulgatus* clade labels; QP sample list | Liu & Good 2024 |
| `forward_time_simulations/simulation_output_*.txt.gz` | Raw output of the forward-time simulations (Fig R): sampled individuals per replicate and time point with their fitness and the frequencies of their synonymous and nonsynonymous mutations, for four recombination rates and up to 11 population sizes (file name = `<s>_<Ns>_<NU>_<NR>_<l/L>`) | `paper_figures/forward_time_simulations/run_simulation.py` driving the C++ simulator in the same directory |

External public data needed by some figures:

- **QP SNV catalogs** (Figs 5, Q, F): [Zenodo record 14853785](https://zenodo.org/records/14853785)
  (Garud & Good 2019; Liu & Good 2024); location set by `config.snv_data_path`.
- **CP-HMM recombination events** (Fig A): `data/gut_microbiome_transfers.csv`
  (44 MB, not vendored) from the supplement of
  [Liu & Good 2024](https://doi.org/10.1371/journal.pbio.3002472).
- ***S. aureus* alignment** (Fig D): `Saureus.fasta` and `Saureus.non-core-sites.txt`
  from Didelot & Wilson 2015, [figshare DOI 10.6084/m9.figshare.19626912](https://doi.org/10.6084/m9.figshare.19626912)
  (`cfml.tgz`, CC BY 4.0); 110 genomes originally sequenced by Everitt et al. 2014,
  aligned to MRSA252. Location set by `config.STAPH_DATA_DIR`.
- **Isolate genomes** (Figs B, C, F): NCBI assemblies retrieved by
  `analysis/download_ncbi_isolates.py`.

## Upstream data prep (not figure code) — all in `analysis/`

- **Isolate pipeline**: `download_ncbi_isolates` → `build_isolate_snv_table`
  (`--midas-species`, one builder for *A. putredinis* + *P. vulgatus*) →
  `run_isolate_cphmm` (presets `aputredinis`/`pvulgatus`; `--workers` for
  parallel) → `build_extended_recombination_cache` → `compute_identical_fraction`,
  `compute_isolate_dnds`, `compute_stratified_isolate_dnds`. The resulting per-pair
  tables are vendored under `data/isolate_dnds/`.
- **QP cohort prep**: `qp_extended_recomb_across_species --min-ext-1d 2` and
  `build_masked_clonal_counts` (extended-tract per-pair table, vendored as
  `data/qp_extended_recomb/per_pair_extended_for_grid_min2.csv`);
  `typical_dnds_vs_ds_across_species` (vendored as
  `data/typical_dnds_across_species/species_summary.csv`). Both need the QP SNV
  catalogs.

## Main-text figures

| Fig | File | Script | Inputs (`data/`) | Notes |
|---|---|---|---|---|
| 1 | [theory_schematic.pdf](figs/theory_schematic.pdf) | — | — | Schematic; drawn by hand. |
| 2 | [fig2_final.png](figs/fig2_final.png) | `plot_dNdS_separation.py` | `gut_microbiome_*_pair_dNdS/` | Panels B–D are the script's `dNdS.pdf`; panel A and the composite were assembled by hand. |
| 3 | [clonal_dNdS_purifying_fit.pdf](figs/clonal_dNdS_purifying_fit.pdf) | `clonal_dNdS_dynamics.py` | `gut_microbiome_*_pair_dNdS/` | One run also writes Figs 4, E, O, P. |
| 4 | [clonal_dNdS_missense_nonsense_panels_with_fits.pdf](figs/clonal_dNdS_missense_nonsense_panels_with_fits.pdf) | `clonal_dNdS_dynamics.py` | `gut_microbiome_*_pair_dNdS/` | |
| 5 | [Bv_pnps_by_freq.pdf](figs/Bv_pnps_by_freq.pdf) | `clonal_snv_prevalence.py` | `gut_microbiome_clonal_pair_dNdS/`, `identical_fraction/`, `Bv_clades.txt` + QP SNV catalogs | One run also writes Fig Q. |

## Supplementary figures (S1 Text)

| Fig | File | Script | Inputs (`data/`) | Notes |
|---|---|---|---|---|
| A | [dsc_vs_dS_conditional_fit_median.pdf](figs/dsc_vs_dS_conditional_fit_median.pdf) | `plot_dsc_conditional_dnds.py` | `gut_microbiome_*_pair_dNdS/` + `gut_microbiome_transfers.csv` | Run with `DSC_ESTIMATOR=median`. Fit helpers in `dsc_conditional_theory.py`. |
| B | [isolate_cphmm_qpcore_stratified_dnds_grid.pdf](figs/isolate_cphmm_qpcore_stratified_dnds_grid.pdf) | `plot_isolate_cphmm_stratified_dnds_grid.py` | `isolate_dnds/{aputredinis,pvulgatus}/stratified_dnds_points.csv` | |
| C | [pvulgatus_vs_qp_dnds_comparison.pdf](figs/pvulgatus_vs_qp_dnds_comparison.pdf) | `compare_pvulgatus_qp_clonal_dnds.py` | `gut_microbiome_*_pair_dNdS/`, `isolate_dnds/pvulgatus{,_extended}/`, `pvulgatus_vs_qp/` | The script is a slightly updated version of the one used for the published figure: the bottom row now uses thinning-averaged bins, shows the main-text fits from Figs 3–4 instead of per-dataset refits, and plots the isolates under the extended recombination mask. The top row and the underlying data are unchanged. |
| D | [Staph_dNdS.pdf](figs/Staph_dNdS.pdf) | `staph_dNdS_separation.py` | *S. aureus* alignment (external, above) | |
| E | [clonal_dNdS_species_grid_all.pdf](figs/clonal_dNdS_species_grid_all.pdf) | `clonal_dNdS_dynamics.py` | `gut_microbiome_*_pair_dNdS/` | |
| F | [response_missed_recomb_extended.pdf](figs/response_missed_recomb_extended.pdf) | `tract_extension_demo.py` | QP SNV catalogs + isolate SNV tables (external) | Imports `response_missed_recomb_examples.py` (loads the example pairs). |
| G | [clonal_dNdS_species_grid_masked_extended_min2.pdf](figs/clonal_dNdS_species_grid_masked_extended_min2.pdf) | `plot_masked_purifying_fit.py` | `qp_extended_recomb/`, `gut_microbiome_typical_pair_dNdS/` | One run writes Figs G and L. |
| H | [typical_dnds_smu_composite.pdf](figs/typical_dnds_smu_composite.pdf) | `combine_typical_dnds_smu.py` | `typical_dnds_across_species/`, `gut_microbiome_*_pair_dNdS/` | Imports `smu_robustness_typical_grid.py` (panels b–c). |
| I | [purify_spread_null_overlay.pdf](figs/purify_spread_null_overlay.pdf) | `purify_spread_null_sim.py` | `gut_microbiome_*_pair_dNdS/` | One run writes Figs I and J. |
| J | [purify_spread_within_species.pdf](figs/purify_spread_within_species.pdf) | `purify_spread_null_sim.py` | `gut_microbiome_*_pair_dNdS/` | |
| K | [clonal_dNdS_purifying_fit_no_Ap.pdf](figs/clonal_dNdS_purifying_fit_no_Ap.pdf) | `clonal_dNdS_dynamics.py` | `gut_microbiome_*_pair_dNdS/` | Run with the *A. putredinis* filter enabled (the commented line near the top of the script); this is its `clonal_dNdS_purifying_fit` output. |
| L | [clonal_dNdS_purifying_fit_withAp_masked_extended_min2.pdf](figs/clonal_dNdS_purifying_fit_withAp_masked_extended_min2.pdf) | `plot_masked_purifying_fit.py` | `qp_extended_recomb/`, `gut_microbiome_typical_pair_dNdS/` | Same run as Fig G. |
| M | [purify_three_class_scan.pdf](figs/purify_three_class_scan.pdf) | `clonal_dNdS_dynamics.py` | `gut_microbiome_*_pair_dNdS/` | From the same *A. putredinis*-excluded run as Fig K. |
| N | [dfe_cumulative_bounds.pdf](figs/dfe_cumulative_bounds.pdf) | `plot_dfe_cumulative_bounds.py` | `gut_microbiome_*_pair_dNdS/` | |
| O | [clonal_dNdS_species_grid_missense.pdf](figs/clonal_dNdS_species_grid_missense.pdf) | `clonal_dNdS_dynamics.py` | `gut_microbiome_*_pair_dNdS/` | |
| P | [clonal_dNdS_species_grid_nonsense.pdf](figs/clonal_dNdS_species_grid_nonsense.pdf) | `clonal_dNdS_dynamics.py` | `gut_microbiome_*_pair_dNdS/` | |
| Q | [clonal_snv_prevalence_grid.pdf](figs/clonal_snv_prevalence_grid.pdf) | `clonal_snv_prevalence.py` | as Fig 5 (all species) | |
| R | [forward_time_simulations.pdf](figs/forward_time_simulations.pdf) | `forward_time_simulations/plot_forward_time_simulation_results.py` | `forward_time_simulations/` | Plots every complete run per recombination rate from the vendored outputs. To regenerate the outputs themselves: in `forward_time_simulations/`, `g++ -O3 -std=c++11 -o simulation main.cpp`, then `python run_simulation.py` (writes `simulation_output_*.txt.gz` there; point the plot script at them with `DNDS_SIM_DIR`). |
