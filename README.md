# dN/dS Dynamics in Recombining Bacterial Populations

This repository contains scripts for the manuscript [Dynamics of dN/dS within recombining bacterial populations, Liu & Good (2024).](todo)
It builds on data and methods from Garud & Good (2019) and Liu & Good (2024).

## Data

The following data are used in this project:

- **SNV catalogs of human gut bacteria**: The genotype data of human gut bacteria analyzed in this study are available as compact SNV catalogs in [this Zenodo repository](https://zenodo.org/records/14853785). These genotype data were originally generated in the analysis of [Garud & Good et al (PLOS Bio 2019)](https://doi.org/10.1371/journal.pbio.3000102).

- **Recombination events inferred by CP-HMM**: Inferred recombination events can be found in the supplementary materials of [Liu & Good (PLOS Bio 2024)](https://doi.org/10.1371/journal.pbio.3002472).

## Analysis
The main analysis scripts are in the `dNdS_analysis` directory. The key scripts include:

- Scripts for estimating dN/dS between strains. All results are saved in the `data/` folder (e.g. `data/gut_microbiome_clonal_pair_dNdS/`). These include the number of synonymous and nonsynonymous sites, the number of observed synonymous and nonsynonymous differences, etc., for each pair of strains of a given species.
    - `dNdS_analysis/compute_close_pair_dNdS.py`: Compute dN/dS between closely related pairs of strains.
    - `dNdS_analysis/compute_clonal_pair_dNdS.py`: Compute dN/dS between clonal pairs of strains (i.e. no recombination).
    - `dNdS_analysis/compute_typical_pair_dNdS.py`: Compute dN/dS between unrelated pairs of strains.
- Scripts for plotting the main figures are in the `dNdS_analysis/plotting_scripts` directory. For example:
    - `dNdS_analysis/plotting_scripts/plot_dNdS_separation.py`: Figure 2 B-D
    - `dNdS_analysis/plotting_scripts/clonal_dNdS_dynamics.ipynb`: Figure 3, 4
    - `dNdS_analysis/plotting_scripts/clonal_snv_prevalence.ipynb`: Figure 5
- Utilities for annotating mutation types (e.g., missense vs. nonsense) are imported from [zhiru-liu/uhgg-helper](https://github.com/zhiru-liu/uhgg-helper), though any preferred method can be used.

## License
This project is licensed under the [MIT License](LICENSE).