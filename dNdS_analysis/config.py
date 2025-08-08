import sys
import os
from pathlib import Path

# specify the paths to some necessary data

# Compact SNV catalogs of all the QP genomes generated in Garud & Good et al 2019
snv_data_path = Path("/Volumes/Botein/GarudGood2019_snvs/snvs_feather")
# the reference genome fasta files in the MIDAS database; for annotating SNV type (missense, synonymous, etc.)
ref_genome_path = Path("/Volumes/Botein/LiuGood2024_files/microbiome_data/midas_db/rep_genomes")

# Need two additional things from Liu & Good 2024 data
# 1. the CP-HMM intermediate file to find close pairs with no detected recombination
# 2. the fraction of identical blocks to identify fully recombined pairs
# Once dNdS is computed, these pairs can be found in the output csv files
LiuGood2024_path = Path("/Volumes/Botein/LiuGood2024_files/zhiru_analysis")
identical_fraction_path = LiuGood2024_path / 'pairwise_clonal_fraction' / 'between_hosts'
# if less than 5% identical, then this pair probably has no clonal region left
fully_recombined_threshold = 0.05

# Get the current folder path (directory of the script file)
current_folder_path = Path(__file__).resolve().parent

# Get the root path (parent directory of the current folder)
root_path = current_folder_path.parent

# Define paths for figures and tables directory
fig_path = root_path / "figs"
table_path = root_path / "tables"

# Define path for the data directory
data_path = root_path / "data"
fig_dat_path = data_path / "figure_data"

# This species is abnormal in that it has no fully recombined pairs; so exclude it from the analysis
blacklist_species = ['Lachnospiraceae_bacterium_51870']