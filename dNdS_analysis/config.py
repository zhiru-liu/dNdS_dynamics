import sys
import os
from pathlib import Path

# might not need this project anymore
microbiome_evolution_path = "/Users/Device6/Documents/Research/bgoodlab/microbiome_evolution"
snv_data_path = Path("/Volumes/Botein/GarudGood2019_snvs")
ref_genome_path = Path("/Volumes/Botein/LiuGood2024_files/microbiome_data/midas_db/rep_genomes")

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

# Might need to use the fraction of identical fraction in Liu&Good 2024
identical_fraction_path = Path('/Volumes/Botein/LiuGood2024_files/zhiru_analysis/pairwise_clonal_fraction/between_hosts')
# if less than 5% identical, then this pair probably has no clonal region left
fully_recombined_threshold = 0.05

# old os.path codes
# current_folder_path = os.path.dirname(os.path.abspath(__file__))
# root_path = os.path.dirname(current_folder_path)

# fig_dir = os.path.join(root_path, "figs")
# table_path = os.path.join(root_path, "tables")

# data_path = os.path.join(root_path, "data")
# fig_dat_dir = os.path.join(data_path, "figure_data")
