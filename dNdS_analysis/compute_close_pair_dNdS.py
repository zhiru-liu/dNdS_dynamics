"""
This main script performs a dNdS (nonsynonymous to synonymous substitution rate) analysis on genetic sequences of microbial species from gut microbiome transfer data.
It processes only closely related pairs (i.e. ones with enough clonal frame such that recombination events can be detected).
It processes multiple species data to compute various metrics related to dNdS in recombined and clonal regions.

Steps involved:
    - Detected recombination events came from Liu&Good PLOS Bio 2024 paper, which is provided in the `gut_microbiome_transfers.csv` file.
    - Loads SNV (Single Nucleotide Variants) data using utilities from the `dNdS_analysis` package.
    - Retrieves transfer data for the species and computes pair-wise metrics across samples.
    - Constructs recombination and mutation masks to distinguish between clonal and recombination-derived variations.
    - Aggregates and saves these metrics to a CSV file named according to the species.

Output:
    - CSV files for each species, containing various combinations of dNdS metrics (clonal vs recombined, 1D vs 4D, etc.)

Note:
    - Takes about 3sec per pairwise comparison; the slowest step is the computation of recombination mask.
    - In gut bacteria dataset, total number of pairs is around 7000, so it will take a few hours to process all pairs.
"""

import pandas as pd
import numpy as np
import sys
from Bio import SeqIO
import gzip
import logging

sys.path.append('/Users/Device6/Documents/Research/bgoodlab/UHGG/')
import uhgg_helper.annotation_utils as annotation_utils

from dNdS_analysis.utils import snv_utils, dynamics_utils
import dNdS_analysis.config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
all_transfers = dynamics_utils.load_detected_transfers()

snv_folder_list = dNdS_analysis.config.snv_data_path.glob('*')
for folder in snv_folder_list:
    folder_name = folder.name
    if folder_name.startswith('.'):
        continue
    species = folder_name
    logging.info('Processing species: {}'.format(species))
    results_path = dNdS_analysis.config.data_path / 'gut_microbiome_close_pair_dNdS' / '{}.csv'.format(species)
    if results_path.exists():
        logging.info('Already processed species: {}'.format(species))
        continue
    
    # load SNV data
    snv_helper = snv_utils.SNVHelper(species, compute_bi_snvs=False, annotate=True)

    # get all transfers for this species
    species_transfers = all_transfers[all_transfers['Species name']==species]
    num_pairs = len(species_transfers.groupby(['Sample 1', 'Sample 2']))

    # prepare dataframes for dNdS results
    columns = ['species_name','sample 1','sample 2',
               'core_diff','core_len',
               'core_len_4D','core_len_1D','core_diff_4D','core_diff_1D',
               'recomb_len_4D','recomb_len_1D','recomb_diff_4D','recomb_diff_1D',
               'clonal_len_4D','clonal_len_1D','clonal_diff_4D','clonal_diff_1D',
               'clonal_n','clonal_m','clonal_nn',
               'recomb_n','recomb_m','recomb_nn',
               'clonal_mut_n','clonal_mut_m','clonal_mut_nn',
               'recomb_mut_n','recomb_mut_m','recomb_mut_nn']
    # clonal_n + recomb_n = core_n in typical pair dNdS

    results_df = pd.DataFrame(index=range(num_pairs), columns=columns)
    results_df['species_name'] = species

    # main work: loop through all pairs
    i = 0
    # TODO: 05/11/2024 - add pairs without recombination events here; can skip all recombination fields in that case
    # TODO: alternatively, have a separate script to compute dNdS for all clonal pairs (no recombination events)
    for pair, grouped in species_transfers.groupby(['Sample 1', 'Sample 2']):
        # build a Interval index for this pair
        pair_transfer_df = grouped
        sample1, sample2 = pair
        # just to make sure since some species might save sample names as integers
        sample1 = str(sample1)
        sample2 = str(sample2)
        logging.info(f'Computing dNdS for pair: {sample1} and {sample2}')

        recombination_mask = np.zeros(snv_helper.coverage.shape[0]).astype(bool)
        for idx, row in pair_transfer_df.iterrows():
            start, end = row['Core genome start loc'], row['Core genome end loc']
            recombination_mask[start:end + 1] = True
        recombination_snv_mask = recombination_mask[snv_helper.core_to_snvs]

        # shape: core genome length
        diff_sites = snv_helper.compute_pairwise_snvs(sample1, sample2)
        covered_mask = snv_helper.compute_pairwise_coverage(sample1, sample2)

        results_df.loc[i, 'sample 1'] = sample1
        results_df.loc[i, 'sample 2'] = sample2
        results_df.loc[i, 'core_len'] = covered_mask.sum()
        results_df.loc[i, 'core_diff'] = diff_sites.sum()
        results_df.loc[i, 'core_len_4D'] = (covered_mask & snv_helper.core_4D).sum()
        results_df.loc[i, 'core_len_1D'] = (covered_mask & snv_helper.core_1D).sum()
        results_df.loc[i, 'core_diff_4D'] = (diff_sites & snv_helper.snv_4D).sum()
        results_df.loc[i, 'core_diff_1D'] = (diff_sites & snv_helper.snv_1D).sum()
        results_df.loc[i, 'recomb_len_4D'] = (covered_mask & recombination_mask & snv_helper.core_4D).sum()
        results_df.loc[i, 'recomb_len_1D'] = (covered_mask & recombination_mask & snv_helper.core_1D).sum()
        results_df.loc[i, 'recomb_diff_4D'] = (diff_sites & recombination_snv_mask & snv_helper.snv_4D).sum()
        results_df.loc[i, 'recomb_diff_1D'] = (diff_sites & recombination_snv_mask & snv_helper.snv_1D).sum()
        results_df.loc[i, 'clonal_len_4D'] = (covered_mask & ~recombination_mask & snv_helper.core_4D).sum()
        results_df.loc[i, 'clonal_len_1D'] = (covered_mask & ~recombination_mask & snv_helper.core_1D).sum()
        results_df.loc[i, 'clonal_diff_4D'] = (diff_sites & ~recombination_snv_mask & snv_helper.snv_4D).sum()
        results_df.loc[i, 'clonal_diff_1D'] = (diff_sites & ~recombination_snv_mask & snv_helper.snv_1D).sum()

        clonal_mut_types_1D = snv_helper.snv_types[(diff_sites & ~recombination_snv_mask & snv_helper.snv_1D)]
        results_df.loc[i, ['clonal_mut_n', 'clonal_mut_m', 'clonal_mut_nn']] = [(clonal_mut_types_1D==x).sum() for x in ['n', 'm', 'nn']]
        results_df.loc[i, ['clonal_n', 'clonal_m', 'clonal_nn']] = snv_helper.mut_df.loc[(covered_mask & ~recombination_mask & snv_helper.core_1D), ['n', 'm', 'nn']].sum().values
        recomb_mut_types_1D = snv_helper.snv_types[(diff_sites & recombination_snv_mask & snv_helper.snv_1D)]
        results_df.loc[i, ['recomb_mut_n', 'recomb_mut_m', 'recomb_mut_nn']] = [(recomb_mut_types_1D==x).sum() for x in ['n', 'm', 'nn']]
        results_df.loc[i, ['recomb_n', 'recomb_m', 'recomb_nn']] = snv_helper.mut_df.loc[(covered_mask & recombination_mask & snv_helper.core_1D), ['n', 'm', 'nn']].sum().values

        i += 1

    results_df.to_csv(results_path, index=False)
    logging.info('Done computing dNdS for species: {}'.format(species))