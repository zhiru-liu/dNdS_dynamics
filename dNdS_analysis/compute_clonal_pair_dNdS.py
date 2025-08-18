"""
Analogous to the compute_typical_pair_dNdS.py script, this script is used to compute dNdS for clonal (i.e. close pair with no detected recombination events).

Need to load CP-HMM results from Liu&Good 2024 data to find closely related pairs; once done running this script,
can find sample names of clonal pairs in the output csv file.
"""

import pandas as pd
import logging
from dNdS_analysis.utils import snv_utils, dynamics_utils
import dNdS_analysis.config as config

def compute_clonal_pair(species):
    snv_helper = snv_utils.SNVHelper(species, compute_bi_snvs=False, annotate=False)
    clonal_pairs = dynamics_utils.find_clonal_pairs(snv_helper)
    if len(clonal_pairs) == 0:
        logging.info(f'No clonal pairs found for species: {species}')
        return None
    snv_helper.annotate_snvs()

    columns = ['species_name','sample 1','sample 2',
               'core_diff','core_len',
               'core_len_4D','core_len_1D',
               'core_diff_4D','core_diff_1D',
               # below are for the breakdown of 1D mutations
               'clonal_n','clonal_m','clonal_nn',
               'clonal_mut_n','clonal_mut_m','clonal_mut_nn']
    results_df = pd.DataFrame(index=range(len(clonal_pairs)), columns=columns)
    results_df['species_name'] = species

    for i in range(len(clonal_pairs)):
        # sample1, sample2 = snv_helper.sample_random_fully_recombined_pair()
        sample1, sample2 = clonal_pairs[i]
        logging.info(f'Computing dNdS for pair: {sample1} and {sample2}')
        # loggin the identical fraction as a sanity check; should be >95 say for clonal pairs
        logging.info(f'Identical block fraction: {snv_helper.get_pair_identical_block(sample1, sample2)}')
        diff_sites = snv_helper.compute_pairwise_snvs(sample1, sample2)
        covered_mask = snv_helper.compute_pairwise_coverage(sample1, sample2)

        results_df.loc[i, 'sample 1'] = sample1
        results_df.loc[i, 'sample 2'] = sample2

        results_df.at[i, 'core_len'] = covered_mask.sum()
        results_df.at[i, 'core_diff'] = diff_sites.sum()
        results_df.at[i, 'core_len_4D'] = (covered_mask & snv_helper.core_4D).sum()
        results_df.at[i, 'core_len_1D'] = (covered_mask & snv_helper.core_1D).sum()
        results_df.at[i, 'core_diff_4D'] = (diff_sites & snv_helper.snv_4D).sum()
        results_df.at[i, 'core_diff_1D'] = (diff_sites & snv_helper.snv_1D).sum()

        # only a subset of all polymorphisms are different between the two samples
        mut_types_1D = snv_helper.snv_types[(diff_sites & snv_helper.snv_1D)]
        results_df.loc[i, ['clonal_mut_n', 'clonal_mut_m', 'clonal_mut_nn']] = [(mut_types_1D==x).sum() for x in ['n', 'm', 'nn']]
        # count the mutational opportunities for all covered core sites between the pair
        results_df.loc[i, ['clonal_n', 'clonal_m', 'clonal_nn']] = snv_helper.mut_df.loc[(covered_mask & snv_helper.core_1D), ['n', 'm', 'nn']].sum().values
        # snv_helper.compute_pair_dNdS(sample1, sample2, results_df, breakdown_df, i)

    return results_df

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    snv_folder_list = config.snv_data_path.glob('*')
    for folder in snv_folder_list:
        folder_name = folder.name
        if folder_name.startswith('.'):
            continue
        species = folder_name
        # species = 'Alistipes_finegoldii_56071'
        clonal_pair_folder = config.data_path / 'gut_microbiome_clonal_pair_dNdS' / f'{species}.csv'
        if clonal_pair_folder.exists():
            logging.info(f'{species} already exists at {clonal_pair_folder}')
            continue
        logging.info(f'Computing clonal pair for species: {species}')
        results_df = compute_clonal_pair(species)
        if results_df is not None:
            results_df.to_csv(clonal_pair_folder, index=False)
            logging.info(f'Saved clonal pair for species: {species} to {clonal_pair_folder}')
