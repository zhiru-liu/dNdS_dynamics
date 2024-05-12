"""
Analogous to the compute_close_pair_dNdS.py script, this script is used to compute dNdS for typical (i.e. fully recombined pairs with minimal clonal regions).
The script randomly selects fully recombined pairs and computes various metrics
such as core differences, core lengths, and breakdown of 1D mutations. The results are saved in a CSV file for each species.
"""

import pandas as pd
import logging
from dNdS_analysis.utils import snv_utils
import dNdS_analysis.config as config

def compute_typical_pair(species, num_pairs=20):
    snv_helper = snv_utils.SNVHelper(species, compute_bi_snvs=False, annotate=True)
    if not snv_helper.check_if_any_pair_fully_recombined():
        logging.info(f'Species: {species} has no fully recombined pair')
        return

    columns = ['species_name','sample 1','sample 2',
               'core_diff','core_len',
               'core_len_4D','core_len_1D',
               'core_diff_4D','core_diff_1D',
               # below are for the breakdown of 1D mutations
               'core_n','core_m','core_nn',
               'core_mut_n','core_mut_m','core_mut_nn']
    results_df = pd.DataFrame(index=range(num_pairs), columns=columns)
    results_df['species_name'] = species

    for i in range(num_pairs):
        sample1, sample2 = snv_helper.sample_random_fully_recombined_pair()
        logging.info(f'Computing dNdS for pair: {sample1} and {sample2}')
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
        results_df.loc[i, ['core_mut_n', 'core_mut_m', 'core_mut_nn']] = [(mut_types_1D==x).sum() for x in ['n', 'm', 'nn']]
        # count the mutational opportunities for all covered core sites between the pair
        results_df.loc[i, ['core_n', 'core_m', 'core_nn']] = snv_helper.mut_df.loc[(covered_mask & snv_helper.core_1D), ['n', 'm', 'nn']].sum().values
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
        typical_pair_folder = config.data_path / 'gut_microbiome_typical_pair_dNdS' / f'{species}.csv'
        if typical_pair_folder.exists():
            logging.info(f'Typical pair for species: {species} already exists at {typical_pair_folder}')
            continue
        logging.info(f'Computing typical pair for species: {species}')
        results_df = compute_typical_pair(species)
        if results_df is not None:
            results_df.to_csv(typical_pair_folder, index=False)
            logging.info(f'Saved typical pair for species: {species} to {typical_pair_folder}')
