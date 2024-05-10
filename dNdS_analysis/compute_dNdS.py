import pandas as pd
import numpy as np
import sys
from Bio import SeqIO
import gzip
import logging

sys.path.append('/Users/Device6/Documents/Research/bgoodlab/UHGG/uhgg_helper')
import annotation_utils

from dNdS_analysis.utils import snv_utils
import dNdS_analysis.config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
all_transfers = pd.read_csv(dNdS_analysis.config.data_path / 'gut_microbiome_transfers.csv', index_col=0)

snv_folder_list = dNdS_analysis.config.snv_data_path.glob('*')
for folder in snv_folder_list:
    folder_name = folder.name
    if folder_name.startswith('.'):
        continue
    species = folder_name
    logging.info('Processing species: {}'.format(species))
    results_path = dNdS_analysis.config.data_path / 'gut_microbiome_dNdS_20240509' / '{}.csv'.format(species)
    if results_path.exists():
        logging.info('Already processed species: {}'.format(species))
        continue

    snvs_df = pd.read_feather(dNdS_analysis.config.snv_data_path / species / 'snv_catalog.feather')
    coverage_df = pd.read_feather(dNdS_analysis.config.snv_data_path/ species / 'coverage.feather')
    alleles_df = pd.read_feather(dNdS_analysis.config.snv_data_path / species / 'alleles.feather')

    # decode all bytes columns
    for col in snvs_df.columns:
        if snvs_df[col].dtype == 'O':
            snvs_df[col] = snvs_df[col].str.decode('utf-8')

    for col in coverage_df.columns:
        if coverage_df[col].dtype == 'O':
            coverage_df[col] = coverage_df[col].str.decode('utf-8')

    for col in alleles_df.columns:
        if alleles_df[col].dtype == 'O':
            alleles_df[col] = alleles_df[col].str.decode('utf-8')

    snvs_df.set_index(['Contig', 'Location'], inplace=True)
    coverage_df.set_index(['Contig', 'Location'], inplace=True)
    alleles_df.set_index(['Contig', 'Location'], inplace=True)
    logging.info('Data loaded')

    bi_snvs, multi_sites = snv_utils.compute_biallelic_snvs(snvs=snvs_df, alleles=alleles_df, coverage=coverage_df)
    logging.info('Done computing bi-allelic SNVs')

    logging.info('Annotating SNVs')
    ref_genome_path = dNdS_analysis.config.ref_genome_path / species / 'genome.fna.gz'
    gene_feature_path = dNdS_analysis.config.ref_genome_path / species / 'genome.features.gz'

    with gzip.open(ref_genome_path, 'rt') as file:
        records = list(SeqIO.parse(file, "fasta"))
        
    with gzip.open(gene_feature_path, 'rt') as file:
        gene_df = pd.read_csv(file, sep='\t')
    gene_df.columns = ['Gene ID', 'Contig', 'Start', 'End', 'Strand', 'Type', 'Info']
    polarized_records = snv_utils.polarize_reference_seq(records, snv_df=bi_snvs)

    res_dfs = []
    for record in polarized_records:
        # annotate one contig
        sample_seq = record.seq
        contig = record.id
        contig_gene_df = gene_df[gene_df['Contig']==contig]
        res = annotation_utils.annotate_sequence_site_types_to_df(sample_seq, contig_gene_df)
        res['Contig'] = contig
        res_dfs.append(res)

    mut_df = pd.concat(res_dfs)
    mut_df.set_index(['Contig', 'Location'], inplace=True)
    mut_df = mut_df.loc[coverage_df.index]
    # rename A mut to A, etc
    mut_df = mut_df.rename(columns={'{} Mut'.format(x): x for x in ['A', 'C', 'G', 'T']})
    logging.info('Done annotating SNVs')

    # get all transfers for this species
    species_transfers = all_transfers[all_transfers['Species name']==species]
    num_pairs = len(species_transfers.groupby(['Sample 1', 'Sample 2']))

    # prepare dataframes for dNdS results
    columns = ['species_name','sample 1','sample 2','core_diff','core_len','core_len_4D','core_len_1D','core_diff_4D','core_diff_1D','recomb_len_4D','recomb_len_1D','recomb_diff_4D','recomb_diff_1D','clonal_len_4D','clonal_len_1D','clonal_diff_4D','clonal_diff_1D']
    results_df = pd.DataFrame(index=range(num_pairs), columns=columns)
    results_df['species_name'] = species

    columns = ['species_name','sample 1','sample 2','clonal_n','clonal_m','clonal_nn','recomb_n','recomb_m','recomb_nn']
    mut_opportunity_df = pd.DataFrame(index=range(num_pairs), columns=columns)
    mut_opportunity_df['species_name'] = species

    # a few masks that will be used by all pairs
    covered_mut_info = mut_df.loc[coverage_df.index]
    covered_1D = covered_mut_info['Site Type']=='1D'
    covered_4D = covered_mut_info['Site Type']=='4D'
    snv_mut_info = mut_df.loc[bi_snvs.index]
    snv_1D_mask = snv_mut_info['Site Type']=='1D'
    snv_4D_mask = snv_mut_info['Site Type']=='4D'
    snv_1D_type = np.array([snv_mut_info.at[idx, col] for idx, col in zip(snv_mut_info.index, bi_snvs.loc[:, 'Alt'])])

    covered_mut_info['s'] = (covered_mut_info.loc[:, ['A', 'T', 'C', 'G']]=='s').sum(axis=1)-1
    covered_mut_info['n'] = (covered_mut_info.loc[:, ['A', 'T', 'C', 'G']]=='n').sum(axis=1)
    covered_mut_info['m'] = (covered_mut_info.loc[:, ['A', 'T', 'C', 'G']]=='m').sum(axis=1)
    covered_mut_info['nn'] = (covered_mut_info.loc[:, ['A', 'T', 'C', 'G']]=='nn').sum(axis=1)

    # main work: loop through all pairs
    i = 0
    logging.info('Computing dNdS for species: {}'.format(species))
    for pair, grouped in species_transfers.groupby(['Sample 1', 'Sample 2']):
        # build a Interval index for this pair
        pair_transfer_df = grouped
        sample1, sample2 = pair
        sample1 = str(sample1)
        sample2 = str(sample2)

        # build recombination mask
        location_intervals = pd.IntervalIndex.from_arrays(pair_transfer_df['Reference genome start loc'], pair_transfer_df['Reference genome end loc'], closed='both')
        # Create MultiIndex using 'Contig' and the interval index for 'Location'
        recombination_index = pd.MultiIndex.from_arrays([pair_transfer_df['Reference contig'], location_intervals], names=['Contig', 'Location'])

        # shape: core genome length
        recombination_mask = coverage_df.index.isin(recombination_index)
        # shape: snv length
        recombination_snv_mask = bi_snvs.index.isin(recombination_index)

        # shape: core genome length
        covered_mask = coverage_df.loc[:, [sample1, sample2]].all(axis=1)

        pair_snvs = bi_snvs.loc[:, [sample1, sample2]]
        # shape: snvs length
        diff_sites = (pair_snvs.iloc[:, 0] != pair_snvs.iloc[:, 1]) & (pair_snvs.iloc[:, 0] != 255) & (pair_snvs.iloc[:, 1] != 255)
        # snv_sites = diff_sites & (diff_sites.index.isin(covered_sites))
        # snv_sites = snv_sites[snv_sites].index

        # snv_mut_info
        # pair_muts = covere.loc[snv_sites]
        # mut_sites_1D = pair_muts[pair_muts['Site Type']=='1D'].index
        # mut_sites_4D = pair_muts[pair_muts['Site Type']=='4D'].index

        # mut_in_recomb_1D = mut_sites_1D.isin(recombination_index)
        # mut_in_recomb_4D = mut_sites_4D.isin(recombination_index)

        # covered_muts = covered_mut_info[covered_mask]
        # core_sites_1D = covered_muts[covered_muts['Site Type']=='1D'].index
        # core_sites_4D = covered_muts[covered_muts['Site Type']=='4D'].index

        # mut_1D_types = [mut_df.at[idx, col] for idx, col in zip(mut_sites_1D, bi_snvs.loc[mut_sites_1D, 'Alt'])]

        results_df.loc[i, 'sample 1'] = sample1
        results_df.loc[i, 'sample 2'] = sample2
        results_df.loc[i, 'core_len'] = covered_mask.sum()
        results_df.loc[i, 'core_diff'] = diff_sites.sum()
        results_df.loc[i, 'core_len_4D'] = (covered_mask & covered_4D).sum()
        results_df.loc[i, 'core_len_1D'] = (covered_mask & covered_1D).sum()
        results_df.loc[i, 'core_diff_4D'] = (diff_sites & snv_4D_mask).sum()
        results_df.loc[i, 'core_diff_1D'] = (diff_sites & snv_1D_mask).sum()
        results_df.loc[i, 'recomb_len_4D'] = (covered_mask & recombination_mask & covered_4D).sum()
        results_df.loc[i, 'recomb_len_1D'] = (covered_mask & recombination_mask & covered_1D).sum()
        results_df.loc[i, 'recomb_diff_4D'] = (diff_sites & recombination_snv_mask & snv_4D_mask).sum()
        results_df.loc[i, 'recomb_diff_1D'] = (diff_sites & recombination_snv_mask & snv_1D_mask).sum()
        results_df.loc[i, 'clonal_len_4D'] = (covered_mask & ~recombination_mask & covered_4D).sum()
        results_df.loc[i, 'clonal_len_1D'] = (covered_mask & ~recombination_mask & covered_1D).sum()
        results_df.loc[i, 'clonal_diff_4D'] = (diff_sites & ~recombination_snv_mask & snv_4D_mask).sum()
        results_df.loc[i, 'clonal_diff_1D'] = (diff_sites & ~recombination_snv_mask & snv_1D_mask).sum()

        mut_opportunity_df.loc[i, 'sample 1'] = sample1
        mut_opportunity_df.loc[i, 'sample 2'] = sample2

        mut_types = snv_1D_type[(diff_sites & ~recombination_snv_mask & snv_1D_mask)]
        mut_opportunity_df.loc[i, ['clonal_mut_n', 'clonal_mut_m', 'clonal_mut_nn']] = [np.sum(mut_types==x) for x in ['n', 'm', 'nn']]
        mut_types = snv_1D_type[(diff_sites & recombination_snv_mask & snv_1D_mask)]
        mut_opportunity_df.loc[i, ['recomb_mut_n', 'recomb_mut_m', 'recomb_mut_nn']] = [np.sum(mut_types==x) for x in ['n', 'm', 'nn']]
        mut_opportunity_df.loc[i, ['clonal_n', 'clonal_m', 'clonal_nn']] = covered_mut_info.loc[(covered_mask & ~recombination_mask & covered_1D), ['n', 'm', 'nn']].sum().values
        mut_opportunity_df.loc[i, ['recomb_n', 'recomb_m', 'recomb_nn']] = covered_mut_info.loc[(covered_mask & recombination_mask & covered_1D), ['n', 'm', 'nn']].sum().values

        i += 1

    logging.info('Done computing dNdS for species: {}'.format(species))
    results_df.to_csv(results_path)
    mut_opportunity_df.to_csv(dNdS_analysis.config.data_path / 'gut_microbiome_dNdS_20240509' / '{}_mut_opportunity.csv'.format(species))
