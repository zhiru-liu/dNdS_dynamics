import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

import dNdS_analysis.config as config

def computed_poisson_thinning(diffs, opportunities):
    # apply this to calculation of all dN/dS
    # specifically when calculating dS
    thinned_diffs_1 = np.random.binomial(diffs, 0.5)
    thinned_diffs_2 = diffs - thinned_diffs_1
    d1 = thinned_diffs_1 / (opportunities.astype(float) / 2)
    d2 = thinned_diffs_2 / (opportunities.astype(float) / 2)
    return d1, d2


def load_detected_transfers():
    all_transfers = pd.read_csv(config.data_path / 'gut_microbiome_transfers.csv', index_col=0, 
                            dtype={"Sample 1": "string", 
                                   "Sample 2": "string",
                                   "between clade?": "string"})
    return all_transfers


def load_dNdS_data():
    dnds_basepath = os.path.join(config.data_path, 'gut_microbiome_close_pair_dNdS')
    dnds_dfs = []
    for file in os.listdir(dnds_basepath):
        if file.startswith('.'):
            continue
        if file.endswith('mut_opportunity.csv'):
            continue
        species_name = file.split('.')[0]
        # transfer_df_path = os.path.join(config.analysis_directory, "closely_related", 'third_pass',
        #                                 species_name + '_all_transfers.pickle')
        # transfer_df = pd.read_pickle(transfer_df_path)
        dnds_df = pd.read_csv(os.path.join(dnds_basepath, species_name + '.csv'))
        # dnds_df['Clonal fraction'] = pair_to_cf
        # dnds_df['Clonal divergence'] = pair_to_div
        # note: some of the values will be NaN because they have no recombination events
        dnds_dfs.append(dnds_df)
    full_dnds_df = pd.concat(dnds_dfs)

    dnds_basepath = os.path.join(config.data_path, 'gut_microbiome_clonal_pair_dNdS')
    dnds_dfs = []
    for file in os.listdir(dnds_basepath):
        if file.startswith('.'):
            continue
        species_name = file.split('.')[0]
        dnds_df = pd.read_csv(os.path.join(dnds_basepath, species_name + '.csv'))
        dnds_dfs.append(dnds_df)
    clonal_dnds_df = pd.concat(dnds_dfs)
    clonal_dnds_df['clonal_diff_4D'] = clonal_dnds_df['core_diff_4D'].copy()
    clonal_dnds_df['clonal_len_4D'] = clonal_dnds_df['core_len_4D'].copy()
    clonal_dnds_df['clonal_diff_1D'] = clonal_dnds_df['core_diff_1D'].copy()
    clonal_dnds_df['clonal_len_1D'] = clonal_dnds_df['core_len_1D'].copy()

    complete_df = pd.concat([full_dnds_df, clonal_dnds_df])
    complete_df = complete_df[~complete_df['species_name'].isin(config.blacklist_species)]
    complete_df.set_index(['species_name', 'sample 1', 'sample 2'], inplace=True, )
    return complete_df


def load_typical_pair_dNdS_data():
    typical_pair_dnds_basepath = os.path.join(config.data_path, 'gut_microbiome_typical_pair_dNdS')
    dnds_dfs = []
    for file in os.listdir(typical_pair_dnds_basepath):
        if file.startswith('.'):
            continue
        species_name = file.split('.')[0]
        if species_name in config.blacklist_species:
            # some species are blacklisted since recombination detection is not reliable
            continue
        dnds_df = pd.read_csv(os.path.join(typical_pair_dnds_basepath, species_name + '.csv'))
        dnds_dfs.append(dnds_df)
    typical_dnds_df = pd.concat(dnds_dfs)
    typical_dnds_df.set_index(['species_name', 'sample 1', 'sample 2'], inplace=True)
    return typical_dnds_df

def find_clonal_pairs(snv_helper):
    """
    Find pairs of samples that are clonal, i.e. closely related but no detected recombination events in 
    Liu&Good 2024 data.
    """
    species = snv_helper.species_name
    all_transfers = load_detected_transfers()
    species_transfers = all_transfers[all_transfers['Species name']==species]
    samples = species_transfers[['Sample 1', 'Sample 2']].drop_duplicates()
    recomb_pairs = list(zip(samples['Sample 1'], samples['Sample 2']))

    # find unique combinations of Sample 1 and Sample 2 in species_transfers
    cphmm_df_path = config.LiuGood2024_path / "closely_related" / 'third_pass' / f'{species}.pickle'
    if not cphmm_df_path.exists():
        return []
    cphmm_df = pd.read_pickle(cphmm_df_path)
    close_pairs = list(cphmm_df.pairs.unique())

    # then map to sample names
    close_pairs = [(snv_helper.samples[i], snv_helper.samples[j]) for i, j in close_pairs]
    # finally remove pairs with recombination events
    # check both orderings of the pair
    for pair in recomb_pairs:
        if pair in close_pairs:
            close_pairs.remove(pair)
        elif pair[::-1] in close_pairs:
            close_pairs.remove(pair[::-1])
    return close_pairs


def fit_Poisson_rate_with_GLM(ns, ks):
    """
    Fit a Poisson rate to the data using a GLM with Poisson family.
    """
    try:
        model = sm.GLM(ks, np.ones(len(ks)), offset=np.log(ns), family=sm.families.Poisson())
        results = model.fit()
        return results.params[0], results.bse[0]
    except ValueError:
        return np.nan, np.nan

def fit_clonal_region_results(full_res):
    """
    Infer substitution rates at synonymous and non-synonymous sites for clonal and recombination events

    full_res: DataFrame of concatenated results from all species
    """
    freq_bins = np.logspace(-6, -2, 40)

    full_dS = full_res['clonal_diff_4D'] / full_res['clonal_len_4D'] / 3

    res_df = pd.DataFrame(index=list(range(len(freq_bins)-1)),
                        columns=['syn rates', 'syn SE', 'ns rates', 'ns SE', 'ms rates', 'ms SE'],
                        dtype=float)

    for i in range(len(freq_bins)-1):
        start = freq_bins[i]
        end = freq_bins[i+1]
        mask = (full_dS > start) & (full_dS < end)
        if mask.sum() < 2:
            res_df.loc[i] = np.nan
            continue
        sub_res = full_res[mask]
        ns, ks = sub_res['clonal_len_4D'].values, sub_res['clonal_diff_4D'].values
        res = fit_Poisson_rate_with_GLM(ns, ks)
        res_df.loc[i, 'syn rates'] = res[0]
        res_df.loc[i, 'syn SE'] = res[1]

        # divide by the number of possible mutations, which is 3
        ns, ks = sub_res['clonal_n'].values / 3, sub_res['clonal_mut_n'].values
        res = fit_Poisson_rate_with_GLM(ns, ks)
        res_df.loc[i, 'ns rates'] = res[0]
        res_df.loc[i, 'ns SE'] = res[1]

        ns, ks = sub_res['clonal_m'].values / 3, sub_res['clonal_mut_m'].values
        res = fit_Poisson_rate_with_GLM(ns, ks)
        res_df.loc[i, 'ms rates'] = res[0]
        res_df.loc[i, 'ms SE'] = res[1]

    # convert to 10-based log
    res_df = res_df * np.log10(np.e)
    return res_df

def fit_recomb_region_results(full_res):
    """
    Similar to fit_clonal_region_results, but for recombination regions
    """
    freq_bins = np.logspace(-6, -2, 40)

    full_dS = full_res['clonal_diff_4D'] / full_res['clonal_len_4D'] / 3

    res_df = pd.DataFrame(index=list(range(len(freq_bins)-1)),
                        columns=['syn rates', 'syn SE', 'recomb ns rates', 'recomb ns SE', 'recomb ms rates', 'recomb ms SE'],
                        dtype=float)

    for i in range(len(freq_bins)-1):
        start = freq_bins[i]
        end = freq_bins[i+1]
        mask = (full_dS > start) & (full_dS < end)
        if mask.sum() < 2:
            res_df.loc[i] = np.nan
            continue
        sub_res = full_res[mask]

        ns, ks = sub_res['clonal_len_4D'].values, sub_res['clonal_diff_4D'].values
        res = fit_Poisson_rate_with_GLM(ns, ks)
        res_df.loc[i, 'syn rates'] = res[0]
        res_df.loc[i, 'syn SE'] = res[1]

        ns, ks = sub_res['recomb_n'].values / 3, sub_res['recomb_mut_n'].values
        res = fit_Poisson_rate_with_GLM(ns, ks)
        res_df.loc[i, 'recomb ns rates'] = res[0]
        res_df.loc[i, 'recomb ns SE'] = res[1]

        ns, ks = sub_res['recomb_m'].values / 3, sub_res['recomb_mut_m'].values
        res = fit_Poisson_rate_with_GLM(ns, ks)
        res_df.loc[i, 'recomb ms rates'] = res[0]
        res_df.loc[i, 'recomb ms SE'] = res[1]
    # convert to 10-based log
    res_df = res_df * np.log10(np.e)
    return res_df

def load_and_fit_typical_pair_results():
    dat_path = config.data_path / 'gut_microbiome_typical_pair_dNdS'
    species_list = [f.name.split('.')[0] for f in dat_path.glob('*.csv')]

    typical_res_df = pd.DataFrame(index=species_list, columns=['syn rates', 'syn SE', 'ns rates', 'ns SE', 'ms rates', 'ms SE'],
                              dtype=float)
    for species in species_list:
        if species in config.blacklist_species:
            # some species are blacklisted since recombination detection is not reliable
            continue
        typical_pair_path = config.data_path / 'gut_microbiome_typical_pair_dNdS' / f'{species}.csv'
        typical_pair = pd.read_csv(typical_pair_path, index_col=0).reset_index()

        ns, ks = typical_pair['core_len_4D'].values, typical_pair['core_diff_4D'].values
        res = fit_Poisson_rate_with_GLM(ns, ks)
        typical_res_df.loc[species, 'syn rates'] = res[0]
        typical_res_df.loc[species, 'syn SE'] = res[1]

        # divide by the number of possible mutations, which is 3
        ns, ks = typical_pair['core_n'].values / 3, typical_pair['core_mut_n'].values
        res = fit_Poisson_rate_with_GLM(ns, ks)
        typical_res_df.loc[species, 'ns rates'] = res[0]
        typical_res_df.loc[species, 'ns SE'] = res[1]

        ns, ks = typical_pair['core_m'].values / 3, typical_pair['core_mut_m'].values
        res = fit_Poisson_rate_with_GLM(ns, ks)
        typical_res_df.loc[species, 'ms rates'] = res[0]
        typical_res_df.loc[species, 'ms SE'] = res[1]
    # convert to 10-based log
    typical_res_df = typical_res_df * np.log10(np.e)
    return typical_res_df