import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sb
import sklearn as sk
import statsmodels as sm
import sys
import os

from pathlib import Path


def rnaseq_load(cancer_type):
    ''' Load RnaSeq dataFrame and remove first row '''
    data_path = Path('data')
    file_name = data_path / 'tcga' / f"{cancer_type}.rnaseqv2__illuminahiseq_rnaseqv2__unc_edu__Level_3__RSEM_genes_normalized__data.data.txt"
    print(f"Loading file: '{file_name}'")
    df = pd.read_csv(file_name, sep="\t", index_col=0, low_memory=False)
    df.drop('gene_id', inplace=True) # Remove second row, we don't use it
    df = df.astype('float')  # Convert values to float
    df.index.name = None  # Remove index name
    return df


def rnaseq_save(df, cancer_type, name):
    ''' Save RnaSeq dataFrame '''
    data_path = Path('data')
    file_name = data_path / 'tcga' / f"{cancer_type}.{name}.csv"
    print(f"Saving to file: '{file_name}'")
    df.to_csv(file_name)


def is_tumor(bar_code):
    ''' Is the barcode tumor? '''
    tissue_type = bar_code.split('-')[3]
    return tissue_type[0] == '0' # First character '0' indicates tumor


def get_normals(df):
    """ Get a dataframe with normal samples """
    tumors = np.array([is_tumor(c) for c in df.columns])
    return df.iloc[:, ~tumors].copy()


def get_tumors(df):
    """ Get a dataframe with tumor samples """
    tumors = np.array([is_tumor(c) for c in df.columns])
    return df.iloc[:, tumors].copy()


def is_valid_gene_name(genes):
    " Invalid gene names start with '?' "
    return [g[0] != '?' for g in genes]


def filter_invalid_genes(df):
    " Remove invalid genes from dataframe "
    keep = np.array(is_valid_gene_name(df.index.values))
    return df.iloc[keep, :]


def filter_duplicated_gene_names(df):
    " Remove rows with duplicated gene names (dataframe index) "
    gene_names = np.array([g.split('|')[0] for g in df.index.values])
    un, uc = np.unique(gene_names, return_counts=True)
    gene_dups = set(un[uc > 1])
    keep = np.array([g not in gene_dups for g in gene_names])
    return df.iloc[keep, :]


def filter_low_normals_count(df, normals_min=30):
    " Remove rows having number of normals < 'normals_min' "
    df_normals = get_normals(df)
    keep = ((df_normals > 0).sum(axis=1) >= normals_min).values
    return df.iloc[keep, :]


def filter_too_many_missing(df, count_min=0.9):
    """ Filter out genes having too many missing values """
    num_samples = df.shape[1]
    keep = ((df > 0).sum(axis=1) / num_samples >= count_min).values
    return df.iloc[keep, :]


def normality_test(x, use_logp1, count_min=30):
    """ Check is a vector 'x' is normally distributed
    (or normally distributed after log+1) 
    Return: Test's p-value, e.g. if p-value < 0.05, the variable
            might not be normal
    """
    try:
        # We need at least 'normals_min' values to calculate normality test
        if (x > 0).sum() < count_min:
            return np.nan
        x = np.log(x + 1) if use_logp1 else x
        norm_test = sm.api.stats.diagnostic.kstest_normal(x)
        return norm_test[1]  # This is the p-value
    except ValueError:
        return np.nan  # Test failed


def normality(df, use_logp1):
    """ Do a normality test, or a log-normality test if '''use_log' is set """
    pvals = [normality_test(df.iloc[i,], use_logp1) for i in range(df.shape[0])]
    pvals = np.array(pvals)
    print(f"Failed tests: {np.isnan(pvals).sum()}")
    pvals = np.nan_to_num(pvals, nan=0) # Test failed? Assume not normal
    return pvals


def filter_non_normals(df, pval_threshold=0.05, use_logp1=True):
    """ Filter out genes having non-gausian distributions in the normal samples """
    df_normals = get_normals(df)
    pvals = normality(df_normals, use_logp1)
    keep = (pvals > pval_threshold)
    return df.iloc[keep, :]


def logp1_normalize(df):
    """ Convert to log + 1 and then normalize (based on normal samples)"""
    df_log = np.log(df + 1)
    df_log_norms = get_normals(df_log)
    mean_normals = df_log_norms.mean(axis=1)
    std_normals = df_log_norms.std(axis=1)
    return df_log.sub(mean_normals, axis=0).div(std_normals, axis=0)


def rename_genes(df):
    """ Rename genes: Remove ID part from index """
    df.index = [g.split('|')[0] for g in df.index]
    return df

filters_default = [filter_invalid_genes
           , filter_duplicated_gene_names
           , filter_low_normals_count
           , filter_too_many_missing
           , filter_non_normals
           ]

transforms_default = [logp1_normalize, rename_genes]


def apply_all(x, funcs):
    for f in funcs:
        print(f"Applying {f.__name__}, x.shape: {x.shape}")
        x = f(x)
    return x


def load_filter_transform(cancer_type, filters=filters_default, transforms=transforms_default):
    """ Load, filter and transform a data frame """
    df = rnaseq_load(cancer_type)
    df = apply_all(df, filters)
    df = apply_all(df, transforms)
    return df


