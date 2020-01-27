import pandas as pd

from .util import *


# Which variants are expected to be 'high impact'?
high_impact_variants = set(['Frame_Shift_Del', 'Frame_Shift_Ins', 'In_Frame_Del',
                            'In_Frame_Ins', 'Indel', 'Intron', 'Missense',
                            'Missense_Mutation', 'Nonsense_Mutation', 'Nonstop_Mutation',
                            'Read-through', 'Splice_Site', 'Splice_Site_Del',
                            'Splice_Site_Ins', 'Splice_Site_SNP', 'Translation_Start_Site'])


def path_to_sample(path):
    """ Return the sample name from MAF file path """
    return path.stem.split('.')[0]


# A simple test case
assert 'TCGA-IH-A3EA-01' == path_to_sample(Path('data/tcga/gdac.broadinstitute.org_SKCM.Mutation_Packager_Calls.Level_3.2016012800.0.0/TCGA-IH-A3EA-01.maf.txt'))


def path_to_cancer_type(path):
    """ Return the cancre type MAF file path
    For example, path='data/tcga/gdac.broadinstitute.org_SKCM.Mutation_Packager_Calls.Level_3.2016012800.0.0/TCGA-IH-A3EA-01.maf.txt'
    we want to return 'SKCM'
    """
    return path.parent.stem.split('.')[2].split('_')[1]


# A simple test case
assert 'SKCM' == path_to_cancer_type(Path('data/tcga/gdac.broadinstitute.org_SKCM.Mutation_Packager_Calls.Level_3.2016012800.0.0/TCGA-IH-A3EA-01.maf.txt'))


def mutated_genes(maf_file):
    """ Parse the MAF file and return a dictionary of genes with
    the count of 'high impact' mutations per gene """
    # print(f"Mutated genes: '{maf_file}'")
    try:
        df = pd.read_csv(maf_file, sep='\t', low_memory=False)
    except UnicodeDecodeError:
        print(f"ERROR reading file '{maf_file}'")
        return None
    keep_rows = [vc in high_impact_variants for vc in df.Variant_Classification]
    count_by_gene = dict()
    for gene in df[keep_rows].Hugo_Symbol:
        count_by_gene[gene] = count_by_gene.get(gene, 0) + 1
    return count_by_gene


def process_maf_dir(path):
    """ Find all MAF files in 'path' and get a dictionary of
    mutated genes per cancer_type and sample """
    by_cancer_type = dict()
    for p in path.find_files('*maf.txt'):
        cancer_type = path_to_cancer_type(p)
        if cancer_type not in by_cancer_type:
            print(f"Adding cancer type '{cancer_type}'")
            by_cancer_type[cancer_type] = dict()
        sample = path_to_sample(p)
        by_cancer_type[cancer_type][sample] = mutated_genes(p)
    return by_cancer_type


def get_all_genes(by_sample):
    """ Return a (sorted) list of genes in all samples """
    all_genes = set([g for gene_dict in by_sample.values() if gene_dict is not None for g in gene_dict.keys()])
    all_genes = list(all_genes)
    all_genes.sort()
    return all_genes


def variants_df(by_sample):
    """ Create a dataframe containing number of variants per sample """
    df = pd.DataFrame(by_sample, index=get_all_genes(by_sample), columns=sorted(list(by_sample.keys())))
    return df.fillna(0).transpose()
