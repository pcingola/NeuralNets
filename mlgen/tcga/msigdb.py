
def gene_genesets(msigdb):
    """ Iterate over all (geneset, gene) pairs from MsigDb dictionary """
    ggs = dict()
    for gs, genes in msigdb.items():
        for gene in genes.keys():
            if gene not in ggs:
                ggs[gene] = gs
            else:
                ggs[gene] += f" {gs}"
    return ggs


def geneset_gene_pairs(msigdb):
    """ Iterate over all (geneset, gene) pairs from MsigDb dictionary """
    for gs, genes in msigdb.items():
        for gene in genes.keys():
            yield gs, gene


def msigdb2df(path):
    """ Read all MsigDb in the path and create a dataframe """
    msigdb = read_msigdb_all(path)
    df = pd.DataFrame(msigdb, dtype='int8', index=msigdb2genes(msigdb), columns=msigdb2gene_sets(msigdb))
    df.fillna(0, inplace=True)
    return df.transpose()


def msigdb2genes(msigdb):
    """ Get a (sorted) list of all genes in MsigDb """
    genes = set([g for by_gene in msigdb.values() for g in by_gene.keys()])
    genes = list(genes)
    genes.sort()
    return genes


def msigdb2gene_sets(msigdb):
    """ Get a (sorted) list of all Gene-Sets in MsigDb """
    gs = list(msigdb.keys())
    gs.sort()
    return gs

def read_msigdb(path):
    """ Read an MsigDb (single) file and return a dictionary (by gene set) of dictionaries (gene name: 1) """
    msigdb = dict()
    for line in path.read_text().split('\n'):
        fields = line.split('\t')
        geneset_name = fields[0]
        if geneset_name:
            msigdb[geneset_name] = {f: 1 for f in fields[2:]}
    return msigdb


def read_msigdb_all(path, regex="*.gmt"):
    """ Read all MsigDb files, return a dictionary of lists of gene names """
    msigdb = dict()
    for p in path.find_files(regex):
        print(f"File: {p}")
        msigdb.update(read_msigdb(p))
    return msigdb


def save_gene_geneset(msigdb, path_save):
    pairs_str = '\n'.join([f"{g},{gss}" for g,gss in gene_genesets(msigdb).items()])
    path_save.write_text(pairs_str)

