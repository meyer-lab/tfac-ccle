"""Contains function for importing and handling OHSU data"""
from os.path import join, dirname
import numpy as np
import pandas as pd

path_here = dirname(dirname(__file__))


def importLINCSprotein():
    """ Import protein characterization from LINCS. """
    dataA = pd.read_csv(join(path_here, "tfac/data/01_Laura_Heiser__Sean_Gross_A.csv"))
    dataB = pd.read_csv(join(path_here, "tfac/data/01_Laura_Heiser__Sean_Gross_B.csv"))
    dataC = pd.read_csv(join(path_here, "tfac/data/01_Laura_Heiser__Sean_Gross_C.csv"))

    dataA["File"] = "A"
    dataB["File"] = "B"
    dataC["File"] = "C"

    return pd.concat([dataA, dataB, dataC])


def ohsu_data():
    """ Import OHSU data for PARAFAC2"""
    atac = pd.read_csv(join(path_here, "tfac/data/ohsu/MDD_ATACseq_Level4.csv"))
    cycIF = pd.read_csv(join(path_here, "tfac/data/ohsu/MDD_cycIF_Level4.csv"))
    GCP = pd.read_csv(join(path_here, "tfac/data/ohsu/MDD_GCP_Level4.csv"))
    IF = pd.read_csv(join(path_here, "tfac/data/ohsu/MDD_IF_Level4.csv"))
    L1000 = pd.read_csv(join(path_here, "tfac/data/ohsu/MDD_L1000_Level4.csv"))
    RNAseq = pd.read_csv(join(path_here, "tfac/data/ohsu/MDD_RNAseq_Level4.csv"))
    RPPA = pd.read_csv(join(path_here, "tfac/data/ohsu/MDD_RPPA_Level4.csv"))
    return atac, cycIF, GCP, IF, L1000, RNAseq, RPPA


def compProteins(comps):
    """Returns the top three weighted proteins for each component in input protein component matrix"""
    i = np.shape(comps)  # input tensor decomp output
    proteins = proteinNames()
    _, compNum = np.shape(comps[i[0] - 1])
    compName = []
    topProtein = []

    for x in range(0, compNum):
        compName.append("Col" + str(x + 1))

    dfComps = pd.DataFrame(data=comps[i[0] - 1], index=proteins, columns=compName)
    for y in range(0, compNum):
        topProtein.append(compName[y])
        rearranged = dfComps.sort_values(by=compName[y], ascending=False)
        rearrangedNames = list(rearranged.index.values)
        for z in range(0, 3):
            topProtein.append(rearrangedNames[z])

    return topProtein


def proteinNames():
    """Return protein names (data columns)"""
    data = importLINCSprotein()
    data = data.drop(columns=["Treatment", "Sample description", "File", "Time"], axis=1)
    proteinN = data.columns.values.tolist()
    return proteinN

def dataCleanUp():
    """Cleaning up OHSU data for PARAFAC2 column order"""
    atac, cycIF, GCP, _, L1000, RNAseq, RPPA = ohsu_data()
    tr = ['BMP2_', 'EGF_', 'HGF_', 'IFNG_', 'OSM_', 'TGFB_', 'PBS_', 'ctrl_0']
    for r in range(0, 7):
        cycIF = cycIF.drop(columns=[tr[r] + '1', tr[r] + '4', tr[r] + '8'])
        GCP = GCP.drop(columns=[tr[r] + '4', tr[r] + '8'])
        L1000 = L1000.drop(columns=[tr[r] + '1', tr[r] + '4', tr[r] + '8'])
        RPPA = RPPA.drop(columns=[tr[r] + '1', tr[r] + '4', tr[r] + '8'])
    atac = atac.sort_index(axis=1)
    chromosomes = atac['peak'].to_list()
    atac = atac.drop(columns='peak').to_numpy()
    cycIF = cycIF.sort_index(axis=1)
    IFproteins = cycIF['feature'].to_list()
    cycIF = cycIF.drop(columns='feature').to_numpy()
    GCP = GCP.dropna()
    GCP = GCP.sort_index(axis=1)
    histones = GCP['histone'].to_list()
    GCP = GCP.drop(columns='histone').to_numpy()
    L1000 = L1000.sort_index(axis=1)
    geneExpression = L1000['probeset'].to_list()
    L1000 = L1000.drop(columns='probeset').to_numpy()
    RNAseq = RNAseq.sort_index(axis=1)
    RNAGenes = RNAseq['ensembl_gene_id'].tolist()
    RNAseq = RNAseq.drop(columns='ensembl_gene_id').to_numpy()
    RPPA = RPPA.sort_index(axis=1)
    RPPAProteins = RPPA['antibody'].tolist()
    RPPA = RPPA.drop(columns='antibody').to_numpy()
    return atac, cycIF, GCP, L1000, RNAseq, RPPA, chromosomes, IFproteins, histones, geneExpression, RNAGenes, RPPAProteins
