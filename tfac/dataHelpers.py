"""Contains function for importing and handling OHSU data"""
from os.path import join, dirname
import numpy as np
import pandas as pd

path_here = dirname(dirname(__file__))


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
    