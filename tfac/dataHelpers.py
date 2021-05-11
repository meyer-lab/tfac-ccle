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

    cols = dataA.columns
    for col in cols:
        dataA[col] = (dataA[col] - dataA[col].mean())/dataA[col].std(ddof=0)

    cols = dataB.columns
    for col in cols:
        dataB[col] = (dataB[col] - dataB[col].mean())/dataB[col].std(ddof=0)

    cols = dataC.columns
    for col in cols:
        dataC[col] = (dataC[col] - dataC[col].mean())/dataC[col].std(ddof=0)

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


    #TODO ADD COLUMNS FOR HOURS 4 and 8 to be filled with NaN
    #TODO TRANSPOSE AND THEN ZSCORE

    cols = RNAseq.columns
    for col in cols:
        RNAseq[col] = (RNAseq[col] - RNAseq[col].mean())/RNAseq[col].std(ddof=0)
    
    return atac, cycIF, GCP, IF, L1000, RNAseq, RPPA


def proteinNames():
    """Return protein names (data columns)"""
    data = importLINCSprotein()
    data = data.drop(columns=["Treatment", "Sample description", "File", "Time"], axis=1)
    proteinN = data.columns.values.tolist()
    return proteinN

def all_data_import():
    """ Import protein characterization from LINCS and OHSU (z-scoring both)"""
    dataA = pd.read_csv(join(path_here, "tfac/data/01_Laura_Heiser__Sean_Gross_A.csv"))
    dataB = pd.read_csv(join(path_here, "tfac/data/01_Laura_Heiser__Sean_Gross_B.csv"))
    dataC = pd.read_csv(join(path_here, "tfac/data/01_Laura_Heiser__Sean_Gross_C.csv"))

    cols = dataA.columns
    for col in cols[3:]:
        dataA[col] = (dataA[col] - dataA[col].mean())/dataA[col].std(ddof=0)

    cols = dataB.columns
    for col in cols[3:]:
        dataB[col] = (dataB[col] - dataB[col].mean())/dataB[col].std(ddof=0)

    cols = dataC.columns
    for col in cols[3:]:
        dataC[col] = (dataC[col] - dataC[col].mean())/dataC[col].std(ddof=0)

    dataA["File"] = "A"
    dataB["File"] = "B"
    dataC["File"] = "C"

    proteins =  pd.concat([dataA, dataB, dataC])

    atac = pd.read_csv(join(path_here, "tfac/data/ohsu/MDD_ATACseq_Level4.csv"))
    cycIF = pd.read_csv(join(path_here, "tfac/data/ohsu/MDD_cycIF_Level4.csv"))
    GCP = pd.read_csv(join(path_here, "tfac/data/ohsu/MDD_GCP_Level4.csv"))
    IF = pd.read_csv(join(path_here, "tfac/data/ohsu/MDD_IF_Level4.csv"))
    L1000 = pd.read_csv(join(path_here, "tfac/data/ohsu/MDD_L1000_Level4.csv"))
    RNAseq = pd.read_csv(join(path_here, "tfac/data/ohsu/MDD_RNAseq_Level4.csv"))
    RPPA = pd.read_csv(join(path_here, "tfac/data/ohsu/MDD_RPPA_Level4.csv"))


    treats = ["PBS","BMP2","IFNG","TGFB","HGF","OSM","EGF"]
    nums = ["0","4","8"]

    temp = RNAseq.T
    cols = temp.columns
    for col in cols[1:]:
        temp[col] = (temp[col] - temp[col].mean())/temp[col].std(ddof=0)

    RNAseq = temp.T

    for treat in treats:
        for num in nums:
            RNAseq[treat + "_" + num] = np.nan
    


    return proteins, atac, cycIF, GCP, IF, L1000, RNAseq, RPPA