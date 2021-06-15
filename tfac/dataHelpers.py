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
    return pd.read_csv(join(path_here, "tfac/data/ohsu/MDD_RNAseq_Level4.csv"))


def proteinNames():
    """Return protein names (data columns)"""
    data = importLINCSprotein()
    data = data.drop(columns=["Treatment", "Sample description", "File", "Time"], axis=1)
    return data.columns.values.tolist()


def form_tensor():
    """ Creates tensor in numpy array form and returns tensor, treatments, and time.
    Returns both the protein and RNAseq tensors in aligned format. """
    df = importLINCSprotein()
    df.drop(columns=["Sample description", "File"], inplace=True)
    times = pd.unique(df["Time"])

    # Group replicates and average
    df = df.groupby(["Treatment", "Time"]).mean()

    for treatment in df.index.unique(level=0):
        df.loc[(treatment, 0), :] = df.loc[('Control', 0)].values

    df.drop('Control', inplace=True)
    df = df.sort_index()
    
    dfArray = df.to_numpy()
    tensor = np.reshape(dfArray, (-1, len(times), dfArray.shape[1]))

    RNAseq = ohsu_data()

    # Copy over control
    for treatment in df.index.unique(level=0):
        RNAseq[treatment + "_0"] = RNAseq["ctrl_0"]

    RNAseq = RNAseq.set_index("ensembl_gene_id").T
    RNAseq.index = RNAseq.index.str.split('_',expand=True)
    RNAseq.index = RNAseq.index.set_levels(RNAseq.index.levels[1].astype(int), level=1)

    RNAseq.drop('ctrl', inplace=True)
    RNAseq = RNAseq.reindex(index=df.index)

    rArray = RNAseq.to_numpy()
    rTensor = np.reshape(rArray, (-1, len(times), rArray.shape[1]))

    assert rTensor.shape[0] == tensor.shape[0]
    assert rTensor.shape[1] == tensor.shape[1]

    return tensor, rTensor, df.index.unique(level=0), times
