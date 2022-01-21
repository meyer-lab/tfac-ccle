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


def ohsu_data(export=False):
    """ Import OHSU data for PARAFAC2"""
    RNAseq = pd.read_csv(join(path_here, "tfac/data/ohsu/MDD_RNAseq_Level4.txt"), delimiter=",", index_col=0)
    # replace zeros with np.nan
    RNAseq = RNAseq.replace(0, np.nan)
    # remove rows with more than 10 zeros
    RNAseq = RNAseq.dropna(axis=0, thresh=5)
    RNAseq = RNAseq.replace(np.nan, 0)
    # column names
    cols = RNAseq.columns
    if export:
        RNAseq.apply(scale, axis=1)
        RNAseq.to_csv(join(path_here, "tfac/data/ohsu/RNAseq.txt"), sep='\t')

    return RNAseq


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

    df.drop('Control', inplace=True, level=0)
    df = df.sort_index()

    dfArray = df.to_numpy()
    tensor = np.reshape(dfArray, (-1, len(times), dfArray.shape[1]))

    # Subtract off control
    tensor -= tensor[0, 0, :]

    RNAseq = pd.read_csv(join(path_here, "tfac/data/ohsu/module_expression.csv"), delimiter=",")
    RNAseq.rename(columns={"Unnamed: 0": "gene_modules"}, inplace=True)

    # Copy over control
    for treatment in df.index.unique(level=0):
        RNAseq[treatment + "_0"] = RNAseq["ctrl_0"]

    RNAseq = RNAseq.set_index("gene_modules").T
    RNAseq.index = RNAseq.index.str.split('_', expand=True)
    RNAseq.index = RNAseq.index.set_levels(RNAseq.index.levels[1].astype(int), level=1)

    RNAseq.drop('ctrl', inplace=True, level=0)
    RNAseq = RNAseq.reindex(index=df.index)

    rArray = RNAseq.to_numpy()
    rTensor = np.reshape(rArray, (-1, len(times), rArray.shape[1]))

    # Normalize the data
    tensor -= np.mean(tensor, axis=(0, 1), keepdims=True) # proteins
    rTensor -= np.nanmean(rTensor, axis=(0, 1), keepdims=True) # genes

    # Match variance of both datasets
    tensor /= np.nansum(np.square(tensor))
    rTensor /= np.nansum(np.square(rTensor))

    assert rTensor.shape[0] == tensor.shape[0]
    assert rTensor.shape[1] == tensor.shape[1]

    return np.append(tensor, rTensor, axis=2), df.index.unique(level=0), times

"Will give a tensor of shape (7, 6, 666)"
"7 treatments, in this order: 'BMP2', 'EGF', 'HGF', 'IFNg', 'OSM', 'PBS', 'TGFb'"
"6 time points (in hours), in this order: 0.0, 1.0, 4.0, 8.0, 24.0, 48.0"
"295 protein data points + 371 gene data points = 666 (!) total data points"
