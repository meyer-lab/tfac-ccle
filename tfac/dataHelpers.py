"""Contains function for importing and handling OHSU data"""
from os.path import join, dirname
import numpy as np
from sklearn.preprocessing import scale
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
    RNAseq = pd.read_csv(join(path_here, "tfac/data/ohsu/MDD_RNAseq_Level4.csv"), delimiter=",", index_col=0)

    row_avg = RNAseq.mean(axis=1)
    for indx in RNAseq.index:
        if row_avg[indx] <= 0.07:
            RNAseq.drop(indx, inplace=True)

    # column names
    cols = RNAseq.columns
    if export:
        RNAseq = RNAseq.apply(scale, axis=1, result_type='expand')
        RNAseq.columns = cols
        RNAseq.to_csv(join(path_here, "tfac/data/ohsu/RNAseq.txt"), sep='\t')

    return RNAseq


def proteinNames():
    """Return protein names (data columns)"""
    data = importLINCSprotein()
    data = data.drop(columns=["Treatment", "Sample description", "File", "Time"], axis=1)
    return data.columns.values.tolist()


def import_LINCS_CCLE():
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

    RNAseq = pd.read_csv(join(path_here, "tfac/data/ohsu/module_expression.csv"), sep=',')
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
    tensor -= np.mean(tensor, axis=(0, 1), keepdims=True)  # proteins
    rTensor -= np.nanmean(rTensor, axis=(0, 1), keepdims=True)  # genes

    # Match variance of both datasets
    tensor /= np.nansum(np.square(tensor))
    rTensor /= np.nansum(np.square(rTensor))

    # scale the proteins based on analysis
    tensor = tensor * 4**-1

    assert rTensor.shape[0] == tensor.shape[0]
    assert rTensor.shape[1] == tensor.shape[1]

    return np.append(tensor, rTensor, axis=2), df.index.unique(level=0), times


"Will give a tensor of shape (7, 6, 498)"
"7 treatments, in this order: 'BMP2', 'EGF', 'HGF', 'IFNg', 'OSM', 'PBS', 'TGFb'"
"6 time points (in hours), in this order: 0.0, 1.0, 4.0, 8.0, 24.0, 48.0"
"295 protein data points + 203 gene data points = 498 total data points"

def import_LINCS_MEMA(datafile):
    """ Cell behavior and phenotypic measurements of MCF10A cells. """

    data = pd.read_csv(join(path_here, datafile), index_col=["Ligand", "ECMp"], delimiter="\t", low_memory=False)
    data = data.reset_index()
    data = data.dropna(axis=1)  # remove columns with no measurements
    data.drop(list(data.filter(regex = '.tsv')), axis = 1, inplace = True)
    data.drop(list(data.filter(regex = '_SE')), axis = 1, inplace = True)
    data.drop(list(data.filter(regex = 'Feret')), axis = 1, inplace = True)
    data.drop(list(data.filter(regex = 'Gated')), axis = 1, inplace = True)
    data.drop(list(data.filter(regex = 'Norm')), axis = 1, inplace = True)
    data.drop(list(data.filter(regex = 'Conc')), axis = 1, inplace = True)
    measurements = data.columns[data.dtypes == float]

    tensor = np.empty((pd.unique(data["Ligand"]).size, pd.unique(data["ECMp"]).size, len(measurements)))

    for ii, ECM in enumerate(pd.unique(data["ECMp"])):
        dataECM = data.loc[data["ECMp"] == ECM]

        for jj, ligs in enumerate(pd.unique(data["Ligand"])):
            selected = dataECM.loc[dataECM["Ligand"] == ligs, measurements]
            tensor[jj, ii, :] = selected.iloc[0, :]

    tensor -= np.mean(tensor, axis=(0, 1), keepdims=True)
    assert np.all(np.isfinite(tensor))
    tensor /= np.std(tensor, axis=(0, 1), keepdims=True)

    goods = np.all(np.isfinite(tensor), axis=(0, 1))
    measurements = measurements[goods]
    tensor = tensor[:, :, goods]
    return tensor, pd.unique(data["Ligand"]), pd.unique(data["ECMp"]), measurements
