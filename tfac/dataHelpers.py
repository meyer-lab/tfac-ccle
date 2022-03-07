"""Contains function for importing and handling OHSU data"""
from os.path import join, dirname
import numpy as np
from sklearn.preprocessing import scale
import pandas as pd
import scipy.cluster.hierarchy as sch

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


def reorder_table(df):
    """ Reorder a table's rows using heirarchical clustering. """
    # Reorder measurements based on similarity
    Y = sch.linkage(df.to_numpy(), method='centroid')
    index = sch.dendrogram(Y, orientation='right')['leaves']
    return df.iloc[index, :]


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


def import_LINCS_MEMA(datafile):
    """ Ligand, ECM, and phenotypic measurements of cells from LINCS MEMA dataset. """
    data = pd.read_csv(join(path_here, "tfac/data/ohsu/", datafile), index_col=["Ligand", "ECMp"], delimiter="\t", low_memory=False)
    data = data.reset_index()
    missingCols = data.columns[data.isna().any()]
    assert len(missingCols) < 15
    data = data.dropna(axis=1)  # remove columns with no measurements
    data.drop(list(data.filter(regex='Conc')), axis=1, inplace=True)
    data.drop(list(data.filter(regex='Feret')), axis=1, inplace=True)
    data.drop(list(data.filter(regex='Orientation')), axis=1, inplace=True)
    data.drop(list(data.filter(regex='_SE')), axis=1, inplace=True)
    data.drop(list(data.filter(regex='LoessSCC')), axis=1, inplace=True)
    measurements = data.columns[data.dtypes == float]

    ligands = pd.unique(data["Ligand"])
    ECMp = pd.unique(data["ECMp"])
    tensor = np.empty((ligands.size, ECMp.size, len(measurements)))

    for ii, ECM in enumerate(ECMp):
        dataECM = data.loc[data["ECMp"] == ECM]

        for jj, ligs in enumerate(ligands):
            selected = dataECM.loc[dataECM["Ligand"] == ligs, measurements]
            tensor[jj, ii, :] = selected.iloc[0, :]
            assert selected.shape[0] == 1

    tensor -= np.nanmean(tensor, axis=(0, 1), keepdims=True)
    tensor /= np.nanstd(tensor, axis=(0, 1), keepdims=True)
    return tensor, ligands, ECMp, measurements
