"""Contains function for importing and handling OHSU data"""
from os.path import join, dirname
import numpy as np
from sklearn.preprocessing import scale
import pandas as pd
import scipy.cluster.hierarchy as sch
import xarray as xa

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

def proteinNames():
    """Return protein names (data columns)"""
    data = importLINCSprotein()
    data = data.drop(columns=["Treatment", "Sample description", "File", "Time"], axis=1)
    return data.columns.values.tolist()

def process_proteins():
    """ import, group, and scale proteins into a dataframe, ready to be merge with RNAseq into a tensor. """
    df = importLINCSprotein()
    df.drop(columns=["Sample description", "File"], inplace=True)
    times = pd.unique(df["Time"])

    # Group replicates and average
    df = df.groupby(["Treatment", "Time"]).mean()

    for treatment in df.index.unique(level=0):
        df.loc[(treatment, 0), :] = df.loc[('Control', 0)].values

    df.drop('Control', inplace=True, level=0)
    df = df.sort_index()

    return df.apply(scale, axis=1, raw=True)

def importLINCSRNAseq(export=False):
    """ Import RNAseq data of MCF10A cells for growthfactor treatments from OHSU LINCS data. This function is only used in gene module analysis. """

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

def process_RNAseq(df):
    """ Import gene modules, group, and scale into a dataframe, ready to be merged with the proteins data.
    Takes the dataframe of proteins as input for treatment labels. """

    # import the RNAseq data and pre-process
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

    return RNAseq.apply(scale, axis=1, raw=True)

def import_LINCS_CCLE():
    """ Creates tensor in numpy array form and returns tensor, treatments, and time.
    Returns both the protein and RNAseq tensors in aligned format. """

    # import the proteins
    df = process_proteins()

    # import the RNAseq data
    RNAseq = process_RNAseq(df)

    # concatenate proteins and RNAseq data
    fullDF = pd.concat([df, RNAseq], axis=1)

    xdf = fullDF.to_xarray().transpose()

    return xdf.to_array().transpose()

def Tensor_LINCS_MEMA(datafile):
    """ Ligand, ECM, and phenotypic measurements of cells from LINCS MEMA dataset. """
    data = pd.read_csv(join(path_here, "tfac/data/ohsu/", datafile), index_col=["Ligand", "ECMp"], delimiter="\t", low_memory=False)

    missingCols = data.columns[data.isna().any()]
    assert len(missingCols) < 15
    data = data.dropna(axis=1)  # remove columns with no measurements
    data.drop(list(data.filter(regex='Conc')), axis=1, inplace=True)
    data.drop(list(data.filter(regex='Feret')), axis=1, inplace=True)
    data.drop(list(data.filter(regex='Orientation')), axis=1, inplace=True)
    data.drop(list(data.filter(regex='_SE')), axis=1, inplace=True)
    data.drop(list(data.filter(regex='LoessSCC')), axis=1, inplace=True)
    data = data.loc[:, data.dtypes == float]
    data.iloc[:, :] = scale(data)

    xdf = data.to_xarray().to_array(dim="Measurement")
    return xdf

def Tensor_LINCS_CycIF():
    """ Imports the cyclic immunofluorescence data from LINCS. """
    data = pd.read_csv(join(path_here, "tfac/data/CycIF/MDD_cycIF_Level4.csv"), delimiter=",", index_col=0) # data size: 660 x 36

    ctrl = data['ctrl_0']
    data.drop(columns='ctrl_0', inplace=True)
    data = data.T
    datacopy = data.copy()

    # to add the control at the start of each treatment
    datacopy.index = datacopy.index.str.split('_', expand=True) # split the treatment name and the time index
    for tr in list(datacopy.index.levels[0]):
        data = pd.concat([pd.DataFrame({tr+'_0': ctrl}).T, data])

    # split the indexes to the treatment and time indexes
    data.index = data.index.str.split('_', expand=True)
    data = data.sort_index(level=1) # sort with respect to the treatments
    data = data.loc[:, data.dtypes == float]
    data.iloc[:, :] = scale(data)

    data.index = pd.MultiIndex.from_tuples(data.index, names=["treatment", "time"]) # 42 x 660

    # a function to only split a string at a specific underscore; in this case we split at the third underscore
    def split_at(string, n):
        words = string.split('_')
        return '_'.join(words[:n]), '_'.join(words[n:])

    # split the data by staining and measurements

    new_cols = []
    for col in data.columns:
        new_cols.append(split_at(col, 3))
    data.columns = pd.MultiIndex.from_tuples(new_cols, names=["stains", "measures"])

    data = data.T.unstack(level=0)
    xdf = data.T.to_xarray().to_array('measures')

    return xdf

def reorder_table(df):
    """ Reorder a table's rows using heirarchical clustering. """
    # Reorder measurements based on similarity
    Y = sch.linkage(df.to_numpy(), method='centroid')
    index = sch.dendrogram(Y, orientation='right')['leaves']
    return df.iloc[index, :]
