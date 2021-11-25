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

    df.drop('Control', inplace=True, level=0)
    df = df.sort_index()

    dfArray = df.to_numpy()
    tensor = np.reshape(dfArray, (-1, len(times), dfArray.shape[1]))

    # Subtract off control
    tensor -= tensor[0, 0, :]

    RNAseq = ohsu_data()

    # Copy over control
    for treatment in df.index.unique(level=0):
        RNAseq[treatment + "_0"] = RNAseq["ctrl_0"]

    RNAseq = RNAseq.set_index("ensembl_gene_id").T
    RNAseq.index = RNAseq.index.str.split('_', expand=True)
    RNAseq.index = RNAseq.index.set_levels(RNAseq.index.levels[1].astype(int), level=1)

    RNAseq.drop('ctrl', inplace=True, level=0)
    RNAseq = RNAseq.reindex(index=df.index)

    rArray = RNAseq.to_numpy()
    rTensor = np.reshape(rArray, (-1, len(times), rArray.shape[1]))

    # Normalize the data
    tensor -= np.mean(tensor, axis=(0, 1), keepdims=True)
    rTensor -= np.mean(rTensor, axis=(0, 1), keepdims=True)

    # Match variance of both datasets
    tensor /= np.nansum(np.square(tensor))
    RNAseq /= np.nansum(np.square(RNAseq))

    assert rTensor.shape[0] == tensor.shape[0]
    assert rTensor.shape[1] == tensor.shape[1]

    return tensor, rTensor, df.index.unique(level=0), times

"Will give a tensor of shape (7, 6, 57662)"
"7 treatments, in this order: 'BMP2', 'EGF', 'HGF', 'IFNg', 'OSM', 'PBS', 'TGFb'"
"6 time points (in hours), in this order: 0.0, 1.0, 4.0, 8.0, 24.0, 48.0"
"295 protein data points + 57367 gene data points = 57662 total data points"
def form_bigtensor():
    tensor, _, _, _ = form_tensor()
    RNAseq = ohsu_data()
    "change the rna sequence data to be the same as tensor protein data (7, 6)"
    RNAseq.drop("ensembl_gene_id", inplace=True, axis=1)
    cols = [ 'BMP2_24', 'BMP2_48', 'EGF_24', 'EGF_48','HGF_24', 'HGF_48', 'IFNg_24',   'IFNg_48', 'OSM_24', 'OSM_48',  'PBS_24','PBS_48', 'TGFb_24',
        'TGFb_48', 'ctrl_0']
    RNAseq = RNAseq[cols]
    RNAseq = RNAseq.drop(["ctrl_0"], axis=1)
    RNAseqnp = RNAseq.to_numpy()
    RNAseqnp = RNAseqnp.T
    "take the rna sequence and break out the time periods to be similar on two axes to the protein"
    RNAtensor = np.zeros((7,6,57367))
    "copy over the values from our rna sequence to the appropriate rows in the tensor"
    np.copyto(RNAtensor[0][4], RNAseqnp[0])
    np.copyto(RNAtensor[0][5], RNAseqnp[1])
    np.copyto(RNAtensor[1][4], RNAseqnp[2])
    np.copyto(RNAtensor[1][5], RNAseqnp[3])
    np.copyto(RNAtensor[2][4], RNAseqnp[4])
    np.copyto(RNAtensor[2][5], RNAseqnp[5])
    np.copyto(RNAtensor[3][4], RNAseqnp[6])
    np.copyto(RNAtensor[3][5], RNAseqnp[7])
    np.copyto(RNAtensor[4][4], RNAseqnp[8])
    np.copyto(RNAtensor[4][5], RNAseqnp[9])
    np.copyto(RNAtensor[5][4], RNAseqnp[10])
    np.copyto(RNAtensor[5][5], RNAseqnp[11])
    np.copyto(RNAtensor[6][4], RNAseqnp[12])
    np.copyto(RNAtensor[6][5], RNAseqnp[13])
    "finally append the two tensors together, with the protein data coming first"
    finaltensor = np.append(tensor, RNAtensor, axis=2)
    return finaltensor