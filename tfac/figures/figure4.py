"""
This creates Figure 4. Gene expression R2X with flattened matrix dimension reconstruction.
"""
from .figureCommon import subplotLabel, getSetup
import numpy as np
import pandas as pd
import seaborn as sns
from tensorly.decomposition import  partial_tucker,parafac2
import tensorly as tl
from tensorly.parafac2_tensor import parafac2_to_slice, apply_parafac2_projections
from ..Data_Mod import form_tensor
from ..tensor import partial_tucker_decomp, find_R2X_partialtucker, flip_factors
from tensorly.metrics.regression import variance as tl_var
tl.set_backend("numpy")

def get_Flattened_Matrices(result, geneexpression, treatment_list, times):
    ids = geneexpression["ensembl_gene_id"]
    geneexpression.drop("ensembl_gene_id", inplace = True, axis = 1)
    #create a 5x42 DataFrame of decompsed component values
    toflatten = result[0]
    flattened = []

    for treatment in range(7):
        flattened.append(pd.DataFrame(toflatten[treatment]).T)

    df = pd.concat(flattened, axis=1)
    
    #component values for control at T=0
    ctrl = df.iloc[:, 0]

    #Remove all times that aren't 24 or 48
    df.drop([0,1,2,3], inplace = True, axis = 1)
    
    #add control back in
    ctrl = pd.DataFrame(ctrl)
    df = pd.concat([ctrl,df], axis = 1)

    #rename columns
    cols = [str(x) + "_" + str(y) for x in treatment_list for y in times[-2:]]
    cols = ["ctrl_0"] + cols
    df.columns = cols

    #sort by column name
    df = df.reindex(sorted(df.columns), axis=1)
    geneexpression = geneexpression.reindex(sorted(geneexpression.columns), axis=1)

    #make dataframes into numpy arrays
    df = df.to_numpy()
    genexpression = geneexpression.to_numpy()
    return df, genexpression

def get_reconstruct(P,X):
    Ppinv = np.linalg.pinv(P.T)
    return Ppinv, np.matmul(Ppinv,X.T)

def find_gene_factors(result, geneexpression, treatment_list, times):
    P, X = get_Flattened_Matrices(result, geneexpression, treatment_list, times)
    Ppinv, W = get_reconstruct(P, X)
    return P, X, Ppinv, W

def gene_R2X(axis):
    tensor, treatment_list, times = form_tensor()
    R2X = np.zeros(13)
    for i in range(1, 13):
        pre_flip_result = partial_tucker_decomp(tensor, [2], i)
        result = flip_factors(pre_flip_result)
        RNAseq = pd.read_csv("tfac/data/ohsu/MDD_RNAseq_Level4.csv")
        P, X, Ppinv, W  = find_gene_factors(result, RNAseq, treatment_list, times)
        Gene_redone = np.matmul(W.T, P)
        R2X[i] = 1 - tl_var(Gene_redone - X) / tl_var(RNAseq.to_numpy())
    sns.scatterplot(np.arange(len(R2X)), R2X, ax = axis)


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    row = 1
    col = 1
    ax, f = getSetup((11, 11), (row, col))

    gene_R2X(ax[0])
    subplotLabel(ax)
    return f
