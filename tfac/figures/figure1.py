"""
This creates Figure 1 - Partial Tucker Decomposition Treatment and Time Plots.
"""
import numpy as np
import pandas as pd
import seaborn as sns
from .figureCommon import subplotLabel, getSetup, get_flipped_tucker
from ..tensor import partial_tucker_decomp, find_R2X_partialtucker, get_flipped_tucker
from ..Data_Mod import form_tensor
from ..pseudoinvnorm import get_Flattened_Matrices, get_reconstruct, find_gene_factors
from ..dataHelpers import importLINCSprotein
from tensorly.decomposition import  partial_tucker,parafac2
import tensorly as tl
from tensorly.parafac2_tensor import parafac2_to_slice, apply_parafac2_projections
from tensorly.metrics.regression import variance as tl_var
tl.set_backend("numpy")

tensor, treatment_list, times = form_tensor()

def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    row = 1
    col = 2
    ax, f = getSetup((12, 6), (row, col))

    R2X_Figure_PartialTucker(ax[0], tensor)
    gene_R2X(ax[1])

    # Add subplot labels
    subplotLabel(ax)

    return f


def R2X_Figure_PartialTucker(ax, input_tensor):
    '''Create Partial Tucker R2X Figure'''
    R2X = np.zeros(13)
    for i in range(1, 13):
        output = partial_tucker_decomp(input_tensor, [2], i)
        R2X[i] = find_R2X_partialtucker(output, input_tensor)
    sns.scatterplot(np.arange(len(R2X)), R2X, ax=ax)
    ax.set_xlabel("Components")
    ax.set_ylabel("Protein R2X")
    ax.set_yticks([0, .2, .4, .6, .8, 1])

def gene_R2X(axis):
    tensor, treatment_list, times = form_tensor()
    R2X = np.zeros(13)
    for i in range(1, 13):
        print(i)
        result = get_flipped_tucker(tensor, i)
        RNAseq = pd.read_csv("tfac/data/ohsu/MDD_RNAseq_Level4.csv")
        P, X, Ppinv, W  = find_gene_factors(result, RNAseq, treatment_list, times)
        Gene_redone = np.matmul(W.T, P)
        R2X[i] = 1 - tl_var(Gene_redone - X) / tl_var(RNAseq.to_numpy())
    sns.scatterplot(np.arange(len(R2X)), R2X, ax = axis)
    axis.set_xlabel("Components")
    axis.set_ylabel("Gene R2X")
    axis.set_yticks([0, .2, .4, .6, .8, 1])