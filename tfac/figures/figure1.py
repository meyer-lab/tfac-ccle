"""
This creates Figure 1 - Partial Tucker R2X and Gene Projected Factors R2X.
"""
import numpy as np
import pandas as pd
import seaborn as sns
from tensorly.metrics.regression import variance as tl_var
from .figureCommon import subplotLabel, getSetup
from ..tensor import partial_tucker_decomp, find_R2X_partialtucker, decomp_to_flipped_factors
from ..Data_Mod import form_tensor
from ..pseudoinvnorm import find_factors
from ..dataHelpers import importLINCSprotein, ohsu_data


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    row = 1
    col = 2
    ax, f = getSetup((8, 4), (row, col))
    tensor, _, _ = form_tensor()
    R2X_Figure_PartialTucker(ax[0], tensor)
    gene_R2X(ax[1])

    # Add subplot labels
    subplotLabel(ax)

    return f


def R2X_Figure_PartialTucker(ax, input_tensor):
    """Create Partial Tucker R2X Figure"""
    R2X = np.zeros(13)
    for i in range(1, 13):
        output = partial_tucker_decomp(input_tensor, [2], i)
        R2X[i] = find_R2X_partialtucker(output, input_tensor)
    sns.scatterplot(np.arange(len(R2X)), R2X, ax=ax)
    ax.set_xlabel("Components")
    ax.set_ylabel("Protein R2X")
    ax.set_yticks([0, .2, .4, .6, .8, 1])


def gene_R2X(axis):
    _, _, _, _, _, RNAseq, _ = ohsu_data()
    RNAseq.drop("ensembl_gene_id", inplace=True, axis=1)
    R2X = np.zeros(13)
    for i in range(1, 13):
        result, treatment_list, times = decomp_to_flipped_factors(i)
        P, X, _, W = find_factors(result, RNAseq, treatment_list, times)
        Gene_redone = np.matmul(W.T, P)
        R2X[i] = 1 - tl_var(Gene_redone - X) / tl_var(RNAseq.to_numpy())
    sns.scatterplot(np.arange(len(R2X)), R2X, ax=axis)
    axis.set_xlabel("Components")
    axis.set_ylabel("Gene R2X")
    axis.set_yticks([0, .2, .4, .6, .8, 1])
