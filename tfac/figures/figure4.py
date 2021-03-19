"""
This creates Figure 4. Gene expression R2X with flattened matrix dimension reconstruction.
"""
import numpy as np
import pandas as pd
import seaborn as sns
from tensorly.metrics.regression import variance as tl_var
from .figureCommon import subplotLabel, getSetup
from ..Data_Mod import form_tensor
from ..tensor import decomp_to_flipped_factors
from ..pseudoinvnorm import find_gene_factors


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    row = 1
    col = 1
    ax, f = getSetup((4, 4), (row, col))
    gene_R2X(ax[0])
    subplotLabel(ax)
    return f
    

def gene_R2X(axis):
    RNAseq = pd.read_csv("tfac/data/ohsu/MDD_RNAseq_Level4.csv")
    RNAseq.drop("ensembl_gene_id", inplace=True, axis=1)
    R2X = np.zeros(13)
    for i in range(1, 13):
        result, treatment_list, times = decomp_to_flipped_factors(i)
        P, X, _, W = find_gene_factors(result, RNAseq, treatment_list, times)
        Gene_redone = np.matmul(W.T, P)
        R2X[i] = 1 - tl_var(Gene_redone - X) / tl_var(RNAseq.to_numpy())
    sns.scatterplot(np.arange(len(R2X)), R2X, ax=axis)
    axis.set_xlabel("Components")
    axis.set_ylabel("Gene R2X")
    axis.set_yticks([0, .2, .4, .6, .8, 1])
    axis.set_title("Variance Explained in Gene Data")

