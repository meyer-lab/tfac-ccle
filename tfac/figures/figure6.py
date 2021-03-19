"""This creates Figure 6. Bar plot for percent variance explained by each gene expression component.
"""
import numpy as np
import pandas as pd
import seaborn as sns
from tensorly.metrics.regression import variance as tl_var
from .figureCommon import subplotLabel, getSetup
from ..dataHelpers import ohsu_data
from ..tensor import decomp_to_flipped_factors
from ..pseudoinvnorm import find_factors


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    row = 1
    col = 1
    ax, f = getSetup((4, 4), (row, col))
    var_diff(ax[0])
    subplotLabel(ax)
    return f


def var_diff(axis):
    """Calculates amount of variance each variance explains from each component of gene expression factors."""
    result, treatment_list, times = decomp_to_flipped_factors(5)
    _, _, _, _, _, RNAseq, _ = ohsu_data()
    RNAseq.drop("ensembl_gene_id", inplace=True, axis=1)
    P, _, _, W = find_factors(result, RNAseq, treatment_list, times)
    X_orig = RNAseq.to_numpy()
    R2X_full = tl_var(X_orig - np.matmul(W.T, P)) / tl_var(X_orig)

    residuals = np.zeros(5)
    for i in range(1, 6):
        # removes respective gene expression and treatment-time per iteration
        removeGene = np.delete(W, i - 1, 0)
        removeTT = np.delete(P, i - 1, 0)
        # reconstructs factors with removed row/col combo
        gene_reconst = np.matmul(removeGene.T, removeTT)
        # calculates the percent variance between remove-one factors
        residuals[i - 1] = tl_var(X_orig - gene_reconst) / tl_var(X_orig)
    sns.barplot(x=np.arange(len(residuals)), y=residuals - R2X_full, ax=axis)
    axis.set_xlabel("Component Removed")
    axis.set_ylabel("Difference in Percent Variance")
    axis.set_xticklabels(["1", "2", "3", "4", "5"])
    axis.set_title("Percent Variance Explained Per Component")

