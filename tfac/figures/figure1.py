"""
This creates Figure 1 - Partial Tucker R2X and Gene Projected Factors R2X.
"""
import numpy as np
import pandas as pd
import seaborn as sns
from tensorly.metrics.regression import variance as tl_var
from tensorly.tenalg import mode_dot
from .figureCommon import subplotLabel, getSetup
from ..tensor import tensor_factor, R2X
from ..pseudoinvnorm import find_factors
from ..dataHelpers import all_data_import


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    row = 1
    col = 2
    ax, f = getSetup((8, 4), (row, col))
    proteins, RNAseq = all_data_import()
    R2X_Figure(ax, proteins, RNAseq)

    # Add subplot labels
    subplotLabel(ax)

    return f


def R2X_Figure(ax, proteinTensor, geneTensor):
    """Create Partial Tucker R2X Figure"""
    R2X_protein = np.zeros(13)
    R2X_gene = np.zeros(13)
    for i in range(1, 13):
        proteinFac, geneFac = tensor_factor(proteinTensor, geneTensor, i)
        R2X_protein[i] = R2X(mode_dot(proteinFac[0], proteinFac[1][0], 2), proteinTensor)
        R2X_gene[i] = R2X(mode_dot(geneFac[0], geneFac[1][0], 2), geneTensor)
    sns.scatterplot(np.arange(len(R2X_protein)), R2X_protein, ax=ax[0])
    ax[0].set_xlabel("Components")
    ax[0].set_ylabel("Protein R2X")
    ax[0].set_yticks([0, .2, .4, .6, .8, 1])
    sns.scatterplot(np.arange(len(R2X_gene)), R2X_gene, ax=ax[1])
    ax[1].set_xlabel("Components")
    ax[1].set_ylabel("Gene Expression R2X")
    ax[1].set_yticks([0, .2, .4, .6, .8, 1])

