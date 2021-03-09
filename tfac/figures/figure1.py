"""
This creates Figure 1 - Partial Tucker Decomposition Treatment and Time Plots.
"""
import numpy as np
import pandas as pd
import seaborn as sns
from .figureCommon import subplotLabel, getSetup
from ..tensor import partial_tucker_decomp, find_R2X_partialtucker, flip_factors
from ..Data_Mod import form_tensor
from ..dataHelpers import importLINCSprotein

def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    row = 1
    col = 1
    ax, f = getSetup((11, 11), (row, col))

    component = 5
    tensor, treatment_list, times = form_tensor()
    R2X_Figure_PartialTucker(ax[0], tensor)
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
    ax.set_xlabel("Rank Decomposition")
    ax.set_ylabel("R2X")
    ax.set_title("Partial Tucker Decomposition")
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
