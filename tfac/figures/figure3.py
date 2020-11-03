"""
This creates Figure 3.
"""
import numpy as np
import pandas as pd
import seaborn as sns
from .figureCommon import subplotLabel, getSetup
from ..tensor import partial_tucker_decomp, flip_factors
from ..Data_Mod import form_tensor
from ..dataHelpers import proteinNames


def protein_heatmap(m_axis):
    result = partial_tucker_decomp(form_tensor()[0], [2], rank=5)
    result = flip_factors(result)

    # this takes the list of arrays from results 1, of shape 1, 295, 5 and puts it into a dataFrame of shape 295, 5 (row, col)
    protMap = pd.DataFrame(result[1][0].T, columns=proteinNames())
    sel = np.max(np.absolute(protMap.values), axis=0)  # creates a np array of the max values of each of the proteins

    # This is the code to remove unnecessary proteins and label everything
    protMap = protMap.loc[:, sel > 0.2]

    # now protMap has all of the correct names and dropped proteins // transverse to make plot look better
    sns.heatmap(protMap.T, cmap='PiYG', vmin=-0.5, vmax=0.5, xticklabels=protMap.index + 1, ax=m_axis)


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    row = 1
    col = 1
    ax, f = getSetup((7, 12), (row, col))
    ax[0].set_title("Protein-Component Heat Plot")
    protein_heatmap(ax[0])
    subplotLabel(ax)
    return f
