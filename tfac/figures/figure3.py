"""
This creates Figure 3.
"""
import numpy as np
import pandas as pd
import seaborn as sns
from .figureCommon import subplotLabel, getSetup
"""from tensorpac.ccle import perform_CMTF"""
from ..dataHelpers import proteinNames, form_tensor


def protein_heatmap(m_axis):
   """ tensor, rTensor, _, _ = form_tensor()
    result = perform_CMTF(tensor, rTensor)

    # this takes the list of arrays from results 1, of shape 1, 295, 5 and puts it into a dataFrame of shape 295, 5 (row, col)
    protMap = pd.DataFrame(result.factors[2].T, columns=proteinNames())

    # now protMap has all of the correct names and dropped proteins // transverse to make plot look better
    sns.heatmap(protMap.T, cmap="PiYG", xticklabels=protMap.index + 1, ax=m_axis)"""


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((7, 12), (1, 1))
    """ax[0].set_title("Protein-Component Heat Plot")
    protein_heatmap(ax[0])"""
    subplotLabel(ax)
    return f
