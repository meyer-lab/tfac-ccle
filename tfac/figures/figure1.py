"""
This creates Figure 1 - Partial Tucker R2X and Gene Projected Factors R2X.
"""
import numpy as np
import seaborn as sns
from .figureCommon import subplotLabel, getSetup
from ..dataHelpers import form_tensor


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((8, 4), (1, 2))
    tensor, rTensor, _, _ = form_tensor()
    R2X_fig(ax[0], tensor, rTensor)

    # Add subplot labels
    subplotLabel(ax)

    return f


def R2X_fig(ax, tensor, rTensor):
    """Create Partial Tucker R2X Figure"""
    R2X = np.array([perform_CMTF(tensor, rTensor, r=i).R2X for i in range(1, 8)])
    sns.scatterplot(x=np.arange(len(R2X)), y=R2X, ax=ax)
    ax.set_xlabel("Components")
    ax.set_ylabel("R2X")
    ax.set_yticks([0, .2, .4, .6, .8, 1])
