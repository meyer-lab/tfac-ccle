"""
This creates Figure 1:
(a) tensor factorization cartoon
(b) tensor_svg.svg from the data folder
(c) R2X of the whole data, including gene expressions and protein levels.
"""
from .common import subplotLabel, getSetup
from tensorpack import Decomposition
from tensorpack.plot import *
from ..dataHelpers import form_tensor


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((8, 3), (1, 3))

    tensor, _, _ = form_tensor()
    # perform tensor decomposition from tensorpack with 8 components
    t = Decomposition(tensor, max_rr=8)
    t.perform_tfac()
    t.perform_PCA(flattenon=1)

    # Set axis for pca
    ax[0].scatter(t.rrs, t.PCAR2X, s=10)
    ax[0].set_ylabel("PCA R2X")
    ax[0].set_xlabel("Number of Components")
    ax[0].set_title("Variance explained by PCA")
    ax[0].set_xticks([x for x in t.rrs])
    ax[0].set_xticklabels([x for x in t.rrs])
    ax[0].set_ylim(0, 1)
    ax[0].set_xlim(0.5, np.amax(t.rrs) + 0.5)

    tfacr2x(ax[1], t)
    reduction(ax[2], t)

    # Add subplot labels
    subplotLabel(ax)

    return f
