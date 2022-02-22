"""
This creates Figure 1:
(a) tensor factorization cartoon
(b) tensor_svg.svg from the data folder
(c) R2X of the whole data, including gene expressions and protein levels.
"""
from .common import subplotLabel, getSetup
from tensorpack import Decomposition
from tensorpack.plot import *
from ..dataHelpers import import_LINCS_CCLE, import_LINCS_MEMA


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((8, 6), (2, 3))

    ccle, _, _ = import_LINCS_CCLE()
    # perform tensor decomposition from tensorpack with 8 components
    tc = Decomposition(ccle, max_rr=7)
    tc.perform_tfac()
    tc.perform_PCA(flattenon=2)

    tfacr2x(ax[0], tc)
    reduction(ax[1], tc)

    # mema
    mema, _, _, _ = import_LINCS_MEMA()
    tm = Decomposition(mema, max_rr=7)
    tm.perform_tfac()
    tm.perform_PCA(flattenon=2)

    tfacr2x(ax[3], tm)
    reduction(ax[4], tm)

    # Add subplot labels
    subplotLabel(ax)
    ax[0].set_title("Variance Explained by Tensor, CCLE")
    ax[3].set_title("Variance Explained by Tensor, MEMA")

    return f
