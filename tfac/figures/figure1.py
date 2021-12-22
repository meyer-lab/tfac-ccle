"""
This creates Figure 1:
(a) tensor factorization cartoon
(b) tensor_svg.svg from the data folder
(c) R2X of the whole data, including gene expressions and protein levels.
"""
from .figureCommon import subplotLabel, getSetup
from tensorpack import Decomposition
from tensorpack.plot import *
from ..dataHelpers import form_tensor


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((8, 10), (3, 1))
    ax[0].axis("off")
    ax[1].axis("off")

    tensor, _, _ = form_tensor()
    # perform tensor decomposition from tensorpack with 8 components
    t = Decomposition(tensor, max_rr=8)
    t.perform_tfac()
    tfacr2x(ax[2], t)

    # Add subplot labels
    subplotLabel(ax)

    return f
