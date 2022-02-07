"""
This creates Figure 1:
(a) tensor factorization cartoon
(b) tensor_svg.svg from the data folder
(c) R2X of the whole data, including gene expressions and protein levels.
"""
from .common import subplotLabel, getSetup
from tensorpack import Decomposition
from matplotlib.ticker import ScalarFormatter
from tensorpack.plot import *
from ..dataHelpers import form_tensor


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((8, 3), (1, 3))

    tensor, _, _ = form_tensor()
    # perform tensor decomposition from tensorpack with 8 components
    t = Decomposition(tensor, max_rr=7)
    t.perform_tfac()

    # PCA for flattened data -- with size 6 x 3486 which is time versus treatment and expressions.
    t.perform_PCA(flattenon=1)
    ax[1].plot(np.arange(1, 7), t.PCAR2X, ".", label="PCA")
    ax[1].set_ylabel("Normalized Explained Variance")
    ax[1].set_xlabel("Number of Components")
    ax[1].set_title("Flattened Matrix Fac R2X")
    ax[1].set_ylim(bottom=0.0)
    ax[1].xaxis.set_major_formatter(ScalarFormatter())
    ax[1].legend()

    tfacr2x(ax[0], t)
    reduction(ax[2], t)

    # Add subplot labels
    subplotLabel(ax)

    return f
