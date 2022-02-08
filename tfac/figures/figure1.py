"""
This creates Figure 1:
(a) tensor factorization cartoon
(b) tensor_svg.svg from the data folder
(c) R2X of the whole data, including gene expressions and protein levels.
"""
import numpy as np
from .common import subplotLabel, getSetup
from tensorpack import Decomposition
from tensorpack.plot import *
from ..dataHelpers import form_tensor


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((6, 3), (1, 2))

    tensor, _, _ = form_tensor()
    num_comps = 8
    # perform tensor decomposition from tensorpack with 8 components
    t = Decomposition(tensor, max_rr=num_comps+1)
    t.perform_tfac()
    tfacr2x(ax[0], t)

    # data reduction
    flat_data = np.reshape(tensor, (tensor.shape[0]*tensor.shape[1], tensor.shape[2])) # 42 x 498
    tt = Decomposition(, max_rr=num_comps+1)
    tt.perform_PCA()
    reduction(ax[1], tt)

    # Add subplot labels
    subplotLabel(ax)

    return f
