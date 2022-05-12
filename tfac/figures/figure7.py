""" Plot the R2X, reduction, and heatmaps for the integrated MEMA dataset. """

import xarray as xa
import numpy as np
import tensorly as tl
from tensorpack import Decomposition, perform_CP
from .common import subplotLabel, getSetup
from tensorpack.plot import tfacr2x, reduction
from ..dataHelpers import integrate_MEMA


def makeFigure():
    # Get list of axis objects
    ax, f = getSetup((6, 3), (1, 2))

    MEMA = integrate_MEMA()
    tc = Decomposition(MEMA.to_numpy(), max_rr=12)
    tc.perform_tfac()
    tc.perform_PCA(flattenon=-3)
    tfacr2x(ax[0], tc)
    reduction(ax[1], tc)

    return f
