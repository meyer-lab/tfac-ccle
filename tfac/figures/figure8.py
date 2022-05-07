""" Plot the R2X and reduction for the integrated MEMA dataset. """

import seaborn as sns
from tensorpack import Decomposition
from .common import subplotLabel, getSetup
from tensorpack.plot import tfacr2x, reduction
from ..dataHelpers import integrate_MEMA


def makeFigure():
    # Get list of axis objects
    ax, f = getSetup((8, 4), (1, 2))

    MEMA = integrate_MEMA()
    tc = Decomposition(MEMA.to_numpy(), max_rr=12)
    tc.perform_tfac()
    tc.perform_PCA(flattenon=-3)
    tfacr2x(ax[0], tc)
    reduction(ax[1], tc)
    subplotLabel(ax)

    return f
