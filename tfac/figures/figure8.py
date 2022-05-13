""" Plot the R2X and reduction for the integrated MEMA dataset. """

import seaborn as sns
import pandas as pd
import numpy as np
from tensorpack import Decomposition
from tensorly.cp_tensor import cp_flip_sign
from .common import subplotLabel, getSetup
from tensorpack.plot import tfacr2x, reduction
from ..dataHelpers import integrate_MEMA, reorder_table


def makeFigure():
    # Get list of axis objects
    ax, f = getSetup((8, 4), (1, 3))

    MEMA = integrate_MEMA()
    tc = Decomposition(MEMA.to_numpy(), max_rr=12)
    tc.perform_tfac()
    tc.perform_PCA(flattenon=-3)
    tfacr2x(ax[0], tc)
    reduction(ax[1], tc)

    fac = tc.tfac[-2]
    fac = cp_flip_sign(fac, 2)
    labels = [str(i) for i in np.arange(1, fac.factors[0].shape[1] + 1)]

    facZero = pd.DataFrame(fac.factors[0], columns=labels, index=MEMA.coords[MEMA.dims[0]])
    facZero = reorder_table(facZero)
    sns.heatmap(facZero.T, ax=ax[2], cmap="PRGn", center=0)
    subplotLabel(ax)

    return f
