""" Plot the heatmaps for the integrated MEMA dataset. """

import pandas as pd
import numpy as np
from tensorly.cp_tensor import cp_flip_sign
import seaborn as sns
from tensorpack import Decomposition
from .common import subplotLabel, getSetup
from ..dataHelpers import integrate_MEMA, reorder_table


def makeFigure():
    # Get list of axis objects
    ax, f = getSetup((30, 25), (3, 1))

    MEMA = integrate_MEMA()
    tc = Decomposition(MEMA.to_numpy(), max_rr=11)
    tc.perform_tfac()
    fac = tc.tfac[-1]
    fac = cp_flip_sign(fac, 2)
    labels = [str(i) for i in np.arange(1, fac.factors[0].shape[1] + 1)]

    for ii in range(1, 4):
        facZero = pd.DataFrame(fac.factors[ii], columns=labels, index=MEMA.coords[MEMA.dims[ii]])
        facZero = reorder_table(facZero)
        sns.heatmap(facZero.T, ax=ax[ii-1], cmap="PRGn", center=0)

    subplotLabel(ax)
    return f
