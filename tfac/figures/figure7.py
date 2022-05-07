""" Plot the heatmaps for the integrated MEMA dataset. """

import pandas as pd
import numpy as np
import seaborn as sns
from tensorpack import Decomposition
from .common import subplotLabel, getSetup
from ..dataHelpers import integrate_MEMA, reorder_table


def makeFigure():
    # Get list of axis objects
    ax, f = getSetup((30, 20), (4, 1))

    MEMA = integrate_MEMA()
    tc = Decomposition(MEMA.to_numpy(), max_rr=11)
    tc.perform_tfac()

    labels = [str(i) for i in np.arange(1, tc.tfac[-1].factors[0].shape[1] + 1)]

    for ii in range(MEMA.ndim):
        facZero = pd.DataFrame(tc.tfac[-1].factors[ii], columns=labels, index=MEMA.coords[MEMA.dims[ii]])
        facZero = reorder_table(facZero)
        sns.heatmap(facZero.T, ax=ax[ii], cmap="PRGn", center=0)

    subplotLabel(ax)
    return f
