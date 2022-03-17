"""
Figure to show each components largest weights for MEMA Data
"""
import numpy as np
import pandas as pd
import seaborn as sns
from .common import getSetup
from tensorly.decomposition import parafac
from tensorly.cp_tensor import cp_flip_sign
from tensorpack.cmtf import cp_normalize
from ..dataHelpers import import_LINCS_MEMA, reorder_table

def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((20, 12), (3, 5))

    MCF10A = import_LINCS_MEMA("mcf10a_ssc_Level4.tsv.xz")

    plot_components(MCF10A, ax)

    return f

def plot_components(tensor, ax):
    tFac = parafac(tensor.to_numpy(), 5, n_iter_max=2000, linesearch=True, tol=1e-9)
    tFac = cp_flip_sign(tFac, 2)
    tFac = cp_normalize(tFac)

    labels = [f"Cmp. {i}" for i in np.arange(1, tFac.factors[0].shape[1] + 1)]

    for ii in range(tensor.ndim):
        facZero = pd.DataFrame(tFac.factors[ii], columns=labels, index=tensor.coords[tensor.dims[ii]])
        facZero = reorder_table(facZero)

        g1 = sns.heatmap(facZero, ax=ax[ii], cmap="PRGn", center=0, vmin=-1, vmax=1)
        g1.set_title(tensor.dims[ii])
