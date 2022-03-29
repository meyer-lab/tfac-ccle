""" Figure 6 for CycIF dataset. """

import pandas as pd
import numpy as np
import tensorly as tl
from tensorly.cp_tensor import cp_flip_sign
from tensorly.decomposition import parafac
from tensorpack.cmtf import cp_normalize
import seaborn as sns
import matplotlib.pyplot as plt
from ..dataHelpers import Tensor_LINCS_CycIF, reorder_table
from ..plotHelpers import plot_heatmaps
from .common import getSetup, subplotLabel


def getsetup(figsize):
    sns.set(style="whitegrid", font_scale=0.7, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})
    fig = plt.figure(figsize=figsize, constrained_layout=True)

    gs = fig.add_gridspec(3, 4)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[2, :])
    return ([ax1, ax2, ax3, ax4], fig)


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((12, 12), (4, 1))

    tensor = Tensor_LINCS_CycIF()
    fac = parafac(tensor.to_numpy(), 5, n_iter_max=2000, linesearch=True, tol=1e-9)
    fac = cp_flip_sign(fac, 2)
    fac = cp_normalize(fac)

    labels = [f"Cmp. {i}" for i in np.arange(1, fac.factors[0].shape[1] + 1)]
    print("R2X: ", 1.0 - np.linalg.norm(tl.cp_to_tensor(fac) - tensor)**2.0 / np.linalg.norm(tensor)**2.0)

    for i in [0, 1, 3]:
        facZero = pd.DataFrame(fac.factors[i], columns=labels, index=tensor.coords[tensor.dims[i]])
        facZero = reorder_table(facZero)
        g1 = sns.heatmap(facZero.T, ax=ax[i], cmap="PRGn", center=0)
        g1.set_title(tensor.dims[i])

    tm = pd.DataFrame(fac.factors[2].T, index=labels, columns=tensor.coords[tensor.dims[2]])
    tm = tm.iloc[:, np.r_[0, 1, 3, 5, 2, 4]]
    # facZero = reorder_table(tm.T)
    g1 = sns.heatmap(tm, ax=ax[2], cmap="PRGn", center=0)
    g1.set_title(tensor.dims[2])

    # Add subplot labels
    subplotLabel(ax)

    return f
